#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG 검색 품질/지연 평가 스크립트
- nDCG@10, MRR@10, Recall@10, P95 latency 출력
- 여러분의 search_utils.Retriever + config.py를 그대로 사용
사용법:
  python eval_rag.py --eval eval.jsonl --k 10 --rerank true
eval.jsonl 포맷(한 줄당 하나):
  {"query": "설계변경 대금조정 기준은?", "gold_id": "uuid-문서ID"}
  # gold가 복수라면 {"gold_ids": ["id1","id2", ...]} 형태도 지원
"""

import argparse, json, time, statistics, math, os, sys
from pathlib import Path

def load_eval(path: Path):
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            row = json.loads(line)
            q = row.get("query")
            gold_ids = row.get("gold_ids")
            gold_id = row.get("gold_id")
            if gold_ids is None:
                gold_ids = [gold_id] if gold_id else []
            items.append({"query": q, "gold_ids": [g for g in gold_ids if g]})
    return items

def dcg_at_k(relevances, k=10):
    """relevances: [1/0 ...] 길이 >= k 권장"""
    dcg = 0.0
    for i, rel in enumerate(relevances[:k], start=1):
        dcg += rel / math.log2(i + 1)
    return dcg

def ndcg_at_k(pred_ids, gold_ids, k=10):
    """gold_ids 중 하나라도 맞으면 rel=1"""
    rels = [1 if pid in gold_ids else 0 for pid in pred_ids[:k]]
    idcg = 1.0  # 단일 정답 가정(복수 정답도 rel=1 하나만 맞으면 1.0)
    return dcg_at_k(rels, k) / idcg if idcg > 0 else 0.0

def mrr_at_k(pred_ids, gold_ids, k=10):
    for i, pid in enumerate(pred_ids[:k], start=1):
        if pid in gold_ids:
            return 1.0 / i
    return 0.0

def recall_at_k(pred_ids, gold_ids, k=10):
    return 1.0 if any(pid in gold_ids for pid in pred_ids[:k]) else 0.0

def p95(values):
    """간단 P95 (샘플 수 충분할 때 권장)"""
    if not values: return 0.0
    values = sorted(values)
    idx = int(math.ceil(0.95 * len(values))) - 1
    idx = max(0, min(idx, len(values)-1))
    return values[idx]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", type=str, required=True, help="평가셋 jsonl 경로")
    parser.add_argument("--k", type=int, default=10, help="Top-k")
    parser.add_argument("--rerank", type=str, default="true", help="리랭크 사용 여부(true/false)")
    parser.add_argument("--warmup", type=int, default=2, help="워밍업 질의 수(지연 안정용)")
    args = parser.parse_args()

    # 프로젝트 모듈 로드
    sys.path.insert(0, os.getcwd())
    import importlib
    cfg = importlib.import_module("config")
    su = importlib.import_module("search_utils")

    rerank_flag = str(args.rerank).lower() in ["1","true","yes","y"]

    # Retriever 초기화
    retriever = su.Retriever(
        opensearch_url=getattr(cfg, "OPENSEARCH_URL", "http://localhost:9200"),
        index_name=getattr(cfg, "OPENSEARCH_INDEX", "law_clauses"),
        embed_model_name=getattr(cfg, "EMBED_MODEL", "BAAI/bge-m3"),
        faiss_dir=str(Path("index")),
        reranker_name=getattr(cfg, "RERANKER_MODEL", None),
    )

    items = load_eval(Path(args.eval))
    if not items:
        print("No eval items.")
        return

    # 워밍업(지연 안정화)
    for it in items[:args.warmup]:
        try:
            retriever.search(it["query"], rerank=rerank_flag, k=args.k)
        except Exception:
            pass

    ndcgs, mrrs, recalls, latencies = [], [], [], []

    for it in items:
        q = it["query"]
        gold_ids = it["gold_ids"]

        t0 = time.perf_counter()
        results = retriever.search(q, rerank=rerank_flag, k=args.k)
        dt = (time.perf_counter() - t0) * 1000.0  # ms
        latencies.append(dt)

        pred_ids = [r["id"] for r in results]
        ndcgs.append(ndcg_at_k(pred_ids, gold_ids, k=args.k))
        mrrs.append(mrr_at_k(pred_ids, gold_ids, k=args.k))
        recalls.append(recall_at_k(pred_ids, gold_ids, k=args.k))

    def avg(xs): return sum(xs)/len(xs) if xs else 0.0

    print("\n=== Metrics ===")
    print(f"Samples        : {len(items)}")
    print(f"Top-k          : {args.k}")
    print(f"Rerank         : {rerank_flag}")
    print(f"nDCG@{args.k:>2}      : {avg(ndcgs):.4f}")
    print(f"MRR@{args.k:>2}       : {avg(mrrs):.4f}")
    print(f"Recall@{args.k:>2}    : {avg(recalls):.4f}")
    print(f"P95 latency(ms): {p95(latencies):.1f}")
    print(f"Avg latency(ms): {avg(latencies):.1f}")

if __name__ == "__main__":
    main()
