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

def run_eval(items, retriever, k: int, rerank_flag: bool, warmup: int, label: str):
    """
    주어진 rerank 설정으로 한 번 평가를 수행하고 결과를 출력한다.
    """
    # 워밍업(지연 안정화)
    for it in items[:warmup]:
        try:
            retriever.search(it["query"], rerank=rerank_flag, k=k)
        except Exception:
            pass

    ndcgs, mrrs, recalls, latencies = [], [], [], []

    for it in items:
        q = it["query"]
        gold_ids = it["gold_ids"]

        t0 = time.perf_counter()
        results = retriever.search(q, rerank=rerank_flag, k=k)
        dt = (time.perf_counter() - t0) * 1000.0  # ms
        latencies.append(dt)

        pred_ids = [r["id"] for r in results]
        ndcgs.append(ndcg_at_k(pred_ids, gold_ids, k=k))
        mrrs.append(mrr_at_k(pred_ids, gold_ids, k=k))
        recalls.append(recall_at_k(pred_ids, gold_ids, k=k))

    def avg(xs): return sum(xs)/len(xs) if xs else 0.0

    print(f"\n=== {label} ===")
    print(f"Samples        : {len(items)}")
    print(f"Top-k          : {k}")
    print(f"Rerank         : {rerank_flag}")
    print(f"nDCG@{k:>2}      : {avg(ndcgs):.4f}")
    print(f"MRR@{k:>2}       : {avg(mrrs):.4f}")
    print(f"Recall@{k:>2}    : {avg(recalls):.4f}")
    print(f"P95 latency(ms): {p95(latencies):.1f}")
    print(f"Avg latency(ms): {avg(latencies):.1f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", type=str, required=True, help="평가셋 jsonl 경로")
    parser.add_argument("--k", type=int, default=10, help="Top-k")
    parser.add_argument("--warmup", type=int, default=2, help="워밍업 질의 수(지연 안정용)")
    parser.add_argument(
        "--mode",
        type=str,
        default="both",
        choices=["light", "full", "both", "single"],
        help="light: CE OFF, full: CE ON, both: 둘 다, single: --rerank 플래그 그대로 사용",
    )
    parser.add_argument(
        "--rerank",
        type=str,
        default="true",
        help="리랭크 사용 여부(true/false, --mode=single일 때만 사용)",
    )
    args = parser.parse_args()

    # 프로젝트 모듈 로드
    sys.path.insert(0, os.getcwd())
    import importlib
    cfg = importlib.import_module("config")
    su = importlib.import_module("search_utils")

    # Retriever 초기화 (기본 설정 사용)
    # 필요 시 cfg를 참고해 명시적으로 초기화할 수도 있음
    retriever = su.Retriever()

    items = load_eval(Path(args.eval))
    if not items:
        print("No eval items.")
        return

    # 실행 모드에 따라 평가
    mode = args.mode.lower()
    if mode == "both":
        # 라이트 모드: BM25 + Dense, CE OFF
        run_eval(
            items,
            retriever,
            k=args.k,
            rerank_flag=False,
            warmup=args.warmup,
            label="라이트 모드 평가 (BM25 + Dense, CE OFF)",
        )
        # 풀 모드: CE ON
        run_eval(
            items,
            retriever,
            k=args.k,
            rerank_flag=True,
            warmup=args.warmup,
            label="풀 모드 평가 (CE ON)",
        )
    elif mode == "light":
        run_eval(
            items,
            retriever,
            k=args.k,
            rerank_flag=False,
            warmup=args.warmup,
            label="라이트 모드 평가 (BM25 + Dense, CE OFF)",
        )
    elif mode == "full":
        run_eval(
            items,
            retriever,
            k=args.k,
            rerank_flag=True,
            warmup=args.warmup,
            label="풀 모드 평가 (CE ON)",
        )
    else:  # single
        rerank_flag = str(args.rerank).lower() in ["1", "true", "yes", "y"]
        run_eval(
            items,
            retriever,
            k=args.k,
            rerank_flag=rerank_flag,
            warmup=args.warmup,
            label=f"단일 모드 평가 (rerank={rerank_flag})",
        )

if __name__ == "__main__":
    main()
