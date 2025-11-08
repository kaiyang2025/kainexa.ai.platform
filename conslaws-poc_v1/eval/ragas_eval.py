# ragas_eval.py
import json, argparse, pathlib, httpx, pandas as pd
from datasets import Dataset
from ragas.metrics import answer_relevancy, faithfulness, context_precision, context_recall
from ragas import evaluate

def load_questions(path: pathlib.Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): 
                continue
            yield json.loads(line)

def main():
    ap = argparse.ArgumentParser("RAGAS E2E 평가")
    ap.add_argument("--input", type=str, required=True, help="질문 JSONL 파일 경로")
    ap.add_argument("--api", type=str, default="http://localhost:8000", help="Answer API base URL")
    ap.add_argument("--k", type=int, default=8, help="Top-k")
    ap.add_argument("--rerank", type=str, default="true", help="리랭크 사용 여부(true/false)")
    ap.add_argument("--out", type=str, default="eval/ragas_result", help="출력 prefix(확장자 자동)")
    args = ap.parse_args()

    api = args.api.rstrip("/")
    use_rerank = str(args.rerank).lower() in ["1","true","yes","y"]

    rows = []
    for q in load_questions(pathlib.Path(args.input)):
        # API 호출: 여러분 서비스의 /answer 스펙에 맞게 필요 시 키 추가
        resp = httpx.post(
            f"{api}/answer",
            json={"query": q["question"], "k": args.k, "rerank": use_rerank, "include_context": True},
            timeout=180.0,
        )
        a = resp.json()
        contexts = [c.get("text","") for c in (a.get("contexts") or [])]
        # gold_citations이 있으면 조항 문자열로 합쳐서 GT로 사용(없어도 RAGAS 동작)
        gts = []
        for c in q.get("gold_citations", []):
            gts.append(f"{c.get('law','')} {c.get('clause_id','')}".strip())
        if not gts:
            gts = [""]  # 빈 GT도 허용(메트릭 중 일부만 계산됨)

        rows.append({
            "question": q["question"],
            "answer": a.get("answer",""),
            "contexts": contexts,
            "ground_truths": ["; ".join(gts)],
        })

    df = pd.DataFrame(rows)[["question","answer","contexts","ground_truths"]]
    ds = Dataset.from_pandas(df)

    result = evaluate(ds, metrics=[answer_relevancy, faithfulness, context_precision, context_recall])
    pdf = result.to_pandas()

    out_prefix = pathlib.Path(args.out)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    # 개별 점수
    (out_prefix.with_suffix(".rows.json")).write_text(pdf.to_json(orient="records", force_ascii=False), encoding="utf-8")
    # 요약(평균)
    summary = {
        "samples": len(pdf),
        "answer_relevancy": float(pdf["answer_relevancy"].mean()),
        "faithfulness": float(pdf["faithfulness"].mean()),
        "context_precision": float(pdf["context_precision"].mean()),
        "context_recall": float(pdf["context_recall"].mean()),
        "k": args.k,
        "rerank": use_rerank,
        "api": api,
        "input": str(args.input),
    }
    (out_prefix.with_suffix(".summary.json")).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[RAGAS] rows  -> {out_prefix.with_suffix('.rows.json')}")
    print(f"[RAGAS] summary -> {out_prefix.with_suffix('.summary.json')}")

if __name__ == "__main__":
    main()
