import json, pathlib, httpx, pandas as pd
from datasets import Dataset
from ragas.metrics import answer_relevancy, faithfulness, context_precision, context_recall
from ragas import evaluate

ROOT = pathlib.Path(__file__).resolve().parents[1]
API = "http://localhost:8000"

def load_questions(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def run_eval():
    qs = load_questions(ROOT / "eval" / "questions_seed.jsonl")
    rows = []
    for q in qs:
        r = httpx.post(f"{API}/answer", json={"query": q["question"], "k": 8, "rerank": True, "include_context": True}, timeout=180)
        a = r.json()
        rows.append({
            "question": q["question"],
            "answer": a["answer"],
            "contexts": [c["text"] for c in (a.get("contexts") or [])],
            "ground_truths": ["; ".join([c['law']+' '+c['clause_id'] for c in q.get('gold_citations',[])])]
        })
    ds = Dataset.from_pandas(pd.DataFrame(rows)[["question","answer","contexts","ground_truths"]])
    result = evaluate(ds, metrics=[answer_relevancy, faithfulness, context_precision, context_recall])
    out = ROOT / "eval" / "ragas_result.json"
    out.write_text(result.to_pandas().to_json(orient="records", force_ascii=False), encoding="utf-8")
    print(f"[RAGAS] saved -> {out}")

if __name__ == "__main__":
    run_eval()
