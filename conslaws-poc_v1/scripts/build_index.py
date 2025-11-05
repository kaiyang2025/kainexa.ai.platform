import os, json, pathlib, faiss, numpy as np
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from opensearchpy import OpenSearch

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data_proc" / "law_clauses.jsonl"
INDEX_DIR = ROOT / "index"
INDEX_DIR.mkdir(parents=True, exist_ok=True)

load_dotenv(ROOT / ".env", override=True)

OPENSEARCH_URL = os.getenv("OPENSEARCH_URL", "http://localhost:9200")
OPENSEARCH_INDEX = os.getenv("OPENSEARCH_INDEX", "law_clauses")
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-m3")

def iter_records(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)

def ensure_index(client: OpenSearch, name: str):
    if client.indices.exists(index=name):
        client.indices.delete(index=name)
    body = {
        "settings": {"index": {"number_of_shards": 1, "number_of_replicas": 0}},
        "mappings": {
            "properties": {
                "law_name": {"type": "keyword"},
                "effective_date": {"type": "keyword"},
                "title": {"type": "text"},
                "clause_id": {"type": "keyword"},
                "text": {"type": "text"},
            }
        },
    }
    client.indices.create(index=name, body=body)

def main():
    rows = list(iter_records(DATA_PATH))
    if not rows:
        raise SystemExit(f"No data: {DATA_PATH}")

    client = OpenSearch(OPENSEARCH_URL, timeout=60)
    ensure_index(client, OPENSEARCH_INDEX)
    for i, r in enumerate(rows):
        client.index(index=OPENSEARCH_INDEX, id=r["id"], body=r, refresh=False)
        if (i + 1) % 500 == 0:
            client.indices.refresh(index=OPENSEARCH_INDEX)
    client.indices.refresh(index=OPENSEARCH_INDEX)
    print(f"[OpenSearch] indexed {len(rows)} docs -> {OPENSEARCH_INDEX}")

    model = SentenceTransformer(EMBED_MODEL)
    texts = [r["text"] for r in rows]
    emb = model.encode(texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
    dim = emb.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(emb)
    faiss.write_index(index, str(INDEX_DIR / "faiss.index"))

    meta = {"ids": [r["id"] for r in rows], "dim": int(dim), "embed_model": EMBED_MODEL}
    (INDEX_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    with open(INDEX_DIR / "docs.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("[FAISS] saved at", INDEX_DIR)

if __name__ == "__main__":
    main()
