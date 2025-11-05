import os, pathlib
from dotenv import load_dotenv

ROOT = pathlib.Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env", override=True)

OPENSEARCH_URL = os.getenv("OPENSEARCH_URL", "http://localhost:9200")
OPENSEARCH_INDEX = os.getenv("OPENSEARCH_INDEX", "law_clauses")
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-m3")
RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-large")
GEN_BACKEND = os.getenv("GEN_BACKEND", "dummy")
GEN_MODEL = os.getenv("GEN_MODEL", "gpt-4o-mini")

BM25_K = int(os.getenv("BM25_K", "50"))
DENSE_K = int(os.getenv("DENSE_K", "50"))
FINAL_K = int(os.getenv("FINAL_K", "8"))
LAMBDA_BM25 = float(os.getenv("LAMBDA_BM25", "0.5"))
LAMBDA_DENSE = float(os.getenv("LAMBDA_DENSE", "0.5"))

PROMPT_PATH = ROOT / "prompts" / "answer_system.txt"
SYSTEM_PROMPT = PROMPT_PATH.read_text(encoding="utf-8") if PROMPT_PATH.exists() else ""
