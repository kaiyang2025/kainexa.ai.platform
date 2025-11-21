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

#BM25_K = int(os.getenv("BM25_K", "50"))
#DENSE_K = int(os.getenv("DENSE_K", "50"))
#FINAL_K = int(os.getenv("FINAL_K", "8"))
#LAMBDA_BM25 = float(os.getenv("LAMBDA_BM25", "0.5"))
#LAMBDA_DENSE = float(os.getenv("LAMBDA_DENSE", "0.5"))

"""
검색 파이프라인 기본 하이퍼파라미터
- v0.6 노트북에서 튜닝한 값들을 기본값으로 반영
"""
BM25_K = int(os.getenv("BM25_K", "200"))      # BM25 후보 수
DENSE_K = int(os.getenv("DENSE_K", "200"))    # Dense 후보 수
FINAL_K = int(os.getenv("FINAL_K", "10"))     # 최종 Top-k

# weighted 결합 시 bm25 비중 (0.0~1.0)
LAMBDA_BM25 = float(os.getenv("LAMBDA_BM25", "0.2"))
LAMBDA_DENSE = float(os.getenv("LAMBDA_DENSE", "0.8"))

# 검색 결합 방식 / 리랭크 ON/OFF
SEARCH_METHOD = os.getenv("SEARCH_METHOD", "weighted")  # "rrf" 또는 "weighted"
USE_RERANK = os.getenv("USE_RERANK", "true").lower() == "true"

PROMPT_PATH = ROOT / "prompts" / "answer_system.txt"
SYSTEM_PROMPT = PROMPT_PATH.read_text(encoding="utf-8") if PROMPT_PATH.exists() else ""

# 리랭크 후보 수 비율(Top-k의 몇 배를 리랭크에 태울지)
#RERANK_CAND_FACTOR = 1.0   # 1.0이면 정확히 k만, 2.0이면 2k
# 리랭크 후보 수 비율(Top-k의 몇 배를 미리 가져올지)
RERANK_CAND_FACTOR = float(os.getenv("RERANK_CAND_FACTOR", "2.0"))
