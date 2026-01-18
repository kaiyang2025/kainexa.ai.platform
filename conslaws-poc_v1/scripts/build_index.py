import os
import json
import pathlib
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from opensearchpy import OpenSearch

# 경로 설정
ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data_proc" / "law_clauses.jsonl"
INDEX_DIR = ROOT / "index"
CHROMA_DIR = INDEX_DIR / "chroma"

# 설정 로드
load_dotenv(ROOT / ".env", override=True)
OPENSEARCH_URL = os.getenv("OPENSEARCH_URL", "http://localhost:9200")
OPENSEARCH_INDEX = os.getenv("OPENSEARCH_INDEX", "law_clauses")
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-m3")

def iter_records(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def ensure_opensearch_index(client: OpenSearch, name: str):
    if client.indices.exists(index=name):
        client.indices.delete(index=name)
    body = {
        "settings": {"index": {"number_of_shards": 1, "number_of_replicas": 0}},
        "mappings": {
            "properties": {
                "law_name": {"type": "keyword"},
                "clause_id": {"type": "keyword"},
                "title": {"type": "text"},
                "text": {"type": "text"},
            }
        },
    }
    client.indices.create(index=name, body=body)

def main():
    # 1. 데이터 로드
    rows = list(iter_records(DATA_PATH))
    if not rows:
        raise SystemExit(f"No data: {DATA_PATH}")
    
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    
    # ---------------------------------------------------------
    # [Task 1] OpenSearch 인덱싱 (BM25용) - 기존 유지
    # ---------------------------------------------------------
    print(f"[1/2] OpenSearch Indexing ({len(rows)} docs)...")
    client = OpenSearch(OPENSEARCH_URL, timeout=60)
    ensure_opensearch_index(client, OPENSEARCH_INDEX)
    
    for i, r in enumerate(rows):
        client.index(index=OPENSEARCH_INDEX, id=r["id"], body=r, refresh=False)
        if (i + 1) % 1000 == 0:
            print(f"   - OpenSearch indexed {i + 1}...")
    client.indices.refresh(index=OPENSEARCH_INDEX)

    # ---------------------------------------------------------
    # [Task 2] ChromaDB 인덱싱 (Dense용) - FAISS 대체
    # ---------------------------------------------------------
    print(f"[2/2] ChromaDB Indexing ({EMBED_MODEL})...")
    
    # 임베딩 모델 로드
    model = SentenceTransformer(EMBED_MODEL)
    texts = [r["text"] for r in rows]
    
    # 임베딩 생성 (Batch)
    print("   - Generating Embeddings...")
    embeddings = model.encode(
        texts, 
        batch_size=64, 
        show_progress_bar=True, 
        convert_to_numpy=True, 
        normalize_embeddings=True
    )
    
    # ChromaDB 클라이언트 초기화
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR), settings=Settings(allow_reset=True))
    collection_name = "laws_bge_m3_v2" # 노트북과 동일한 컬렉션명 사용
    
    try:
        chroma_client.delete_collection(collection_name)
    except:
        pass
    
    collection = chroma_client.create_collection(
        name=collection_name, 
        metadata={"hnsw:space": "cosine"} # 코사인 유사도 사용
    )
    
    # 데이터 준비
    ids = [str(r["id"]) for r in rows]
    documents = texts
    metadatas = []
    for r in rows:
        # ChromaDB 메타데이터는 None을 허용하지 않으므로 빈 문자열 처리
        metadatas.append({
            "law_name": r.get("law_name") or "",
            "clause_id": r.get("clause_id") or "",
            "title": r.get("title") or ""
        })
    
    # 배치 업로드 (Chroma 권장)
    batch_size = 5000
    total = len(ids)
    
    print("   - Uploading to ChromaDB...")
    for i in range(0, total, batch_size):
        end = min(i + batch_size, total)
        collection.add(
            ids=ids[i:end],
            embeddings=embeddings[i:end].tolist(),
            metadatas=metadatas[i:end],
            documents=documents[i:end]
        )
        print(f"   - Progress: {end}/{total}")
        
    print(f"[Done] ChromaDB saved at {CHROMA_DIR}")

    # docs.jsonl은 Retriever에서 빠른 조회를 위해 여전히 필요함
    with open(INDEX_DIR / "docs.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()