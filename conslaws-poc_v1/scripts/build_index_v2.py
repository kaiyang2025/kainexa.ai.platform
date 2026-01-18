import os
import json
import pathlib
import time
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from opensearchpy import OpenSearch, helpers

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

def recreate_opensearch_index(client: OpenSearch, name: str):
    """기존 인덱스 삭제 후 재생성 (매핑 명시)"""
    if client.indices.exists(index=name):
        print(f"   [OpenSearch] 기존 인덱스 '{name}' 삭제 중...")
        client.indices.delete(index=name)
    
    body = {
        "settings": {
            "index": {
                "number_of_shards": 1, 
                "number_of_replicas": 0,
                "refresh_interval": "1s" # 빠른 검색 반영
            }
        },
        "mappings": {
            "properties": {
                "id": {"type": "keyword"},
                "law_name": {"type": "keyword"},
                "clause_id": {"type": "keyword"},
                "title": {"type": "text", "analyzer": "standard"},
                "text": {"type": "text", "analyzer": "standard"},
            }
        },
    }
    client.indices.create(index=name, body=body)
    print(f"   [OpenSearch] 새 인덱스 '{name}' 생성 완료")

def main():
    print("=== 인덱스 재구축 (V2: Bulk & Safe Mode) ===")
    
    # 1. 데이터 로드
    if not DATA_PATH.exists():
        print(f"[Error] 데이터 파일 없음: {DATA_PATH}")
        print("먼저 parse_and_chunk_v2.py를 실행하세요.")
        return

    rows = list(iter_records(DATA_PATH))
    total_docs = len(rows)
    print(f"[1] 데이터 로드 완료: {total_docs}건")
    
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    
    # ---------------------------------------------------------
    # [Task 2] OpenSearch 인덱싱 (Bulk 사용)
    # ---------------------------------------------------------
    print(f"\n[2] OpenSearch 인덱싱 시작...")
    client = OpenSearch(OPENSEARCH_URL, timeout=60)
    
    try:
        recreate_opensearch_index(client, OPENSEARCH_INDEX)
        
        # Bulk 데이터 생성
        actions = []
        for r in rows:
            actions.append({
                "_index": OPENSEARCH_INDEX,
                "_id": r["id"],
                "_source": r
            })
        
        # 한 번에 전송 (Chunk 단위 자동 처리)
        success, failed = helpers.bulk(client, actions, stats_only=True, chunk_size=500)
        client.indices.refresh(index=OPENSEARCH_INDEX)
        print(f"   -> 성공: {success}건, 실패: {failed}건")
        
        if success != total_docs:
            print("   ⚠️ 경고: 일부 문서가 OpenSearch에 저장되지 않았습니다!")
            
    except Exception as e:
        print(f"   ❌ OpenSearch 에러 발생: {e}")
        return

    # ---------------------------------------------------------
    # [Task 3] ChromaDB 인덱싱
    # ---------------------------------------------------------
    print(f"\n[3] ChromaDB 인덱싱 시작 ({EMBED_MODEL})...")
    
    try:
        model = SentenceTransformer(EMBED_MODEL)
        texts = [r["text"] for r in rows]
        
        print("   - 임베딩 생성 중 (GPU/CPU)...")
        embeddings = model.encode(
            texts, 
            batch_size=64, 
            show_progress_bar=True, 
            convert_to_numpy=True, 
            normalize_embeddings=True
        )
        
        chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR), settings=Settings(allow_reset=True))
        collection_name = "laws_bge_m3_v2"
        
        try:
            chroma_client.delete_collection(collection_name)
            print(f"   - 기존 컬렉션 '{collection_name}' 삭제됨")
        except:
            pass
        
        collection = chroma_client.create_collection(
            name=collection_name, 
            metadata={"hnsw:space": "cosine"}
        )
        
        # 데이터 준비
        ids = [str(r["id"]) for r in rows]
        metadatas = []
        for r in rows:
            metadatas.append({
                "law_name": r.get("law_name") or "",
                "clause_id": r.get("clause_id") or "",
                "title": r.get("title") or ""
            })
        
        # 배치 업로드
        batch_size = 5000
        for i in range(0, total_docs, batch_size):
            end = min(i + batch_size, total_docs)
            collection.add(
                ids=ids[i:end],
                embeddings=embeddings[i:end].tolist(),
                metadatas=metadatas[i:end],
                documents=texts[i:end]
            )
            print(f"   - ChromaDB 저장 진행률: {end}/{total_docs}")
            
        print(f"   -> ChromaDB 저장 완료: {CHROMA_DIR}")

    except Exception as e:
        print(f"   ❌ ChromaDB 에러 발생: {e}")
        return

    # ---------------------------------------------------------
    # [Task 4] docs.jsonl 갱신
    # ---------------------------------------------------------
    print(f"\n[4] 검색용 메타데이터 저장 (docs.jsonl)...")
    docs_file = INDEX_DIR / "docs.jsonl"
    with open(docs_file, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            
    print("\n✅ 모든 인덱싱 작업이 완료되었습니다.")
    print(f"   - 총 문서 수: {total_docs}건")

if __name__ == "__main__":
    main()