import os
import sys
import pathlib
import chromadb
from chromadb.config import Settings
from opensearchpy import OpenSearch
from dotenv import load_dotenv

# 설정 로드
ROOT = pathlib.Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env", override=True)

# Config
OPENSEARCH_URL = os.getenv("OPENSEARCH_URL", "http://localhost:9200")
OPENSEARCH_INDEX = os.getenv("OPENSEARCH_INDEX", "law_clauses")
CHROMA_DIR = ROOT / "index" / "chroma"
CHROMA_COLLECTION = "laws_bge_m3_v2"
MAPPING_FILE = ROOT / "index" / "law_enfor_mapping.json"
DOCS_FILE = ROOT / "index" / "docs.jsonl"

def check_files():
    print("=== [1] 파일 점검 ===")
    if DOCS_FILE.exists():
        count = sum(1 for _ in open(DOCS_FILE, "r", encoding="utf-8"))
        print(f"✅ docs.jsonl 존재함 (라인 수: {count}개)")
    else:
        print("❌ docs.jsonl 없음 (Retriever 작동 불가)")

    if MAPPING_FILE.exists():
        import json
        with open(MAPPING_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"✅ law_enfor_mapping.json 존재함 (매핑 키: {len(data)}개)")
    else:
        print("❌ law_enfor_mapping.json 없음 (확장 기능 작동 불가)")

def check_opensearch():
    print("\n=== [2] OpenSearch 점검 ===")
    try:
        client = OpenSearch(OPENSEARCH_URL, timeout=5)
        if not client.indices.exists(index=OPENSEARCH_INDEX):
            print(f"❌ 인덱스 '{OPENSEARCH_INDEX}'가 존재하지 않습니다.")
            return
        
        count = client.count(index=OPENSEARCH_INDEX)["count"]
        print(f"✅ 인덱스 '{OPENSEARCH_INDEX}' 연결 성공")
        print(f"📊 저장된 문서 수: {count}개")
        
        # 샘플 데이터 확인
        res = client.search(index=OPENSEARCH_INDEX, body={"size": 1})
        if res['hits']['hits']:
            sample_id = res['hits']['hits'][0]['_id']
            print(f"🔎 샘플 ID 확인: {sample_id}")
            if "|" not in sample_id:
                print("⚠️ 경고: ID 형식이 '법령명|조항' 패턴이 아닙니다. (UUID일 가능성 있음)")
            else:
                print("🆗 ID 형식이 정상입니다.")
    except Exception as e:
        print(f"❌ OpenSearch 연결 실패: {e}")

def check_chroma():
    print("\n=== [3] ChromaDB 점검 ===")
    if not CHROMA_DIR.exists():
        print(f"❌ ChromaDB 폴더가 없습니다: {CHROMA_DIR}")
        return

    try:
        client = chromadb.PersistentClient(path=str(CHROMA_DIR), settings=Settings(allow_reset=True))
        try:
            coll = client.get_collection(CHROMA_COLLECTION)
            count = coll.count()
            print(f"✅ 컬렉션 '{CHROMA_COLLECTION}' 로드 성공")
            print(f"📊 저장된 벡터 수: {count}개")
        except ValueError:
             print(f"❌ 컬렉션 '{CHROMA_COLLECTION}'을 찾을 수 없습니다.")
    except Exception as e:
        print(f"❌ ChromaDB 로드 실패: {e}")

if __name__ == "__main__":
    check_files()
    check_opensearch()
    check_chroma()