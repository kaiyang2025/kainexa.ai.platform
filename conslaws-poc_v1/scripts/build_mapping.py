import json
import re
import pathlib
from collections import defaultdict
from typing import Dict, Set

# ---------------- Config ----------------
ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data_proc" / "law_clauses.jsonl"
OUTPUT_MAPPING_FILE = ROOT / "index" / "law_enfor_mapping.json"

def norm_clause_id(cid: str) -> str:
    """ID 정규화 (parse_and_chunk_v2와 동일한 로직 적용)"""
    if not cid: return cid
    # 공백 제거 및 포맷 통일
    s = re.sub(r"\s+", "", cid)
    s = re.sub(r"제(\d+)조의(\d+)", r"제\1조의\2", s)
    s = re.sub(r"제(\d+)조", r"제\1조", s)
    return s

def build_mapping_table():
    print(f"[Info] 매핑 테이블 생성 시작... (Source: {DATA_PATH})")
    
    if not DATA_PATH.exists():
        print(f"[Error] 데이터 파일이 없습니다. parse_and_chunk_v2.py를 먼저 실행하세요.")
        return

    mapping_table: Dict[str, Set[str]] = defaultdict(set)
    # 정규식: "법 제N조" 또는 "법 제N조의M"
    regex_law_ref = re.compile(r"법\s*제\s*(\d+)\s*조(?:\s*의\s*(\d+))?")
    
    count_matches = 0

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            doc = json.loads(line)
            
            law_name = doc.get("law_name", "")
            # 시행령만 분석 대상
            if "시행령" not in law_name:
                continue
                
            text = doc.get("text", "")
            doc_id = doc.get("id", "") # 예: "건설산업기본법 시행령|제10조-①"
            
            # ID 파싱 (항 번호 제거 -> 조 단위 키 생성)
            if "|" in doc_id:
                base_part = doc_id.split("|")[1] # 제10조-①
                clause_only = base_part.split("-")[0] # 제10조
                # 시행령 Key: "건설산업기본법 시행령|제10조"
                article_key = f"{law_name}|{norm_clause_id(clause_only)}"
            else:
                continue

            # 본문에서 '법 제N조' 참조 찾기
            matches = regex_law_ref.findall(text)
            if matches:
                target_law_name = law_name.replace(" 시행령", "").strip()
                for main, sub in matches:
                    ref_clause = f"제{main}조"
                    if sub: ref_clause += f"의{sub}"
                    
                    # 법령 Key: "건설산업기본법|제10조"
                    ref_law_id = f"{target_law_name}|{norm_clause_id(ref_clause)}"
                    
                    mapping_table[article_key].add(ref_law_id)
                    count_matches += 1

    # Set -> List 변환 및 저장
    final_mapping = {k: sorted(list(v)) for k, v in mapping_table.items()}
    
    OUTPUT_MAPPING_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_MAPPING_FILE, "w", encoding="utf-8") as f:
        json.dump(final_mapping, f, ensure_ascii=False, indent=4)
        
    print(f"[Done] 매핑 파일 생성 완료: {OUTPUT_MAPPING_FILE}")
    print(f"   - 매핑된 관계 수: {len(final_mapping)}건 (총 참조 발견: {count_matches})")

if __name__ == "__main__":
    build_mapping_table()