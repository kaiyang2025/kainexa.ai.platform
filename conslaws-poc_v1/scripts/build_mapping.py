import json
import re
import pathlib
from collections import defaultdict
from typing import Dict, List, Set

# ---------------- Config ----------------
ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data_proc" / "law_clauses.jsonl"
OUTPUT_MAPPING_FILE = ROOT / "index" / "law_enfor_mapping.json"

def norm_clause_id(cid: str) -> str:
    """
    ID 정규화 함수
    - parse_and_chunk_v2.py 와 동일한 로직을 사용하여 ID 불일치 방지
    - 예: "제 3 조" -> "제3조", "제13조 의 2" -> "제13조의2"
    """
    if not cid: return cid
    # 공백 제거
    s = re.sub(r"\s+", "", cid)
    # 포맷 통일
    s = re.sub(r"제(\d+)조의(\d+)", r"제\1조의\2", s)
    s = re.sub(r"제(\d+)조", r"제\1조", s)
    return s

def build_mapping_table():
    print(f"[Info] 매핑 테이블 생성 시작... (Source: {DATA_PATH})")
    
    if not DATA_PATH.exists():
        print(f"[Error] 데이터 파일이 없습니다. parse_and_chunk_v2.py를 먼저 실행하세요.")
        return

    # 매핑 구조: {"시행령|제3조": ["법률|제3조", ...]}
    mapping_table: Dict[str, Set[str]] = defaultdict(set)
    
    # [핵심 수정 1] 주피터 노트북에서 제공한 정규식 적용
    # group(1): 조 번호 (예: 13)
    # group(2): '의' 뒤의 번호 (예: 2). 없으면 None
    regex_law_ref = re.compile(r"법\s*제\s*(\d+)\s*조(?:\s*의\s*(\d+))?")
    
    count_matches = 0
    count_enfor_docs = 0

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            doc = json.loads(line)
            
            law_name = doc.get("law_name", "")
            # 시행령 문서만 분석 대상
            if "시행령" not in law_name:
                continue
            
            count_enfor_docs += 1
            text = doc.get("text", "")
            doc_id = doc.get("id", "") # 예: "건설산업기본법 시행령|제10조-①"
            
            # 1. 시행령 ID 파싱: 항 번호("-①")를 제거하고 "조" 단위 키 생성
            if "|" in doc_id:
                # "법명|제10조-①" -> split -> base="제10조-①" -> split -> clause="제10조"
                try:
                    base_part = doc_id.split("|")[1] 
                    clause_only = base_part.split("-")[0] 
                    # 키 생성 시에도 정규화 적용
                    article_key = f"{law_name}|{norm_clause_id(clause_only)}"
                except IndexError:
                    continue
            else:
                continue

            # 2. 본문 텍스트에서 '법 제N조' 참조 찾기
            matches = regex_law_ref.findall(text)
            
            if matches:
                # 모법 이름 추론 ("... 시행령" -> "...")
                target_law_name = law_name.replace(" 시행령", "").strip()
                
                for main_num, sub_num in matches:
                    # [핵심 수정 2] ID 재조립 (주피터 노트북 로직 반영 + 정규화)
                    clean_main = main_num.strip()
                    
                    if sub_num:
                        clean_sub = sub_num.strip()
                        ref_clause = f"제{clean_main}조의{clean_sub}"
                    else:
                        ref_clause = f"제{clean_main}조"
                    
                    # 법령 Key 생성: "건설산업기본법|제10조" (반드시 정규화!)
                    ref_law_id = f"{target_law_name}|{norm_clause_id(ref_clause)}"
                    
                    # 매핑 테이블에 추가 (시행령 -> 법령)
                    mapping_table[article_key].add(ref_law_id)
                    count_matches += 1

    # 3. JSON 저장 (Set -> Sorted List 변환)
    final_mapping = {k: sorted(list(v)) for k, v in mapping_table.items()}
    
    OUTPUT_MAPPING_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_MAPPING_FILE, "w", encoding="utf-8") as f:
        json.dump(final_mapping, f, ensure_ascii=False, indent=4)
        
    print(f"[Done] 매핑 파일 생성 완료: {OUTPUT_MAPPING_FILE}")
    print(f"   - 분석한 시행령 조항(Chunk) 수: {count_enfor_docs}")
    print(f"   - 발견된 참조(링크) 총 횟수: {count_matches}")
    print(f"   - 생성된 매핑 키(시행령 조항) 수: {len(final_mapping)}")
    
    # 4. 검증용 샘플 출력
    if final_mapping:
        print("\n[Sample Data Check]")
        # '의'가 포함된 케이스가 있는지 우선적으로 확인
        found_sample = False
        for k, v in final_mapping.items():
            if "의" in k or any("의" in x for x in v):
                print(f"   {k}  ->  {v}")
                found_sample = True
                break
        
        # 없으면 아무거나 출력
        if not found_sample:
            first_key = list(final_mapping.keys())[0]
            print(f"   {first_key}  ->  {final_mapping[first_key]}")

if __name__ == "__main__":
    build_mapping_table()