import json
import re
import pathlib
from collections import defaultdict
from typing import Dict, List, Set

# ---------------- Config ----------------
# 파일 경로 설정
ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data_proc" / "law_clauses.jsonl"
OUTPUT_MAPPING_FILE = ROOT / "index" / "law_enfor_mapping.json"

def load_jsonl(path: pathlib.Path):
    """JSONL 파일 로드 제너레이터"""
    if not path.exists():
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def build_mapping_table():
    print(f"[Info] 매핑 테이블 생성 시작...")
    print(f"   - 입력 데이터: {DATA_PATH}")

    # 1. 시행령 데이터만 필터링하면서 매핑 정보 추출
    # 구조: mapping[시행령_조_ID] = {법령_조_ID1, 법령_조_ID2, ...}
    mapping_table: Dict[str, Set[str]] = defaultdict(set)
    
    # 정규식: "법 제N조" 또는 "법 제N조의M" 패턴 찾기 (노트북 로직 반영)
    # 예: "법 제2조", "법 제2조의2"
    regex_law_ref = re.compile(r"법\s*제\s*(\d+)\s*조(?:\s*의\s*(\d+))?")
    
    count_enfor_docs = 0
    count_matches = 0

    for doc in load_jsonl(DATA_PATH):
        law_name = doc.get("law_name", "")
        
        # '시행령'이 포함된 문서만 대상
        if "시행령" not in law_name:
            continue
            
        count_enfor_docs += 1
        text = doc.get("text", "")
        doc_id = doc.get("id", "")
        
        # ---------------------------------------------------------
        # [Step 1] 시행령 ID 정규화 (항 단위 -> 조 단위)
        # 예: "건설산업기본법 시행령|제10조-①" -> "건설산업기본법 시행령|제10조"
        # ---------------------------------------------------------
        if "-" in doc_id:
            article_key = doc_id.split("-")[0]
        else:
            article_key = doc_id
            
        # ---------------------------------------------------------
        # [Step 2] 본문에서 '법 제N조' 참조 찾기
        # ---------------------------------------------------------
        matches = regex_law_ref.findall(text)
        
        if matches:
            # 모법 이름 추론 (예: "건설산업기본법 시행령" -> "건설산업기본법")
            target_law_name = law_name.replace(" 시행령", "").strip()
            
            for main_num, sub_num in matches:
                # 조항 ID 재조립
                # main_num: "10", sub_num: "2" (있을 경우)
                clean_main = main_num.strip()
                if sub_num:
                    clean_sub = sub_num.strip()
                    ref_clause = f"제{clean_main}조의{clean_sub}"
                else:
                    ref_clause = f"제{clean_main}조"
                
                # 참조 대상 ID (모법)
                ref_law_id = f"{target_law_name}|{ref_clause}"
                
                # 매핑 테이블에 추가
                mapping_table[article_key].add(ref_law_id)
                count_matches += 1

    # ---------------------------------------------------------
    # [Step 3] JSON 저장 (Set -> List 변환)
    # ---------------------------------------------------------
    # 정렬하여 저장 (일관성 유지)
    final_mapping = {k: sorted(list(v)) for k, v in mapping_table.items()}
    
    # 출력 폴더 확인
    OUTPUT_MAPPING_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    with open(OUTPUT_MAPPING_FILE, "w", encoding="utf-8") as f:
        json.dump(final_mapping, f, ensure_ascii=False, indent=4)
        
    print(f"[Done] 매핑 파일 생성 완료: {OUTPUT_MAPPING_FILE}")
    print(f"   - 처리된 시행령 Chunk 수: {count_enfor_docs}")
    print(f"   - 발견된 참조(링크) 수: {count_matches}")
    print(f"   - 매핑된 조(Article) 수: {len(final_mapping)}")
    
    # 검증용 샘플 출력
    if final_mapping:
        print("\n[Sample Mapping]")
        first_key = list(final_mapping.keys())[0]
        print(f"   {first_key} -> {final_mapping[first_key]}")

if __name__ == "__main__":
    if not DATA_PATH.exists():
        print(f"[Error] 데이터 파일이 없습니다: {DATA_PATH}")
        print("먼저 'python scripts/parse_and_chunk.py'를 실행하여 데이터를 생성하세요.")
    else:
        build_mapping_table()