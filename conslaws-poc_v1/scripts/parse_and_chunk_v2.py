# -*- coding: utf-8 -*-
"""
parse_and_chunk_v2.py
- 주피터 노트북(V2.0)의 Context #1 생성 로직을 이식한 스크립트
- PDF 원문(건설산업기본법, 하도급법 등)을 읽어 정제된 JSONL을 생성합니다.
- 실행 전: pip install regex pdfplumber
"""

import json
import re
import regex
import pathlib
import pdfplumber
from collections import defaultdict
from typing import List, Tuple

# ---------------- Config ----------------
BASE_DIR = pathlib.Path(__file__).resolve().parents[1]
DATA_RAW_DIR = BASE_DIR / "data_raw"
DATA_PROC_DIR = BASE_DIR / "data_proc"
OUTPUT_FILE = DATA_PROC_DIR / "law_clauses.jsonl"

# 처리 대상 파일 목록 (법령명, 파일명)
# data_raw 폴더 내에 해당 PDF 파일들이 있어야 합니다.
TARGET_FILES = [
    ("하도급거래 공정화에 관한 법률", "하도급거래 공정화에 관한 법률.pdf"),
    ("하도급거래 공정화에 관한 법률 시행령", "하도급거래 공정화에 관한 법률 시행령.pdf"),
    ("건설산업기본법", "건설산업기본법.pdf"),
    ("건설산업기본법 시행령", "건설산업기본법 시행령.pdf"),
]

INCLUDE_SUPPLEMENTS = False  # 부칙 포함 여부

# ---------------- Regex Patterns & Helpers ----------------
CIRCLED = "①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳"
CIRCLED_MAP = {str(i): ch for i, ch in enumerate(CIRCLED, start=1)}

def to_circled(n: str):
    return CIRCLED_MAP.get(str(int(n))) if n and str(n).isdigit() else None

# 조문 분리 (노트북 V2 로직)
# ‘제n조’ 뒤에 곧바로 ‘제m항’이 오면(참조문) 헤더 아님
RE_ARTICLE_SPLIT = regex.compile(
    r'(?m)^(?=\s*\[?\s*제\s*\d+\s*조(?:\s*의\s*\d+)?\]?(?=\s*(?:\(|$|[,\-]))(?!\s*제\s*\d+\s*항))'
)

# 조문 헤더 추출
RE_ARTICLE_HEAD = regex.compile(
    r'^\s*\[?\s*(?P<base>제\s*\d+\s*조(?:\s*의\s*\d+)?)(?:\]?)'
    r'(?:\((?P<title>[^)]*)\))?[, \t]*(?P<body>.*)$',
    regex.S
)

# 항 시작 패턴: ①~⑳ | (3) | 제3항
RE_PARA_START = regex.compile(r'(?m)^\s*(?:([①-⑳])|\((\d+)\)|제\s*(\d+)\s*항)\s*')

# 부칙 헤더
RE_SUP_HEAD = regex.compile(r'(?m)^\s*부칙\b.*$')

# 각주/개정표기 제거용
RE_AMEND_FOOTNOTE = re.compile(r"(<[^>]*\d{4}\.\s*\d{1,2}\.\s*\d{1,2}\.>|[\[][^]]*\d{4}\.\s*\d{1,2}\.\s*\d{1,2}\.[^\]]*[\]])")

def norm_law_name(x: str) -> str:
    """법령명 정규화 (별칭 처리)"""
    s = regex.sub(r"\s+", " ", x or "").strip()
    aliases = {
        "하도급법": "하도급거래 공정화에 관한 법률",
        "건산법": "건설산업기본법",
        # 필요 시 추가
    }
    return aliases.get(s, s)

def norm_clause_id(cid: str) -> str:
    """조항 ID 정규화 (공백 제거, 표준화)"""
    if not cid: return cid
    s = regex.sub(r"\s+", "", cid)
    s = regex.sub(r"제(\d+)\s*조\s*의\s*(\d+)", r"제\1조의\2", s)  # 제13조의2
    s = regex.sub(r"제(\d+)\s*조", r"제\1조", s)
    return s

def read_text(path: pathlib.Path) -> str:
    """PDF(pdfplumber) 또는 TXT 읽기"""
    if path.suffix.lower() == ".pdf":
        texts = []
        try:
            with pdfplumber.open(path) as pdf:
                for page in pdf.pages:
                    t = page.extract_text() or ""
                    texts.append(t)
            full_text = "\n".join(texts)
            if not full_text.strip():
                raise RuntimeError("PDF 텍스트 추출 실패 (빈 내용)")
            return full_text
        except Exception as e:
            print(f"[Error] PDF 읽기 실패: {path} ({e})")
            return ""
    else:
        # TXT 읽기 (여러 인코딩 시도)
        for enc in ("utf-8", "utf-8-sig", "cp949", "euc-kr"):
            try:
                return path.read_text(encoding=enc)
            except:
                continue
        return ""

def preprocess_text(text: str) -> str:
    """노트북 V2 전처리 로직"""
    # 각주/개정표기 제거
    text = RE_AMEND_FOOTNOTE.sub(" ", text)
    
    # 붙여쓴 참조 표준화 (제9조제1항 -> 제9조 제1항)
    text = re.sub(r"(제\d+조의\d+)(제\d+항)", r"\1 \2", text)
    text = re.sub(r"(제\d+조)(제\d+항)",       r"\1 \2", text)
    text = re.sub(r"(제\d+항)(제\d+호)",       r"\1 \2", text)
    text = re.sub(r"(제\d+조의\d+)(제\d+호)", r"\1 \2", text)
    text = re.sub(r"(제\d+조)(제\d+호)",      r"\1 \2", text)
    
    # 공백 정리
    text = regex.sub(r"[ \t]+", " ", text)
    return text

def slice_main_and_supplements(full: str):
    """본문과 부칙 분리"""
    lines = full.splitlines()
    idxs = [i for i, ln in enumerate(lines) if RE_SUP_HEAD.match(ln)]
    if not idxs:
        return [("본문", full)]
    
    parts = [("본문", "\n".join(lines[:idxs[0]]).strip())]
    for j, start in enumerate(idxs):
        end = idxs[j+1] if j+1 < len(idxs) else len(lines)
        parts.append((f"부칙{j+1}", "\n".join(lines[start:end]).strip()))
    return [(name, text) for name, text in parts if text]

def parse_article(art_text: str):
    """조문 파싱 및 항 분리"""
    m = RE_ARTICLE_HEAD.match(art_text.strip())
    if m:
        clause_base = m.group("base").strip()
        title = (m.group("title") or "").strip()
        body = (m.group("body") or "").strip()
    else:
        head, _, tail = art_text.partition("\n")
        m2 = regex.search(r'(제\s*\d+\s*조(?:\s*의\s*\d+)?)', head)
        clause_base = m2.group(1) if m2 else head.strip().split()[0]
        title = ""
        body = (tail or "").strip()

    base_norm = norm_clause_id(clause_base)
    header_norm = f"{base_norm}({title})" if title else base_norm

    # 항 탐지
    spans = []
    for match in RE_PARA_START.finditer(body):
        start = match.start()
        # ① or (2) or 제3항 -> 정규화
        sym_circ = match.group(1) or to_circled(match.group(2)) or to_circled(match.group(3))
        spans.append((start, sym_circ))

    chunks = []
    if spans:
        spans.append((len(body), None))
        for (s_i, sym), (s_j, _) in zip(spans, spans[1:]):
            seg = body[s_i:s_j].strip()
            if not seg: continue
            # ID: 제1조-①
            cid = f"{base_norm}-{sym}" if sym else base_norm
            # Text: 제1조(목적) \n ① 이 법은...
            chunks.append(("항", sym or "", f"{header_norm}\n{seg}"))
    else:
        # 항 없음 -> 본문 전체
        body_clean = body.strip()
        chunks.append(("본문", "", f"{header_norm}\n{body_clean}" if body_clean else header_norm))

    return base_norm, title, chunks

def resolve_collisions_keep_longest(docs):
    """중복 조항 발생 시 내용이 가장 긴 것 보존"""
    groups = defaultdict(list)
    for d in docs:
        groups[(d["law_name"], d["clause_id"])].append(d)
    
    resolved = []
    dropped_count = 0
    for _, lst in groups.items():
        if len(lst) == 1:
            resolved.append(lst[0])
        else:
            best = max(lst, key=lambda x: len(x["text"]))
            resolved.append(best)
            dropped_count += len(lst) - 1
            
    if dropped_count > 0:
        print(f"[Info] 중복 제거됨: {dropped_count} 건 (가장 긴 텍스트 유지)")
    return resolved

# ---------------- Main Execution ----------------
def main():
    DATA_PROC_DIR.mkdir(parents=True, exist_ok=True)
    all_docs = []
    
    print(f"[Info] 파싱 시작... 대상 파일: {len(TARGET_FILES)}개")
    print(f"       소스 폴더: {DATA_RAW_DIR}")

    for law_name_raw, filename in TARGET_FILES:
        path = DATA_RAW_DIR / filename
        if not path.exists():
            print(f"[Warn] 파일 없음 (Skipping): {path}")
            continue
        
        print(f"   -> Processing: {filename} ...")
        full_raw = read_text(path)
        if not full_raw:
            continue
            
        full = preprocess_text(full_raw)
        blocks = slice_main_and_supplements(full)

        tmp_docs = []
        for block_name, text in blocks:
            if block_name.startswith("부칙") and not INCLUDE_SUPPLEMENTS:
                continue
            
            # 조 단위 Split
            arts = [a for a in RE_ARTICLE_SPLIT.split(text.strip()) if regex.search(r'제\s*\d+\s*조', a)]
            
            for art in arts:
                base, title, chunks = parse_article(art)
                law_name = norm_law_name(law_name_raw)
                
                # 부칙인 경우 ID에 prefix 추가 (선택사항)
                if block_name.startswith("부칙"):
                    base = f"{block_name} {base}"
                
                for _, sym, tx in chunks:
                    cid_raw = f"{base}-{sym}" if sym else base
                    cid = norm_clause_id(cid_raw)
                    
                    # "삭제"만 있는 조항 건너뛰기
                    if re.search(r'(^|\n)\s*삭제\s*$', tx):
                        continue

                    # ID 생성: 법령명|제N조-①
                    det_id = f"{law_name}|{cid}"
                    
                    tmp_docs.append({
                        "id": det_id,
                        "doc_id": law_name,
                        "law_name": law_name,
                        "title": title,
                        "clause_id": cid,
                        "text": tx,
                        "source_path": str(filename),
                        "original_text": tx, # 호환성 필드
                        "is_summary": False
                    })
        
        # Fallback: 만약 분리가 전혀 안되었다면(정규식 미매칭 등), 전체에서 다시 시도
        if not tmp_docs and blocks:
            print(f"      [Fallback] 조문 분리 실패, 전체 텍스트 재시도...")
            arts = [a for a in RE_ARTICLE_SPLIT.split(full.strip()) if regex.search(r'제\s*\d+\s*조', a)]
            for art in arts:
                base, title, chunks = parse_article(art)
                law_name = norm_law_name(law_name_raw)
                for _, sym, tx in chunks:
                    cid_raw = f"{base}-{sym}" if sym else base
                    cid = norm_clause_id(cid_raw)
                    det_id = f"{law_name}|{cid}"
                    if re.search(r'(^|\n)\s*삭제\s*$', tx): continue
                    
                    tmp_docs.append({
                        "id": det_id,
                        "doc_id": law_name,
                        "law_name": law_name,
                        "title": title,
                        "clause_id": cid,
                        "text": tx,
                        "source_path": str(filename),
                        "original_text": tx,
                        "is_summary": False
                    })
        
        all_docs.extend(tmp_docs)
        print(f"      추출된 Chunk 수: {len(tmp_docs)}")

    # 중복 제거
    final_docs = resolve_collisions_keep_longest(all_docs)

    # 저장
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for d in final_docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    
    print(f"[Done] 총 {len(final_docs)}개 chunk 저장 완료 -> {OUTPUT_FILE}")

    # 간단 통계
    lens = [len(d["text"]) for d in final_docs]
    if lens:
        avg_len = sum(lens) / len(lens)
        print(f"       평균 텍스트 길이: {avg_len:.1f}자")

if __name__ == "__main__":
    main()