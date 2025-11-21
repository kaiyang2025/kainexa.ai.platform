import re, json, uuid, pathlib

DATA_DIR = pathlib.Path(__file__).resolve().parents[1] / "data_raw"
OUT_PATH = pathlib.Path(__file__).resolve().parents[1] / "data_proc" / "law_clauses.jsonl"

LAW_META = {
    "건설산업기본법": {"effective_date": "2025-08-28"},
    "하도급거래 공정화에 관한 법률": {"effective_date": "2025-10-02", "alias": "하도급법"},
}

# article split/lookahead at '제N조' (works for '제22조의2' as well)
RE_ARTICLE_SPLIT = re.compile(r'(?=제\d+조)')
# article head with optional '의N': e.g., '제22조(제목)...' or '제22조의2(제목)...'
RE_ARTICLE_HEAD = re.compile(r'^(제(?P<num>\d+조(?:의\d+)?))\((?P<title>[^)]*)\)\s*(?P<body>.*)$', re.S)
RE_PARA = re.compile(r'(?:^|\n)([①-⑳])\s*')

def guess_law_name_from_path(path: pathlib.Path):
    name = path.stem
    if "건설산업기본법_v2.0" in name:
        return "건설산업기본법"
    if "하도급법_v2.0" in name:
        return "하도급거래 공정화에 관한 법률"
    return name

def split_articles(text: str):
    parts = [p.strip() for p in RE_ARTICLE_SPLIT.split(text) if p.strip()]
    parts = [p for p in parts if p.startswith("제") and "조" in p[:10]]
    return parts

def parse_article(art_text: str):
    m = RE_ARTICLE_HEAD.match(art_text)
    if m:
        clause_base = m.group("num")     # '22조' or '22조의2' 포함 (앞의 '제'는 제외됨)
        clause_base = "제" + clause_base # '제22조' 또는 '제22조의2'
        title = m.group("title").strip()
        body = m.group("body").strip()
    else:
        head, _, tail = art_text.partition("\n")
        m2 = re.search(r'(제\d+조(?:의\d+)?)', head)
        clause_base = m2.group(1) if m2 else head.strip().split()[0]
        title = ""
        body = tail.strip() if tail else art_text

    chunks = []
    para_splits = RE_PARA.split(body)
    if len(para_splits) > 1:
        it = iter(para_splits)
        _ = next(it, "")  # prefix
        for sym, content in zip(it, it):
            chunks.append(("항", sym, content.strip()))
    else:
        chunks.append(("본문", "", body.strip()))
    return clause_base, title, chunks

#def to_records(law_name: str, effective_date: str, article_text: str, source_path: str):
#    clause_base, title, chunks = parse_article(article_text)
#    records = []
#    for kind, sym, content in chunks:
#        clause_id = clause_base
#        if kind == "항" and sym:
#            clause_id += f"-{sym}"
#        rec = {
#            "id": str(uuid.uuid4()),
#            "doc_id": f"{law_name}_{effective_date}",
#            "law_name": law_name,
#            "effective_date": effective_date,
#            "title": title,
#            "clause_id": clause_id,
#            "text": content,
#            "source_path": source_path,
#        }
#        records.append(rec)
#    return records

def to_records(law_name: str, effective_date: str, article_text: str, source_path: str):
    """
    단일 조문(article_text)을 clause(본문/항) 단위로 분해한 후
    연구용 docs.jsonl 스키마에 가깝게 레코드를 생성한다.

    - id        : '{law_name}|{clause_id}' (안정적인 키)
    - doc_id    : law_name (법령 단위 ID)
    - clause_id : '제34조', '제34조-①' 등
    - para_symbol: 항 번호 기호(①, ②, …)
    - kind      : '본문' 또는 '항'
    - char_count, word_count: 텍스트 길이 메타
    - is_deleted: '② 삭제' 등 삭제 조항 여부
    - is_supplement: 부칙 여부(기본 False, main()에서 필요 시 True로 세팅)
    """
    clause_base, title, chunks = parse_article(article_text)
    records = []

    for kind, sym, content in chunks:
        # 내용 정리
        text = (content or "").strip()
        if not text:
            continue

        # clause_id 구성
        clause_id = clause_base
        para_symbol = ""
        if kind == "항" and sym:
            para_symbol = sym  # 예: ①, ② ...
            clause_id = f"{clause_base}-{sym}"

        # '삭제' 전용 항(예: "② 삭제") 처리
        is_deleted = False
        normalized = re.sub(r"\s+", "", text)
        if "삭제" in normalized and len(normalized) <= 10:
            # 1단계에서는 아예 인덱싱하지 않고 건너뛴다.
            # 삭제 항도 남기고 싶다면 `continue` 대신 is_deleted=True만 세팅.
            is_deleted = True
            continue

        rec = {
            # 안정적인 ID: '건설산업기본법|제34조-②'
            "id": f"{law_name}|{clause_id}",
            "doc_id": law_name,
            "law_name": law_name,
            "effective_date": effective_date,
            "title": title,
            "clause_id": clause_id,
            "para_symbol": para_symbol,
            "kind": kind,
            "text": text,
            "source_path": source_path,
            "char_count": len(text),
            "word_count": len(text.split()),
            "is_deleted": is_deleted,
            "is_supplement": False,
        }
        records.append(rec)

    return records

def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with open(OUT_PATH, "w", encoding="utf-8") as out:
        for fp in DATA_DIR.glob("*_v2.0.txt"):
            txt = fp.read_text(encoding="utf-8", errors="ignore")
            law_name = guess_law_name_from_path(fp)
            eff = LAW_META.get(law_name, {}).get("effective_date", "")
            for a in split_articles(txt):
                for r in to_records(law_name, eff, a, str(fp)):
                    out.write(json.dumps(r, ensure_ascii=False) + "\n")
                    total += 1
    print(f"[parse_and_chunk] wrote {total} records -> {OUT_PATH}")

if __name__ == "__main__":
    main()
