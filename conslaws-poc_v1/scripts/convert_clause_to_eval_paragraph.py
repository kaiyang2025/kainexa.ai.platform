#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
convert_clause_to_eval_paragraph.py
- Convert eval_clause_ids(_paragraph).jsonl (with "gold_clause_ids": ["법령명|제N조(의M)-ⓧ", ...])
  into eval_ids.jsonl that the Eval tab expects (with "gold_ids": ["uuid", ...]).
- It uses index/docs.jsonl produced by your parse_and_chunk.py + build_index.py.

Usage
-----
python convert_clause_to_eval_paragraph.py \
  --src /mnt/data/eval_clause_ids_paragraph.jsonl \
  --docs /path/to/index/docs.jsonl \
  --out /mnt/data/eval_ids.jsonl
"""
import argparse, json, re
from pathlib import Path

ALIASES = {
    "하도급법": "하도급거래 공정화에 관한 법률",
    "건설기본법": "건설산업기본법",
}

CIRC = {"1":"①","2":"②","3":"③","4":"④","5":"⑤","6":"⑥","7":"⑦","8":"⑧","9":"⑨","10":"⑩",
        "11":"⑪","12":"⑫","13":"⑬","14":"⑭","15":"⑮","16":"⑯","17":"⑰","18":"⑱","19":"⑲","20":"⑳"}

def normalize_clause_id(cid: str) -> str:
    if cid is None:
        return ""
    cid = cid.strip()
    # unify spaces
    cid = re.sub(r"\s+", "", cid)
    # already in "제N조(의M)-ⓧ"
    if re.search(r"제\d+조(?:의\d+)?-[①-⑳]$", cid):
        return cid
    # handle "제29조의3-제1항" / "제29조-3-3항" / "제29조의3-1" etc.
    m = re.match(r'^(제\d+조(?:의\d+)?)(?:-(?:제)?(\d+)항?)?$', cid)
    if m:
        base, n = m.group(1), m.group(2)
        return f"{base}-{CIRC.get(n,n)}" if n else base
    return cid

def load_jsonl(p: Path):
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if line:
                yield json.loads(line)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="eval_clause_ids(_paragraph).jsonl")
    ap.add_argument("--docs", required=True, help="index/docs.jsonl produced by build_index.py")
    ap.add_argument("--out", required=True, help="output eval_ids.jsonl")
    args = ap.parse_args()

    # Build map: (law_name, clause_id) -> [doc_ids...]
    key2ids = {}
    with open(args.docs, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): 
                continue
            d = json.loads(line)
            law = d.get("law_name"); cid = d.get("clause_id"); did = d.get("id")
            if law and cid and did:
                key2ids.setdefault((law, cid), []).append(did)

    out_rows = []
    missing = 0
    total_keys = 0

    for row in load_jsonl(Path(args.src)):
        q = row.get("query") or row.get("question")
        gold_clause_ids = row.get("gold_clause_ids", [])
        gold_ids = []
        key_list_norm = []
        for g in gold_clause_ids:
            total_keys += 1
            if "|" not in g:
                continue
            law, raw_cid = g.split("|", 1)
            law = ALIASES.get(law, law).strip()
            cid = normalize_clause_id(raw_cid)
            key_list_norm.append(f"{law}|{cid}")
            ids = key2ids.get((law, cid), [])
            if not ids:
                missing += 1
            gold_ids.extend(ids)
        gold_ids = sorted(set(gold_ids))
        out_rows.append({
            "query": q,
            "gold_ids": gold_ids,
            "metadata": {"gold_clause_ids": key_list_norm}
        })

    with open(args.out, "w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[OK] wrote {args.out}")
    print(f"Total clause keys: {total_keys}, missing keys (no doc id found): {missing}")
    if missing > 0:
        print("Hint: check law_name/clause_id normalization or rebuild index/docs.jsonl.")

if __name__ == "__main__":
    main()
