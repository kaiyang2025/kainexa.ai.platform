# -*- coding: utf-8 -*-
"""
streamlit_app.py
- 검색/생성 데모 + 평가(Eval) 탭
- 지표: nDCG@k / MRR@k / Recall@k / P95 latency
- 개선점 반영:
  1) nDCG가 다중 정답(gold_ids) 지원
  2) 업로드 파서가 query/question, gold_ids/gold_id 모두 허용
  3) API 호출 예외 처리 강화 + dict 접근 안전화
  4) 대용량 평가 UX: 진행률/CSV 내보내기
  5) cand_factor(리랭크 후보폭) 안내
"""
from __future__ import annotations


import os
import json
import math
import time
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
import streamlit as st
import httpx

# ---------------------------- 설정 ----------------------------
API_DEFAULT = os.environ.get("API_URL", "http://localhost:8000")
API = st.secrets.get("API_URL", API_DEFAULT )

st.set_page_config(page_title="건설 법령 RAG", layout="wide")
st.title("🏗️ 건설 법령 RAG")
st.sidebar.markdown("### ⚙️ 설정")
st.sidebar.write(f"**API**: `{API}`")

# 공통 옵션(사이드바)
k = st.sidebar.slider("Top-k", min_value=3, max_value=30, value=10, step=1)
rerank = st.sidebar.checkbox("리랭크 사용(CrossEncoder)", value=True)
cand_factor = st.sidebar.slider("cand_factor (리랭크 후보폭 = k×cand_factor)", min_value=1.0, max_value=5.0, value=2.0, step=0.1)
warmup = st.sidebar.number_input("Warmup(평가 전 예열 호출 수)", min_value=0, max_value=20, value=2, step=1)
st.sidebar.caption(f"실제 리랭크 후보 수 ≈ **{int(round(k * max(1.0, cand_factor)))}**")

# (선택) 생성 백엔드/모델
with st.sidebar.expander("생성 모델 (선택)", expanded=False):
    gen_backend = st.text_input("gen_backend", value="auto")
    gen_model = st.text_input("gen_model", value="gpt-4o-mini")

# ---------------------------- 유틸 함수 ----------------------------
def _safe_get_candidates(sr: Any) -> List[Dict[str, Any]]:
    """
    다양한 형태의 /search 응답에서 결과 리스트를 최대한 유연하게 추출
    예상 형태:
      - {"results": [ {...}, ... ]}
      - {"hits": [ {...}, ... ]}
      - [ {...}, ... ]
      - {"items": [ ... ]}
    """
    if isinstance(sr, list):
        return [x for x in sr if isinstance(x, dict)]
    if isinstance(sr, dict):
        for key in ("results", "hits", "items"):
            if isinstance(sr.get(key), list):
                return [x for x in sr[key] if isinstance(x, dict)]
        # 단일 객체일 수도 있음
        if "id" in sr or "_id" in sr:
            return [sr]
    return []


def _extract_id(rec: Dict[str, Any]) -> Optional[str]:
    return rec.get("id") or rec.get("_id") or rec.get("doc_id")

# ===================== Eval helpers (간단 메트릭 계산) ======================
def _dcg_at_k(rels, k=10):
    return sum((1.0 if r else 0.0) / math.log2(i + 2) for i, r in enumerate(rels[:k]))

def _ndcg_at_k(pred_ids, gold_ids, k=10):
    if not gold_ids:
        return 0.0
    rels = [1 if pid in gold_ids else 0 for pid in pred_ids[:k]]
    dcg = _dcg_at_k(rels, k)
    m = min(len(gold_ids), k)
    ideal_rels = [1] * m + [0] * max(0, k - m)
    idcg = _dcg_at_k(ideal_rels, k)
    return (dcg / idcg) if idcg else 0.0

def _mrr_at_k(pred_ids: List[str], gold_ids: List[str], k: int = 10) -> float:
    """
    첫 정답의 역순위 평균
    """
    gold = set(gold_ids)
    for i, pid in enumerate(pred_ids[:k], 1):
        if pid in gold:
            return 1.0 / i
    return 0.0

def _recall_at_k(pred_ids: List[str], gold_ids: List[str], k: int = 10) -> float:
    if not gold_ids:
        return 0.0
    hit = len(set(pred_ids[:k]).intersection(set(gold_ids)))
    return hit / float(len(set(gold_ids)))


def _p95(values: List[float]) -> float:
    if not values:
        return 0.0
    return float(np.quantile(np.array(values, dtype=float), 0.95))


def _load_eval_jsonl(uploaded_file) -> List[Dict[str, Any]]:
    """
    업로더 파서: query/question, gold_ids/gold_id 모두 허용
    - IR용 포맷: {"query": "...", "gold_ids": ["uuid", ...]}
    - 호환: {"question": "..."} (gold_ids 비어 있으면 지표는 0)
    """
    raw = uploaded_file.read()
    text = raw.decode("utf-8", errors="ignore") if isinstance(raw, (bytes, bytearray)) else str(raw)
    items = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        q = row.get("query") or row.get("question")
        gold_ids = row.get("gold_ids")
        if gold_ids is None:
            gid = row.get("gold_id")
            gold_ids = [gid] if gid else []
        if q:
            items.append({"query": q, "gold_ids": [g for g in gold_ids if g]})
    return items
# ========================================================================
def _call_search(query: str, topk: int, rerank: bool, cand_factor: float) -> List[Dict[str, Any]]:
    try:
        resp = httpx.get(
            f"{API}/search",
            params={"q": query, "k": topk, "rerank": str(rerank).lower(), "cand_factor": cand_factor},
            timeout=120,
        )
        resp.raise_for_status()
        sr = resp.json()
    except Exception as e:
        st.error(f"검색 호출 실패: {e}")
        return []
    return _safe_get_candidates(sr)


def _call_answer(query: str, topk: int, rerank: bool, cand_factor: float, gen_backend: str, gen_model: str) -> Dict[str, Any]:
    try:
        resp = httpx.post(
            f"{API}/answer",
            json={
                "query": query,
                "k": topk,
                "rerank": rerank,
                "include_context": True,
                "gen_backend": gen_backend,
                "gen_model": gen_model,
                "cand_factor": cand_factor,
            },
            timeout=180,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        st.error(f"생성 호출 실패: {e}")
        return {}


# =========================== Tabs: 검색 / 평가 ===========================
tab_search, tab_eval = st.tabs(["🔎 검색", "📊 평가"])

# ============================ 🔎 검색 / 생성 ============================
with tab_search:
    st.subheader("검색")
    q = st.text_input("질문/검색어를 입력하세요", value="하도급대금 직접지급 요건은?")
    col1, col2 = st.columns([1, 1])

    with col1:
        if st.button("검색 실행", use_container_width=True):
            results = _call_search(q, k, rerank, cand_factor)
            if not results:
                st.warning("검색 결과가 없습니다.")
            else:
                rows = []
                for i, r in enumerate(results[:k], 1):
                    rows.append({
                        "rank": i,
                        "id": _extract_id(r),
                        "score": r.get("score"),
                        "law_name": r.get("law_name"),
                        "clause_id": r.get("clause_id"),
                        "title": r.get("title"),
                        "text": (r.get("text") or "")[:220] + ("…" if r.get("text") and len(r.get("text")) > 220 else "")
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    with col2:
        if st.button("생성 실행(답변)", type="primary", use_container_width=True):
            ar = _call_answer(q, k, rerank, cand_factor, gen_backend, gen_model)
            if not ar:
                st.warning("답변이 없습니다.")
            else:
                st.markdown("#### 답변")
                st.write(ar.get("answer", ""))

                citations = ar.get("citations", [])
                if citations:
                    st.markdown("##### 인용(법령/조문)")
                    st.caption(", ".join([f"[{c.get('law','') or c.get('law_name','')} {c.get('clause_id','')}]" for c in citations]))

                used = ar.get("used_contexts") or ar.get("contexts") or []
                if used:
                    st.markdown("##### 사용 컨텍스트(상위 3개)")
                    for i, c in enumerate(used[:3], 1):
                        st.write(f"**[{i}]** {(c.get('title') or c.get('clause_id') or '')}")
                        st.write((c.get("text") or "")[:500])

# ============================ 📊 평가(Eval) ============================
with tab_eval:
    st.subheader("평가 · IR 지표 (nDCG/MRR/Recall/P95)")
    st.caption("업로드 포맷: `{'query': '...', 'gold_ids': ['uuid1','uuid2',...]}` (또는 `question`/`gold_id`도 허용)")

    up = st.file_uploader("eval_ids.jsonl 업로드", type=["jsonl"])
    run = st.button("평가 실행", type="primary")

    if up and run:
        items = _load_eval_jsonl(up)
        if not items:
            st.error("유효한 항목이 없습니다.")
            st.stop()

        # gold 없는 항목 안내
        no_gold = [it for it in items if not it.get("gold_ids")]
        if no_gold:
            st.info(f"gold_ids가 비어있는 항목 {len(no_gold)}개가 있습니다. 해당 항목은 지표에 반영되지 않거나 0으로 계산될 수 있습니다.")

        # Warmup
        if warmup > 0:
            st.write(f"Warmup {warmup}회 진행 중…")
            for it in items[:warmup]:
                _ = _call_search(it["query"], k, rerank, cand_factor)
            st.success("Warmup 완료")

        rows = []
        n_total = len(items)
        progress = st.progress(0.0)

        latencies_ms: List[float] = []
        ndcgs: List[float] = []
        mrrs: List[float] = []
        recalls: List[float] = []

        for idx, it in enumerate(items, 1):
            q = it["query"]
            gold_ids = it.get("gold_ids", [])
            t0 = time.perf_counter()
            results = _call_search(q, k, rerank, cand_factor)
            t1 = time.perf_counter()
            elapsed_ms = (t1 - t0) * 1000.0

            pred_ids = []
            show_rows = []
            for rnk, rec in enumerate(results[:k], 1):
                pid = _extract_id(rec)
                pred_ids.append(pid)
                show_rows.append({
                    "rank": rnk,
                    "id": pid,
                    "score": rec.get("score"),
                    "law_name": rec.get("law_name"),
                    "clause_id": rec.get("clause_id"),
                    "title": rec.get("title")
                })

            # 지표
            ndcg = _ndcg_at_k(pred_ids, gold_ids, k=k) if gold_ids else 0.0
            mrr_ = _mrr_at_k(pred_ids, gold_ids, k=k) if gold_ids else 0.0
            rec_ = _recall_at_k(pred_ids, gold_ids, k=k) if gold_ids else 0.0

            latencies_ms.append(elapsed_ms)
            if gold_ids:
                ndcgs.append(ndcg)
                mrrs.append(mrr_)
                recalls.append(rec_)

            rows.append({
                "query": q,
                "gold_ids": ", ".join(gold_ids) if gold_ids else "",
                "pred_ids(topk)": ", ".join([p for p in pred_ids if p]),
                "nDCG@k": round(ndcg, 4),
                "MRR@k": round(mrr_, 4),
                "Recall@k": round(rec_, 4),
                "latency_ms": round(elapsed_ms, 1)
            })

            progress.progress(idx / max(1, n_total))

        # 집계
        mean_ndcg = float(np.mean(ndcgs)) if ndcgs else 0.0
        mean_mrr = float(np.mean(mrrs)) if mrrs else 0.0
        mean_recall = float(np.mean(recalls)) if recalls else 0.0
        p95_lat = _p95(latencies_ms)
        avg_lat = float(np.mean(latencies_ms)) if latencies_ms else 0.0

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("nDCG@k", f"{mean_ndcg:.3f}")
        c2.metric("MRR@k", f"{mean_mrr:.3f}")
        c3.metric("Recall@k", f"{mean_recall:.3f}")
        c4.metric("P95 latency (ms)", f"{p95_lat:.1f}")
        c5.metric("Avg latency (ms)", f"{avg_lat:.1f}")

        st.divider()
        st.markdown("#### 질의별 상세")
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

        # CSV 다운로드
        csv = df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "결과 CSV 다운로드",
            data=csv,
            file_name="eval_results.csv",
            mime="text/csv",
            use_container_width=True
        )

        st.caption(f"설정 요약: k={k}, rerank={rerank}, cand_factor={cand_factor} → 리랭크 후보 ≈ {int(round(k * max(1.0, cand_factor)))}개")
        n_eval = len(ndcgs)
        st.caption(f"평가 표본: {n_eval}/{n_total} (gold_ids 보유)")
        
    with st.expander("지표 설명"):
        st.markdown(f"""
- **nDCG@{int(k)}**: 상위 {int(k)}개 순위에서 정답이 얼마나 위에 배치되었는지(로그 할인). 1.0에 가까울수록 좋음.
- **MRR@{int(k)}**: 첫 정답의 역순위 평균(1등=1.0, 5등=0.2). 정답을 몇 번째에서 찾는지의 직관 지표.
- **Recall@{int(k)}**: 상위 {int(k)}개 안에 정답이 한 번이라도 포함되었는지(회수율).
- **P95 latency**: 전체 요청 중 95%가 이 시간 이내에 끝났음을 의미(최악에 가까운 지연).
        """)
# =======================================================================
