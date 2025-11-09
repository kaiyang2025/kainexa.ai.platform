import streamlit as st
import httpx
import json, math, time
import pandas as pd


API = st.secrets.get("API_URL", "http://localhost:8000")

st.set_page_config(page_title="건설법령 RAG POC", layout="wide")
st.title("🏗️ 건설법령 RAG POC (v2)")

# === Sidebar: 리랭크 후보폭 설정 ===
st.sidebar.markdown("### 검색/리랭크 설정")
# cand_factor: 리랭커에 태울 후보 수를 k의 몇 배로 할지 (예: 2.0이면 Top-2k를 리랭크)
cand_factor = st.sidebar.slider("리랭크 후보폭 (cand_factor × k)", 1.0, 5.0, 2.0, 0.5)
st.sidebar.caption("예) Top-k=8, cand_factor=2.0 → 상위 16개를 리랭크")

# ===================== Eval helpers (간단 메트릭 계산) ======================
def _dcg_at_k(rels, k=10):
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(rels[:k]))
def _ndcg_at_k(pred_ids, gold_ids, k=10):
    rels = [1 if pid in gold_ids else 0 for pid in pred_ids[:k]]
    idcg = 1.0  # 단일 정답 가정
    return (_dcg_at_k(rels, k) / idcg) if idcg > 0 else 0.0
def _mrr_at_k(pred_ids, gold_ids, k=10):
    for i, pid in enumerate(pred_ids[:k], start=1):
        if pid in gold_ids: return 1.0 / i
    return 0.0
def _recall_at_k(pred_ids, gold_ids, k=10):
    return 1.0 if any(pid in gold_ids for pid in pred_ids[:k]) else 0.0
def _p95(values):
    if not values: return 0.0
    s = sorted(values); idx = max(0, min(int(math.ceil(0.95*len(s))) - 1, len(s)-1))
    return s[idx]
def _load_eval_jsonl(uploaded_file):
    items = []
    for raw in uploaded_file:
        line = raw.decode("utf-8") if isinstance(raw,(bytes,bytearray)) else raw
        if not line.strip(): continue
        row = json.loads(line)
        q = row.get("query"); gold_ids = row.get("gold_ids"); gold_id = row.get("gold_id")
        if gold_ids is None: gold_ids = [gold_id] if gold_id else []
        if q: items.append({"query": q, "gold_ids": [g for g in gold_ids if g]})
    return items
# ========================================================================

# =========================== Tabs: 검색 / 평가 ===========================
tab_search, tab_eval = st.tabs(["검색", "평가(Eval)"])

with tab_search:
    q = st.text_input("질문을 입력하세요", value="발주자의 공사대금 지급보증 의무는?")
    c1, c2, c3, c4 = st.columns(4)
    with c1: k = st.number_input("Top-K", min_value=3, max_value=20, value=8, step=1)
    with c2: rerank = st.checkbox("Rerank 사용", value=True)
    with c3: backend = st.selectbox("생성 백엔드", ["openai","dummy"], index=1)
    with c4: model = st.text_input("모델", value="gpt-4o-mini")

    if st.button("검색/답변 실행", use_container_width=True):
        with st.spinner("검색 중..."):
            sr = httpx.get(
                f"{API}/search",
                params={"q": q, "k": k, "rerank": str(rerank).lower(), "cand_factor": cand_factor},
                timeout=120,
            ).json()
        st.subheader("🔎 검색 결과 (상위)")
        for i, h in enumerate(sr["results"], 1):
            with st.expander(f"{i}. [{h['law_name']}] {h['clause_id']} — {h.get('title','')[:40]}"):
                st.write(h["text"])

        with st.spinner("생성 중..."):
            ar = httpx.post(
                f"{API}/answer",
                json={
                    "query": q, "k": k, "rerank": rerank, "include_context": True,
                    "gen_backend": backend, "gen_model": model, "cand_factor": cand_factor,
                },
                timeout=180,
            ).json()
        st.subheader("🧠 답변")
        st.write(ar["answer"])
        st.caption("근거 인용: " + ", ".join([f"[{c['law']} {c['clause_id']}]" for c in ar["citations"]]))
        st.divider()
        st.subheader("📚 컨텍스트")
        for i, c in enumerate(ar.get("contexts", []), 1):
            with st.expander(f"{i}. [{c['law_name']}] {c['clause_id']} — {c.get('title','')[:40]}"):
                st.write(c["text"])

with tab_eval:
    st.subheader("RAG 검색 품질/지연 평가")
    st.caption("eval.jsonl 업로드 → nDCG@10 / MRR@10 / Recall@10 / P95 latency 계산")
    up = st.file_uploader("평가셋 파일 업로드 (eval.jsonl)", type=["jsonl"])
    ec1, ec2, ec3 = st.columns(3)
    with ec1: ek = st.number_input("Top-k", min_value=1, max_value=50, value=10, step=1)
    with ec2: ererank = st.checkbox("리랭크 사용", value=True)
    with ec3: warmup = st.number_input("워밍업 쿼리 수", min_value=0, max_value=10, value=2, step=1)
    if st.button("평가 실행", use_container_width=True):
        if not up:
            st.warning("eval.jsonl 파일을 업로드해 주세요.")
        else:
            items = _load_eval_jsonl(up)
            if not items:
                st.error("유효한 항목이 없습니다. 각 줄은 {'query':..., 'gold_id' 또는 'gold_ids': [...]} 형식이어야 합니다.")
            else:
                # 워밍업 호출로 지연 안정화
                for it in items[:int(warmup)]:
                    try:
                        httpx.get(f"{API}/search",
                                  params={"q": it['query'], "k": ek, "rerank": str(ererank).lower(), "cand_factor": cand_factor},
                                  timeout=60)
                    except Exception:
                        pass
                rows, lats = [], []
                with st.spinner("평가 중..."):
                    for it in items:
                        t0 = time.perf_counter()
                        r = httpx.get(
                            f"{API}/search",
                            params={"q": it['query'], "k": ek, "rerank": str(ererank).lower(), "cand_factor": cand_factor},
                            timeout=120,
                        ).json()
                        ms = (time.perf_counter() - t0) * 1000.0
                        lats.append(ms)
                        pred_ids = [h["id"] for h in r.get("results", [])]
                        rows.append({
                            "query": it["query"],
                            "gold_ids": ", ".join(it["gold_ids"]),
                            f"nDCG@{int(ek)}": _ndcg_at_k(pred_ids, it["gold_ids"], k=int(ek)),
                            f"MRR@{int(ek)}":  _mrr_at_k(pred_ids, it["gold_ids"], k=int(ek)),
                            f"Recall@{int(ek)}": _recall_at_k(pred_ids, it["gold_ids"], k=int(ek)),
                            "latency_ms": ms,
                            "pred_ids": ", ".join(pred_ids),
                        })
                # 요약 메트릭
                def _avg(xs): return sum(xs)/len(xs) if xs else 0.0
                df = pd.DataFrame(rows)
                ndcg_avg = float(_avg(df[f"nDCG@{int(ek)}"].tolist()))
                mrr_avg  = float(_avg(df[f"MRR@{int(ek)}"].tolist()))
                rec_avg  = float(_avg(df[f"Recall@{int(ek)}"].tolist()))
                p95_ms   = float(_p95(lats))
                avg_ms   = float(_avg(lats))
                m1,m2,m3,m4,m5 = st.columns(5)
                m1.metric(f"nDCG@{int(ek)}", f"{ndcg_avg:.4f}")
                m2.metric(f"MRR@{int(ek)}",  f"{mrr_avg:.4f}")
                m3.metric(f"Recall@{int(ek)}", f"{rec_avg:.4f}")
                m4.metric("P95 Latency (ms)", f"{p95_ms:.1f}")
                m5.metric("Avg Latency (ms)", f"{avg_ms:.1f}")
                # 상세 테이블 & 지연 차트
                st.markdown("#### 개별 질의별 결과")
                st.dataframe(df, use_container_width=True, height=380)
                st.markdown("#### 지연(밀리초) 분포")
                st.bar_chart(pd.DataFrame({"latency_ms": df["latency_ms"]}))

    with st.expander("지표 설명"):
        st.markdown(f"""
- **nDCG@{int(ek)}**: 상위 {int(ek)}개 순위에서 정답이 얼마나 위에 배치되었는지(로그 할인). 1.0에 가까울수록 좋음.
- **MRR@{int(ek)}**: 첫 정답의 역순위 평균(1등=1.0, 5등=0.2). 정답을 몇 번째에서 찾는지의 직관 지표.
- **Recall@{int(ek)}**: 상위 {int(ek)}개 안에 정답이 한 번이라도 포함되었는지(회수율).
- **P95 latency**: 전체 요청 중 95%가 이 시간 이내에 끝났음을 의미(최악에 가까운 지연).
        """)
# =======================================================================
