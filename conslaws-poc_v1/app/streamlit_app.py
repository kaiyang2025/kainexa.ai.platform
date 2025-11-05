import streamlit as st
import httpx

API = st.secrets.get("API_URL", "http://localhost:8000")

st.set_page_config(page_title="건설법령 RAG POC", layout="wide")
st.title("🏗️ 건설법령 RAG POC (v2)")

q = st.text_input("질문을 입력하세요", value="발주자의 공사대금 지급보증 의무는?")
c1, c2, c3, c4 = st.columns(4)
with c1: k = st.number_input("Top-K", min_value=3, max_value=20, value=8, step=1)
with c2: rerank = st.checkbox("Rerank 사용", value=True)
with c3: backend = st.selectbox("생성 백엔드", ["openai","dummy"], index=1)
with c4: model = st.text_input("모델", value="gpt-4o-mini")

if st.button("검색/답변 실행", use_container_width=True):
    with st.spinner("검색 중..."):
        sr = httpx.get(f"{API}/search", params={"q": q, "k": k, "rerank": str(rerank).lower()}, timeout=120).json()
    st.subheader("🔎 검색 결과 (상위)")
    for i, h in enumerate(sr["results"], 1):
        with st.expander(f"{i}. [{h['law_name']}] {h['clause_id']} — {h.get('title','')[:40]}"):
            st.write(h["text"])

    with st.spinner("생성 중..."):
        ar = httpx.post(f"{API}/answer", json={"query": q, "k": k, "rerank": rerank, "include_context": True, "gen_backend": backend, "gen_model": model}, timeout=180).json()

    st.subheader("🧠 답변")
    st.write(ar["answer"])
    st.caption("근거 인용: " + ", ".join([f"[{c['law']} {c['clause_id']}]" for c in ar["citations"]]))

    st.divider()
    st.subheader("📚 컨텍스트")
    for i, c in enumerate(ar.get("contexts", []), 1):
        with st.expander(f"{i}. [{c['law_name']}] {c['clause_id']} — {c.get('title','')[:40]}"):
            st.write(c["text"])
