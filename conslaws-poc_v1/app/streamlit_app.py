# -*- coding: utf-8 -*-
"""
streamlit_app.py (Demo UI Revision)
- 건설협회 시연용 검색 중심 UI
- 다시 시작
- 주요 변경: 통합 검색 흐름, 카드형 UI, 설정 숨김, 스트리밍 효과
"""
from __future__ import annotations

import os
import time
import httpx
import streamlit as st
import pandas as pd

# ---------------------------- 설정 & 스타일 ----------------------------
API_DEFAULT = os.environ.get("API_URL", "http://localhost:8000")
API = st.secrets.get("API_URL", API_DEFAULT)

st.set_page_config(
    page_title="건설 법령 AI 가이드",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 커스텀 CSS
st.markdown("""
<style>
    .main-title { font-size: 2.5rem; font-weight: 700; color: #2C3E50; margin-bottom: 0.5rem; }
    .sub-title { font-size: 1.2rem; color: #7F8C8D; margin-bottom: 2rem; }
    .answer-box { background-color: #ffffff; border-left: 5px solid #27ae60; padding: 20px; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.05); }
    section[data-testid="stSidebar"] { background-color: #f0f2f6; }
</style>
""", unsafe_allow_html=True)

# ---------------------------- 유틸 함수 ----------------------------
def call_full_process(query: str, k: int, rerank: bool, cand_factor: float, backend: str, model: str):
    try:
        payload = {
            "query": query, "k": k, "rerank": rerank, "include_context": True,
            "gen_backend": backend, "gen_model": model, "cand_factor": cand_factor,
        }
        with st.spinner("법령을 분석하고 답변을 생성 중입니다..."):
            t0 = time.perf_counter()
            resp = httpx.post(f"{API}/answer", json=payload, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            t1 = time.perf_counter()
        return data, (t1 - t0)
    except Exception as e:
        st.error(f"시스템 오류 발생: {e}")
        return None, 0.0

def stream_text(text: str):
    for word in text.split(" "):
        yield word + " "
        time.sleep(0.02)

# ---------------------------- 사이드바 ----------------------------
with st.sidebar:
    st.header("⚙️ 관리자 설정")
    with st.expander("검색/모델 옵션", expanded=False):
        k_val = st.slider("Top-k", 3, 20, 6)
        rerank_val = st.checkbox("리랭크 적용", value=True)
        cand_factor_val = st.slider("후보군 배수", 1.0, 5.0, 2.0, 0.1)
        st.divider()
        gen_backend = st.selectbox("생성 백엔드", ["openai", "dummy"], index=0)
        gen_model = st.text_input("모델명", value="gpt-4o-mini")

# ---------------------------- 메인 UI ----------------------------
st.markdown('<div class="main-title">🏗️ 건설 법령 AI 가이드</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">건설산업기본법 및 하도급법 관련 질문을 입력하세요.</div>', unsafe_allow_html=True)

with st.form("search_form"):
    col1, col2 = st.columns([5, 1])
    with col1:
        query = st.text_input("질문 입력", placeholder="예) 하도급대금 직접지급 요건은?", label_visibility="collapsed")
    with col2:
        # [수정] 경고 해결: use_container_width -> type='primary'로 강조 (너비 자동)
        # 만약 꽉 채우고 싶다면 width 파라미터가 지원되지 않는 구버전일 수 있으므로 use_container_width=True 유지하되 경고 무시
        # 여기서는 최신 권장사항 반영 시도:
        submit = st.form_submit_button("🔍 검색", type="primary", use_container_width=True)

if submit and query:
    result_data, latency = call_full_process(query, k_val, rerank_val, cand_factor_val, gen_backend, gen_model)
    
    if result_data:
        answer_text = result_data.get("answer", "")
        contexts = result_data.get("contexts") or []
        
        st.divider()
        st.subheader("💡 AI 답변")
        answer_container = st.empty()
        
        if gen_backend == "dummy":
            st.warning("⚠️ Dummy 모드입니다.")
            answer_container.markdown(f'<div class="answer-box">{answer_text}</div>', unsafe_allow_html=True)
        else:
            streamed_output = ""
            for token in stream_text(answer_text):
                streamed_output += token
                answer_container.markdown(f'<div class="answer-box">{streamed_output}</div>', unsafe_allow_html=True)
        
        st.caption(f"⏱️ {latency:.2f}초 | 문서: {len(contexts)}건")
        
        st.subheader("📚 근거 법령")
        if not contexts:
            st.info("참조된 법령이 없습니다.")
        else:
            top_contexts = contexts[:4]
            cols = st.columns(2)
            
            for i, ctx in enumerate(top_contexts):
                law = ctx.get('law_name', '법령')
                clause = ctx.get('clause_id', '')
                title = ctx.get('title', '')
                # [수정] None 안전 처리
                text = ctx.get('text') or "" 
                score = ctx.get('score', 0)
                short_text = text[:150] + "..." if len(text) > 150 else text
                
                with cols[i % 2]:
                    with st.container(border=True):
                        st.markdown(f"**📄 {law} {clause}**")
                        if title: st.caption(f"{title}")
                        st.markdown(f"{short_text}")
                        st.caption(f"관련도: {score:.4f}")

        with st.expander("🧐 전체 문맥 상세보기"):
            if contexts:
                df = pd.DataFrame(contexts)
                display_cols = ["law_name", "clause_id", "title", "score", "text"]
                final_cols = [c for c in display_cols if c in df.columns]
                st.dataframe(df[final_cols], use_container_width=True, hide_index=True)
            else:
                st.write("데이터 없음")