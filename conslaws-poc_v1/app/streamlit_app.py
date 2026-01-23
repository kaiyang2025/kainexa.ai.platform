# -*- coding: utf-8 -*-
"""
streamlit_app.py
- 보안 강화: 로그인 전 사이드바 숨김 (main_app 내부로 이동)
- 설정 고도화: Final Top-k 와 Retrieval Top-k 분리
- 모델 기본값 변경: openai/gpt-oss-120b
"""
from __future__ import annotations

import os
import time
import httpx
import streamlit as st
import pandas as pd

# ---------------------------- 1. 설정 및 로그인 정보 ----------------------------
# [보안] 사용자 정보 (요청하신 최신 정보 반영)
ADMIN_USER = "kangwon"
ADMIN_PASS = "kangwon2026!"

API_DEFAULT = os.environ.get("API_URL", "http://localhost:8000")
API = st.secrets.get("API_URL", API_DEFAULT)

# 페이지 설정 (최상단)
st.set_page_config(
    page_title="건설 법령 Copilot",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="collapsed" # 초기 상태 닫힘
)

# ---------------------------- 2. 유틸 함수 ----------------------------
def call_full_process(query: str, k: int, rerank: bool, cand_factor: float, backend: str, model: str):
    try:
        payload = {
            "query": query, 
            "k": k, 
            "rerank": rerank, 
            "include_context": True,
            "gen_backend": backend, 
            "gen_model": model, 
            "cand_factor": cand_factor,
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

# ---------------------------- 3. 로그인 화면 로직 ----------------------------
def login():
    # 로그인 화면 CSS
    st.markdown("""
    <style>
        .login-container { margin-top: 100px; padding: 40px; border-radius: 10px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); background-color: white; text-align: center;}
    </style>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<h1 style='text-align: center;'>🔐 강원대학교 Access</h1>", unsafe_allow_html=True)
        st.info("관계자 외 접근이 제한된 시스템입니다.")
        
        with st.form("login_form"):
            username = st.text_input("아이디", placeholder="ID를 입력하세요")
            password = st.text_input("비밀번호", type="password", placeholder="Password를 입력하세요")
            submit_login = st.form_submit_button("로그인", type="primary", use_container_width=True)
        
        if submit_login:
            if username == ADMIN_USER and password == ADMIN_PASS:
                st.session_state['logged_in'] = True
                st.success("로그인 성공! 잠시만 기다려주세요...")
                time.sleep(0.5)
                st.rerun() # 새로고침 -> main_app() 진입
            else:
                st.error("아이디 또는 비밀번호가 올바르지 않습니다.")

# ---------------------------- 4. 메인 앱 실행 로직 (로그인 후) ----------------------------
def main_app():
    # 커스텀 CSS
    st.markdown("""
    <style>
        .main-title { font-size: 2.5rem; font-weight: 700; color: #2C3E50; margin-bottom: 0.5rem; }
        .sub-title { font-size: 1.2rem; color: #7F8C8D; margin-bottom: 2rem; }
        .answer-box { background-color: #ffffff; border-left: 5px solid #27ae60; padding: 20px; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.05); }
        section[data-testid="stSidebar"] { background-color: #f0f2f6; }
    </style>
    """, unsafe_allow_html=True)

    # ★★★ 사이드바 로직을 main_app 함수 내부로 이동 (핵심 수정) ★★★
    with st.sidebar:
        st.header("⚙️ 관리자 설정")
        if st.button("로그아웃", use_container_width=True):
            st.session_state['logged_in'] = False
            st.rerun()
            
        with st.expander("검색/모델 옵션", expanded=True):
            st.subheader("1. 검색 개수 설정")
            
            # [1] 최종 답변에 사용할 문서 개수 (Final Top-k)
            k_val = st.slider(
                "최종 결과 (Final Top-k)", 
                min_value=3, max_value=10, value=5, 
                help="LLM에게 전달할 최종 문서의 개수입니다."
            )
            
            # [2] 1차 검색(Retrieval) 개수 설정
            default_retrieval = k_val * 4
            retrieval_k = st.slider(
                "1차 검색 (Retrieval Top-k)", 
                min_value=10, max_value=50, value=default_retrieval, step=5,
                help="BM25와 벡터 검색이 각각 가져올 후보 문서의 개수입니다."
            )
            
            # [내부 로직] cand_factor 자동 계산
            if k_val > 0:
                cand_factor_val = retrieval_k / (k_val * 2)
            else:
                cand_factor_val = 2.0
                
            st.caption(f"👉 BM25: {retrieval_k}개 / Dense: {retrieval_k}개")
            st.caption(f"👉 Reranker 입력: {retrieval_k * 2}개 ➡ 출력: {k_val}개")

            st.divider()
            
            st.subheader("2. 모델 설정")
            rerank_val = st.checkbox("리랭크(Re-rank) 적용", value=True)
            gen_backend = st.selectbox("생성 백엔드", ["custom", "dummy"], index=0)
            # 모델명 기본값 수정
            gen_model = st.text_input("모델명", value="openai/gpt-oss-120b")

    # --- 메인 화면 구성 ---
    st.markdown('<div class="main-title">🏗️ 건설 법령 Copilot </div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">건설산업기본법 및 하도급법 관련 질문을 입력하세요.</div>', unsafe_allow_html=True)

    with st.form("search_form"):
        col1, col2 = st.columns([5, 1])
        with col1:
            query = st.text_input("질문 입력", placeholder="예) 하도급대금 직접지급 요건은?", label_visibility="collapsed")
        with col2:
            submit = st.form_submit_button("🔍 검색", type="primary", use_container_width=True)

    if submit:
        if not query:
            st.warning("⚠️ 질문을 입력한 후 검색 버튼을 눌러주세요.")
        else:
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
                
                # --- 근거 법령 표시 (Top 5) ---
                st.subheader("📚 근거 법령")
                if not contexts:
                    st.info("참조된 법령이 없습니다.")
                else:
                    # 화면 표시 개수: k_val 설정값만큼 보여줌 (최대 5개~10개)
                    top_contexts = contexts[:k_val]
                    cols = st.columns(2)
                    
                    for i, ctx in enumerate(top_contexts):
                        law = ctx.get('law_name', '법령')
                        clause = ctx.get('clause_id', '')
                        title = ctx.get('title', '')
                        text = ctx.get('text') or "" 
                        score = ctx.get('fused_score', ctx.get('score', 0))
                        short_text = text[:150] + "..." if len(text) > 150 else text
                        
                        with cols[i % 2]:
                            with st.container(border=True):
                                st.markdown(f"**📄 {law} {clause}**")
                                if title: st.caption(f"_{title}_")
                                st.markdown(f"{short_text}")
                                st.caption(f"유사도: {score:.4f}")

                with st.expander("🧐 전체 문맥 상세보기"):
                    if contexts:
                        df = pd.DataFrame(contexts)
                        # 컬럼 존재 여부 확인 후 선택
                        display_cols = ["law_name", "clause_id", "title", "fused_score", "text"]
                        final_cols = [c for c in display_cols if c in df.columns]
                        st.dataframe(df[final_cols], use_container_width=True, hide_index=True)
                    else:
                        st.write("데이터 없음")

# ---------------------------- 5. 앱 실행 진입점 ----------------------------
# 세션 초기화
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

# 상태 분기
if not st.session_state['logged_in']:
    login()
else:
    main_app()