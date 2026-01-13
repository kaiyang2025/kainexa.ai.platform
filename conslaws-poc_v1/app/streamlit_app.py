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

# 커스텀 CSS: 헤더 숨김, 카드 스타일링, 폰트 조정
st.markdown("""
<style>
    /* 메인 타이틀 폰트 및 여백 */
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #2C3E50;
        margin-bottom: 0.5rem;
    }
    .sub-title {
        font-size: 1.2rem;
        color: #7F8C8D;
        margin-bottom: 2rem;
    }
    /* 카드 스타일 (검색 결과) */
    .source-card {
        background-color: #F8F9FA;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #E9ECEF;
        margin-bottom: 10px;
    }
    .source-header {
        font-weight: bold;
        color: #2980B9;
        font-size: 1.05rem;
    }
    .source-text {
        font-size: 0.9rem;
        color: #2C3E50;
        margin-top: 5px;
    }
    /* 답변 박스 강조 */
    .answer-box {
        background-color: #ffffff;
        border-left: 5px solid #27ae60;
        padding: 20px;
        border-radius: 5px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    /* 사이드바 조정 */
    section[data-testid="stSidebar"] {
        background-color: #f0f2f6;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------- 유틸 함수 (API 호출) ----------------------------

def call_full_process(query: str, k: int, rerank: bool, cand_factor: float, backend: str, model: str):
    """검색과 답변을 한 번에 처리"""
    try:
        # 1. 답변 요청 (Answer API가 내부적으로 검색도 수행함)
        payload = {
            "query": query,
            "k": k,
            "rerank": rerank,
            "include_context": True, # UI 표시용
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
    """타자 치는 효과 제너레이터"""
    for word in text.split(" "):
        yield word + " "
        time.sleep(0.02)

# ---------------------------- 사이드바 (관리자용) ----------------------------
with st.sidebar:
    st.header("⚙️ 관리자 설정")
    st.caption("시연을 위한 파라미터 조정")
    
    with st.expander("검색/모델 옵션", expanded=False):
        k_val = st.slider("Top-k (참조 문서 수)", 3, 20, 6)
        rerank_val = st.checkbox("리랭크(Cross-Encoder) 적용", value=True)
        cand_factor_val = st.slider("후보군 배수 (cand_factor)", 1.0, 5.0, 2.0, 0.1)
        st.caption(f"실제 검색 후보: {int(k_val * cand_factor_val)}개 → Top {k_val}")
        
        st.divider()
        gen_backend = st.selectbox("생성 백엔드", ["openai", "dummy"], index=0)
        gen_model = st.text_input("모델명", value="gpt-4o-mini")
        
    st.info("💡 **협회 담당자 시연 모드**\n기본 설정값이 최적화되어 있습니다.")

# ---------------------------- 메인 UI ----------------------------

# 1. 헤더 영역
st.markdown('<div class="main-title">🏗️ 건설 법령 AI 가이드</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">건설산업기본법 및 하도급법 관련 질문을 입력하세요. AI가 법적 근거와 함께 답변해 드립니다.</div>', unsafe_allow_html=True)

# 2. 검색창 영역
with st.form("search_form"):
    col1, col2 = st.columns([5, 1])
    with col1:
        query = st.text_input(
            "질문 입력", 
            placeholder="예) 하도급대금 직접지급 요건은 무엇인가요?", 
            label_visibility="collapsed"
        )
    with col2:
        # 폼 제출 버튼
        submit = st.form_submit_button("🔍 검색", use_container_width=True, type="primary")

# 3. 결과 표시 영역
if submit and query:
    # API 호출
    result_data, latency = call_full_process(query, k_val, rerank_val, cand_factor_val, gen_backend, gen_model)
    
    if result_data:
        answer_text = result_data.get("answer", "")
        citations = result_data.get("citations", [])
        contexts = result_data.get("contexts") or result_data.get("used_contexts") or []
        
        st.divider()
        
        # [A] AI 답변 섹션
        st.subheader("💡 AI 답변")
        answer_container = st.empty()
        
        # 스트리밍 효과 (dummy 백엔드일 경우 텍스트가 짧을 수 있음)
        if gen_backend == "dummy":
            st.warning("⚠️ 현재 Dummy 모드입니다. OpenAI API 키를 설정하면 실제 답변이 생성됩니다.")
            answer_container.markdown(f'<div class="answer-box">{answer_text}</div>', unsafe_allow_html=True)
        else:
            # 실제 생성된 텍스트 타자 효과
            streamed_output = ""
            for token in stream_text(answer_text):
                streamed_output += token
                # 마크다운 렌더링을 위해 컨테이너 업데이트
                answer_container.markdown(f'<div class="answer-box">{streamed_output}</div>', unsafe_allow_html=True)
        
        st.caption(f"⏱️ 처리 시간: {latency:.2f}초 | 참조 문서: {len(contexts)}건")
        
        # [B] 근거 법령 섹션 (카드 UI)
        st.subheader("📚 근거 법령 및 조항")
        
        if not contexts:
            st.info("참조된 법령 문맥이 없습니다.")
        else:
            # 상위 3~4개만 카드로 보여주기
            top_contexts = contexts[:4]
            cols = st.columns(2) # 2열 배치
            
            for i, ctx in enumerate(top_contexts):
                law = ctx.get('law_name', '법령')
                clause = ctx.get('clause_id', '')
                title = ctx.get('title', '')
                text = ctx.get('text', '')
                score = ctx.get('score', 0)
                
                # 텍스트 길이 제한
                short_text = text[:150] + "..." if len(text) > 150 else text
                
                with cols[i % 2]:
                    with st.container(border=True):
                        st.markdown(f"**📄 {law} {clause}**")
                        if title:
                            st.caption(f"제목: {title}")
                        st.markdown(f"{short_text}")
                        # 점수 배지 (리랭크 점수 등)
                        st.caption(f"관련도 점수: {score:.4f}")

        # [C] 상세 근거 확인 (Expander)
        with st.expander("🧐 검색된 전체 문맥 상세보기 (전문가용)"):
            if contexts:
                df = pd.DataFrame(contexts)
                # 필요한 컬럼만 추출하여 표시
                display_cols = ["law_name", "clause_id", "title", "score", "text"]
                # 존재하는 컬럼만 필터링
                final_cols = [c for c in display_cols if c in df.columns]
                st.dataframe(
                    df[final_cols], 
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.write("데이터 없음")

# 초기 화면 가이드
elif not query:
    st.markdown("---")
    st.markdown("### 📌 추천 질문 예시")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.info("**하도급**\n하도급대금 직접지급 사유는?")
    with c2:
        st.info("**건설업**\n건설업 등록 기준은 무엇인가?")
    with c3:
        st.info("**벌칙**\n부실시공에 대한 처벌 규정은?")