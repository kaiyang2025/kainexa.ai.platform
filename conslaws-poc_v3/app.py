# app.py
import streamlit as st
import pandas as pd
import mock_data
import time

# --- 1. 페이지 설정 및 테마 ---
st.set_page_config(page_title="Kainexa Dispute OS", layout="wide")

st.markdown("""
<style>
    /* ChatGPT 스타일 채팅창 */
    .stChatMessage { background-color: transparent !important; }
    /* 타임라인 스타일 */
    .timeline-card { border-left: 3px solid #007bff; padding-left: 15px; margin-bottom: 15px; }
    .gap-alert { color: #dc3545; font-weight: bold; background: #fff5f5; padding: 5px; border-radius: 4px; }
    /* 우측 패널 스타일 */
    .right-panel { background-color: #fcfcfc; padding: 20px; border-radius: 10px; border: 1px solid #eee; }
</style>
""", unsafe_allow_html=True)

# --- 2. 상태 관리 ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "반갑습니다. 분석할 프로젝트를 선택하거나 새로운 분쟁 건을 생성해주세요."}]
if "project_selected" not in st.session_state:
    st.session_state.project_selected = None

# --- 3. 사이드바 (프로젝트 및 문서함 관리) ---
with st.sidebar:
    st.markdown("### 🏗️ Kainexa Workspace")
    
    # 1. 프로젝트 생성 및 선택
    project_name = st.selectbox("프로젝트 선택", ["+ 새 프로젝트 생성"] + mock_data.get_project_list())
    if project_name != "+ 새 프로젝트 생성":
        st.session_state.project_selected = project_name
    
    st.markdown("---")
    
    # 2. 문서함 (공유 vs 개인)
    st.subheader("📁 Document Library")
    lib_tabs = st.tabs(["공유(Shared)", "개인(Private)"])
    
    with lib_tabs[0]:
        st.caption("계약서, 공사일지, 회의록 (FIDIC)")
        st.checkbox("FIDIC_Red_Book.pdf", value=True)
        st.checkbox("Daily_Logs_July.xlsx", value=True)
        
    with lib_tabs[1]:
        st.caption("현장 사진, 개인 메모, 미공식 기록")
        st.file_uploader("파일 추가", type=['pdf', 'jpg', 'png'])
        st.checkbox("현장_사진_0712.jpg")

# --- 4. 메인 화면 레이아웃 (2-Pane) ---
if st.session_state.project_selected:
    col_chat, col_insight = st.columns([1.2, 1])

    # --- Left: ChatGPT 스타일 대화창 ---
    with col_chat:
        st.subheader(f"💬 {st.session_state.project_selected}")
        
        # 채팅 메시지 표시
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        # 채팅 입력
        if prompt := st.chat_input("사건에 대해 물어보거나 증거 분석을 지시하세요..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)

            with st.chat_message("assistant"):
                with st.spinner("에이전트 협업 분석 중..."):
                    time.sleep(1)
                    # 시나리오 기반 답변
                    if "증거" in prompt or "확률" in prompt:
                        response = "현재 7월 15일자 증거가 누락되어 승소 확률이 65%로 제한적입니다. 개인 문서함의 '운반일지'를 추가 분석에 포함할까요?"
                    else:
                        response = f"{st.session_state.project_selected}에 대한 법리 검토를 진행 중입니다. 우측 타임라인에서 누락된 Red Flag 구간을 확인해주세요."
                    st.write(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})

    # --- Right: 가변형 Insight 패널 (Risk & Timeline) ---
    with col_insight:
        with st.container():
            st.markdown('<div class="right-panel">', unsafe_allow_html=True)
            
            # 4. Risk Dashboard
            risk = mock_data.get_risk_data(st.session_state.project_selected)
            st.subheader("🚩 Risk Dashboard")
            c1, c2 = st.columns(2)
            c1.metric("Win Probability", f"{risk['score']}%")
            c2.write(f"**Status:** {risk['status']}")
            
            if risk['missing_docs']:
                st.markdown(f'<p class="gap-alert">⚠️ 누락 증거: {", ".join(risk["missing_docs"])}</p>', unsafe_allow_html=True)

            st.markdown("---")

            # 3. Visual Evidence Timeline
            st.subheader("📍 Evidence Timeline")
            timeline = mock_data.get_timeline_data(st.session_state.project_selected)
            for item in timeline:
                color = "#dc3545" if item['status'] == "Missing" else "#28a745"
                st.markdown(f"""
                <div class="timeline-card" style="border-left-color: {color};">
                    <small>{item['date']}</small> | <b>{item['event']}</b><br>
                    <span style="font-size: 0.8rem; color: #666;">Type: {item['type']} | Status: {item['status']}</span>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("---")
            
            # 6. 최종 산출물 (Template Selector)
            st.subheader("📄 Submission Package")
            template = st.selectbox("출력 템플릿 선택", mock_data.get_templates())
            if st.button("Evidence Pack 생성 및 제출 패키징"):
                with st.status("패키징 생성 중..."):
                    time.sleep(1.5)
                    st.write("서면 초안 작성 완료")
                    st.write("증거 인덱싱(Citation) 완료")
                st.success("✅ 제출 패키지(ZIP)가 준비되었습니다.")
                st.download_button("다운로드", data="file_content", file_name="Kainexa_Submission_Pack.zip")
            
            st.markdown('</div>', unsafe_allow_html=True)
else:
    st.info("왼쪽 사이드바에서 프로젝트를 선택하거나 새로 생성하여 분석을 시작하세요.")