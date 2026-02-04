# app.py
import streamlit as st
import pandas as pd
import mock_data
import time

# --- 1. 페이지 설정 및 CSS ---
st.set_page_config(page_title="Kainexa AI Assistant", layout="wide")

# 스타일링 (Kainexa 브랜드 느낌)
st.markdown("""
<style>
    .red-flag { background-color: #ffe6e6; padding: 10px; border-left: 5px solid red; border-radius: 5px; margin-bottom: 10px; }
    .success-box { background-color: #e6ffe6; padding: 10px; border-left: 5px solid green; border-radius: 5px; }
    .citation { color: blue; font-weight: bold; cursor: pointer; }
</style>
""", unsafe_allow_html=True)

# --- 2. 상태 관리 (State Management) ---
if 'data_loaded' not in st.session_state:
    st.session_state['data_loaded'] = False # 추가 문서 로드 여부
if 'workflow_step' not in st.session_state:
    st.session_state['workflow_step'] = 'Drafting' # 승인 단계

# --- 3. 사이드바 (Nav & Mock Controls) ---
with st.sidebar:
    st.title("🏗️ Kainexa Copilot")
    st.caption("Project: Gangwon Univ. Library (Delay Claim)")
    
    st.markdown("---")
    st.subheader("📁 문서함 (Evidence Inbox)")
    st.write("✅ 계약서 (FIDIC Red Book)")
    st.write("✅ 7월 공사일지 (1~14일)")
    
    # [Mock] 파일 업로드 시늉
    uploaded = st.file_uploader("추가 문서 업로드 (누락분)", type=['pdf', 'xlsx'])
    if uploaded:
        with st.spinner("AI가 문서를 분석하여 온톨로지를 매핑 중입니다..."):
            time.sleep(1.5) # 분석하는 척 딜레이
        st.session_state['data_loaded'] = True
        st.success("✅ '7월 15~20일 작업일보' 분석 완료! (Events: 2 extracted)")

# --- 4. 메인 화면 ---

# 헤더
st.header("Construction Dispute Readiness Dashboard")
st.markdown("---")

col1, col2 = st.columns([6, 4])

with col1:
    st.subheader("📅 Master Timeline & Gap Analysis")
    
    # 데이터 가져오기
    if st.session_state['data_loaded']:
        timeline_data = mock_data.get_filled_timeline()
    else:
        timeline_data = mock_data.get_initial_timeline()
    
    df = pd.DataFrame(timeline_data)
    
    # [P0 핵심 기능] Gap Analysis 시각화
    # 7월 14일과 21일 사이가 비어있으면 경고
    has_gap = True
    if st.session_state['data_loaded']:
        has_gap = False
    
    if has_gap:
        st.markdown("""
        <div class="red-flag">
            <b>🚨 Critical Gap Detected (위험 감지)</b><br>
            2024-07-14 이후 <b>6일간의 입증 자료가 누락</b>되었습니다. 
            이 기간의 '작업일보'나 '장비가동일보'가 없으면 클레임이 기각될 확률이 85%입니다.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="success-box">
            <b>✅ Gap Resolved (공백 해결)</b><br>
            모든 기간에 대한 입증 자료(Evidence)가 연결되었습니다. Evidence Pack 생성이 가능합니다.
        </div>
        """, unsafe_allow_html=True)

    # 타임라인 테이블 보여주기
    st.dataframe(
        df, 
        column_config={
            "date": "날짜",
            "event": "이벤트(Event)",
            "source": "근거 문서(Evidence)",
            "type": st.column_config.TextColumn("유형", help="Event Type")
        },
        use_container_width=True,
        hide_index=True
    )

with col2:
    st.subheader("📄 Evidence Pack Preview")
    
    # 탭 구성
    tab1, tab2 = st.tabs(["Draft (초안)", "Approval (승인)"])
    
    with tab1:
        if has_gap:
            st.warning("⚠️ 증거가 불충분하여 초안을 생성할 수 없습니다. 타임라인의 공백을 먼저 해결해주세요.")
            st.button("초안 생성 (비활성)", disabled=True)
        else:
            st.info("💡 모든 문장에 근거(Citation)가 자동 태깅되었습니다.")
            st.markdown(mock_data.get_draft_text()) # [P0 핵심 기능] Citation 보여주기
            
            st.markdown("---")
            st.download_button("📥 PDF 다운로드 (Evidence Pack)", data="mock", file_name="Claim_Pack_v0.pdf")

    with tab2:
        # [Mock] 승인 워크플로우
        st.write(f"현재 상태: **{st.session_state['workflow_step']}**")
        
        step_col1, step_col2, step_col3 = st.columns(3)
        step_col1.markdown("✅ 작성(Author)")
        
        if st.session_state['workflow_step'] == 'Drafting':
            step_col2.markdown("⬜ 현장소장(Site)")
        else:
            step_col2.markdown("✅ 현장소장(Site)")
            
        step_col3.markdown("⬜ 법무팀(Legal)")
        
        if st.button("현장소장 승인 요청"):
            with st.spinner("워크플로우 라우팅 중..."):
                time.sleep(1)
            st.session_state['workflow_step'] = 'Reviewing'
            st.toast("현장소장님에게 승인 요청 메일이 발송되었습니다!", icon="📧")
            st.rerun()