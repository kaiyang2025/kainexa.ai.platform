# app.py
import streamlit as st
import pandas as pd
import mock_data
import time

# --- 1. 페이지 설정 및 CSS ---
st.set_page_config(page_title="Kainexa | Construction Dispute OS", layout="wide")

# 경고창 제거 및 스타일링
st.markdown("""
<style>
    .agent-log { font-family: 'Courier New', monospace; font-size: 0.85rem; color: #d1d1d1; background: #262730; padding: 10px; border-radius: 5px; margin-bottom: 5px; border-left: 3px solid #00ff00; }
    .status-running { color: #ffaa00; font-weight: bold; }
    .status-done { color: #00ff00; font-weight: bold; }
    .stDeployButton {display:none;} /* 데모 시 불필요한 버튼 숨김 */
</style>
""", unsafe_allow_html=True)

# --- 2. 상태 관리 ---
if 'processed' not in st.session_state:
    st.session_state['processed'] = False
if 'agent_status' not in st.session_state:
    st.session_state['agent_status'] = "IDLE"

# --- 3. 사이드바 (전문가용 모니터링창) ---
with st.sidebar:
    # 로고 오류 수정 (텍스트 로고로 대체하여 안정성 확보)
    st.markdown("<h2 style='text-align: center; color: #007bff;'>🏗️ KAINEXA</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 0.8rem;'>Construction Dispute Readiness OS</p>", unsafe_allow_html=True)
    st.title("🤖 Agentic Monitor")
    
    # 에이전트 상태 시각화 (동적 변경) [cite: 74, 124]
    status_color = "#00ff00" if st.session_state['processed'] else "#cccccc"
    st.markdown(f"**Current Status:** <span style='color:{status_color};'>{st.session_state['agent_status']}</span>", unsafe_allow_html=True)
    
    st.write(f"● **Planner Agent**: `{'DONE' if st.session_state['processed'] else 'IDLE'}`")
    st.write(f"● **Clause Agent**: `{'DONE' if st.session_state['processed'] else 'IDLE'}`")
    st.write(f"● **Evidence Agent**: `{'DONE' if st.session_state['processed'] else 'IDLE'}`")
    st.write(f"● **Strategy Agent**: `{'DONE' if st.session_state['processed'] else 'IDLE'}`")
    
    st.markdown("---")
    st.subheader("📁 Data Ingestion")
    uploaded_file = st.file_uploader("현장 데이터 업로드 (PDF, XLSX)", type=['pdf', 'xlsx'], help="공문, 작업일보, 계약서 등")
    
    if uploaded_file and not st.session_state['processed']:
        st.session_state['agent_status'] = "RUNNING..."
        # 메인 화면에서 로그가 먼저 실행되도록 유도

# --- 4. 메인 화면 ---
st.header("Construction Dispute Readiness Dashboard")
st.caption("Project: Gangwon Univ. Library Expansion | Case ID: CLM-2024-007")

# 에이전트 실행 애니메이션 및 로그 [cite: 246, 248]
if uploaded_file and not st.session_state['processed']:
    with st.status("🛠️ **Kainexa Agents 협업 추론 중...**", expanded=True) as status:
        st.write("Planner Agent: 분쟁 유형 식별 및 입증 전략 수립...")
        time.sleep(0.8)
        st.write("Clause Agent: 도급계약서 제25조(불가항력) 추출 및 요건 분석...")
        time.sleep(0.8)
        st.write("Evidence Agent: 누락된 7월 16, 18일 작업일보 데이터 팩트 매핑...")
        time.sleep(0.8)
        st.write("Strategy Agent: 지체상금 면책 논리 완결성 검증 완료.")
        st.session_state['processed'] = True
        st.session_state['agent_status'] = "COMPLETED"
        status.update(label="✅ 분석 완료: 모든 증거가 타임라인에 연결되었습니다.", state="complete", expanded=False)
    st.rerun()

st.markdown("---")

col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("📅 Timeline & Gap Analysis")
    # [P0 핵심] 타임라인 시각화 [cite: 165, 176]
    timeline = mock_data.get_filled_timeline() if st.session_state['processed'] else mock_data.get_initial_timeline()
    
    if not st.session_state['processed']:
        st.error("🚨 **Critical Gap Detected**: 7월 15일~20일 사이의 증거가 누락되었습니다.")
    else:
        st.success("✅ **Gap Resolved**: 누락된 6일간의 증거가 추가되어 인과관계가 소명되었습니다.") [cite: 133, 141]
    
    st.table(pd.DataFrame(timeline))

    st.subheader("⚖️ Element-Evidence Matrix") # [cite: 166, 178]
    st.table(pd.DataFrame(mock_data.get_element_matrix()))

with col_right:
    st.subheader("📄 Evidence Pack Preview")
    tabs = st.tabs(["Draft Statement", "Exhibit Index", "Approval"])
    
    with tabs[0]:
        if st.session_state['processed']:
            st.info("💡 모든 문장에 근거(Citation)가 자동 태깅되었습니다.") [cite: 117, 189]
            st.markdown(mock_data.get_advanced_draft())
            st.download_button("📥 최종 Evidence Pack 다운로드", data="pdf_content", file_name="Kainexa_Claim_Package.pdf")
        else:
            st.warning("데이터 분석이 완료되면 법리 서면 초안이 생성됩니다.")

    with tabs[2]:
        st.write(f"상태: **{'Ready to Submit' if st.session_state['processed'] else 'Drafting'}**")
        st.button("현장소장 승인 요청", disabled=not st.session_state['processed'])
        st.button("법무팀 검토 요청", disabled=not st.session_state['processed'])