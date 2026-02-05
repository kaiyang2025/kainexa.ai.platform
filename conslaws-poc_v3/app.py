# app.py
import streamlit as st
import pandas as pd
import mock_data
import time

# --- 1. 페이지 설정 ---
st.set_page_config(page_title="Kainexa | Construction Dispute OS", layout="wide")

# 스타일링
st.markdown("""
<style>
    .agent-log { font-family: 'Courier New', monospace; font-size: 0.85rem; color: #d1d1d1; background: #262730; padding: 10px; border-radius: 5px; margin-bottom: 5px; border-left: 3px solid #00ff00; }
    .status-badge { padding: 4px 8px; border-radius: 4px; font-weight: bold; font-size: 0.8rem; }
    .stTable { font-size: 0.9rem; }
</style>
""", unsafe_allow_html=True)

# --- 2. 상태 관리 ---
if 'processed' not in st.session_state:
    st.session_state['processed'] = False

# --- 3. 사이드바 (에이전트 상태창) ---
with st.sidebar:
    st.image("https://via.placeholder.com/150x50?text=Kainexa+Logo", use_column_width=True) # 로고 자리
    st.title("🤖 Agentic Monitor")
    
    # 에이전트 실시간 상태
    st.write("● **Planner Agent**: `IDLE`" if not st.session_state['processed'] else "● **Planner Agent**: `SLEEP`")
    st.write("● **Clause Agent**: `IDLE`" if not st.session_state['processed'] else "● **Clause Agent**: `SLEEP`")
    st.write("● **Evidence Agent**: `IDLE`" if not st.session_state['processed'] else "● **Evidence Agent**: `SLEEP`")
    st.markdown("---")
    
    st.subheader("Data Ingestion")
    uploaded_file = st.file_uploader("Upload Case Documents (PDF, EML, XLSX)", type=['pdf'])
    
    if uploaded_file:
        st.session_state['processed'] = True

# --- 4. 메인 화면 ---
st.title("Construction Dispute Readiness Dashboard")
st.caption("Project: Gangwon Univ. Library Expansion | Claim ID: CLM-2024-007")

# 에이전트 실행 애니메이션 (업로드 시)
if uploaded_file:
    with st.expander("🛠️ **Agentic Workflow Reasoning (실시간 추론 로그)**", expanded=True):
        log_placeholder = st.empty()
        logs = [
            " [Planner] 분쟁 유형 식별: '공기 지연(Delay)' 및 '지체상금 면책' 전략 수립 중...",
            " [Clause] 도급계약서 제25조 '불가항력' 조항 로드 완료.",
            " [Evidence] 업로드된 파일에서 '운반일지' 및 '작업일보' OCR 텍스트 추출 중...",
            " [Evidence] 추출된 팩트(도로침수)와 기상청 데이터 인과관계 매핑 성공.",
            " [Strategy] 지연 사유 분석 완료: 불가항력(60%) + 발주처 귀책(40%)",
            " [Legal] Evidence Pack v1.0 구성 완료. 최종 검토 대기 중."
        ]
        current_logs = ""
        for log in logs:
            current_logs += f'<div class="agent-log">{log}</div>'
            log_placeholder.markdown(current_logs, unsafe_allow_html=True)
            time.sleep(0.6)

st.markdown("---")

# 레이아웃 구성
col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("📅 Timeline & Gap Analysis")
    # 타임라인 데이터
    timeline = mock_data.get_filled_timeline() if st.session_state['processed'] else mock_data.get_initial_timeline()
    
    if not st.session_state['processed']:
        st.error("🚨 **Critical Gap Detected**: 7월 15일~20일 사이의 증거가 누락되어 청구 요건을 충족하지 못합니다.")
    else:
        st.success("✅ **Gap Resolved**: 추가 문서에서 7월 16, 18일 증거가 확보되어 인과관계 입증이 완료되었습니다.")
    
    st.table(pd.DataFrame(timeline))

    st.subheader("⚖️ Element-Evidence Matrix")
    st.markdown("AI가 법리 요건별로 증거를 매핑한 결과입니다.")
    st.table(pd.DataFrame(mock_data.get_element_matrix()))

with col_right:
    st.subheader("📄 Evidence Pack Preview")
    tabs = st.tabs(["Draft Statement", "Exhibit Index", "Approval"])
    
    with tabs[0]:
        if st.session_state['processed']:
            st.info("💡 모든 문장에 근거(Citation)가 자동 태그되었습니다. 파란색 링크를 클릭하면 원문을 확인할 수 있습니다.")
            st.markdown(mock_data.get_advanced_draft())
            st.download_button("📥 Download Final Evidence Pack", data="pdf_content", file_name="Kainexa_Claim_Package.pdf")
        else:
            st.warning("데이터 분석이 완료되면 법리 서면 초안이 생성됩니다.")
            
    with tabs[1]:
        st.write("자동 생성된 증거 목록(Exhibit Index)입니다.")
        st.write("1. [Ex-01] 기상청 강수량 데이터 (2024.07)")
        st.write("2. [Ex-02] 공문 No.IS-24-055")
        if st.session_state['processed']:
            st.write("3. [Ex-03] 운반일지 (2024.07.16) - **NEW**")
            st.write("4. [Ex-04] 작업일보 (2024.07.18) - **NEW**")

    with tabs[2]:
        st.write("Workflow Status: **Ready for Review**")
        st.button("Request Approval (Site Manager)")
        st.button("Request Approval (Legal Team)")