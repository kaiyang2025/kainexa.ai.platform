# app.py
import streamlit as st
import pandas as pd
import mock_data
import time

st.set_page_config(page_title="Kainexa | Dispute OS", layout="wide")

# --- 전문적인 UI/UX 스타일링 ---
st.markdown("""
<style>
    /* 리스크 대시보드 카드 스타일 */
    .metric-card { background-color: #f8f9fa; border-radius: 10px; padding: 20px; border-top: 5px solid #007bff; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    /* 타임라인 스타일 */
    .timeline-item { border-left: 3px solid #ddd; padding-left: 20px; margin-bottom: 20px; position: relative; }
    .timeline-dot { position: absolute; left: -9px; top: 5px; width: 15px; height: 15px; border-radius: 50%; background: #007bff; }
</style>
""", unsafe_allow_html=True)

# --- 상태 관리 ---
if "messages" not in st.session_state:
    st.session_state.messages = mock_data.get_chat_history()

# --- 사이드바: 3. Risk Dashboard (미니 버전) ---
with st.sidebar:
    st.image("https://via.placeholder.com/200x60?text=KAINEXA", use_container_width=True)
    st.title("🚩 Real-time Risk")
    metrics = mock_data.get_risk_metrics()
    
    st.metric("Win Probability", f"{metrics['win_probability']}%", "+5%")
    st.metric("Total Claim", metrics['total_claim_amount'])
    
    if metrics['overall_risk'] == "High":
        st.error("⚠️ Overall Risk: HIGH (Evidence Missing)")
    
    st.markdown("---")
    uploaded = st.file_uploader("Upload Evidence", type=['pdf'])

# --- 메인 화면: 3개 탭 구성 ---
st.title("Dispute Readiness Workspace")
tab1, tab2, tab3 = st.tabs(["📊 Risk Dashboard & Timeline", "💬 Interactive Copilot", "📄 Evidence Pack"])

# --- Tab 1: Risk Dashboard & Visual Timeline ---
with tab1:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("🚩 Risk Analysis")
        st.info("💡 **AI Insight**: 7월 15일 구간의 '운반일지'가 보완되면 승소 확률이 92%로 상승합니다.")
        
        # 상세 리스크 모니터링
        chart_data = pd.DataFrame({"Category": ["Contract", "Evidence", "Timeline", "Precedent"], "Score": [90, 40, 70, 85]})
        st.bar_chart(chart_data.set_index("Category"))

    with col2:
        st.subheader("📍 Visual Evidence Timeline")
        # 시각적 타임라인 구현
        for item in mock_data.get_visual_timeline():
            st.markdown(f"""
            <div class="timeline-item">
                <div class="timeline-dot" style="background: {item['color']};"></div>
                <b style="color: {item['color']};">{item['date']}</b> - <b>{item['title']}</b><br>
                <small>{item['desc']}</small>
            </div>
            """, unsafe_allow_html=True)

# --- Tab 2: 1. Interactive Copilot ---
with tab2:
    st.subheader("🤖 Interactive Legal Copilot")
    st.caption("사건 맥락을 이해하는 AI와 대화하며 서면을 완성하세요.")
    
    # 채팅 인터페이스
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    if prompt := st.chat_input("이 사건에 대해 궁금한 점을 물어보세요..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("법률 온톨로지 탐색 중..."):
                time.sleep(1)
                response = "검토하신 '집중호우' 사유는 도급계약서 제25조에 따른 면책 요건을 충족합니다. 관련 증거 3건을 포함하여 서면 초안에 반영해두었습니다."
                st.write(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

# --- Tab 3: Evidence Pack (기존 산출물) ---
with tab3:
    st.subheader("📄 Evidence Pack & Draft")
    st.markdown(mock_data.get_advanced_draft())
    st.button("📥 최종 Evidence Pack 다운로드 (PDF)")