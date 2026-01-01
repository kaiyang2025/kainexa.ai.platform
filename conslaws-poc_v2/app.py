import streamlit as st
import os
from config_model import RAGConfig
from rag_engine import ConstructionRAG

# 페이지 설정
st.set_page_config(page_title="Kainexa 건설법령 RAG", layout="wide")

# 데이터 경로 (서버 환경에 맞게 수정 필요)
BASE_DIR = os.getcwd() # 현재 폴더 기준
DATA_FILE = os.path.join(BASE_DIR, "data", "chunks_tonghap.jsonl") # 통합 청킹 파일
GLOSSARY_FILE = os.path.join(BASE_DIR, "data", "construction_law_glossary.csv")

# ---------------------------------------------------------
# 시스템 초기화 (캐싱하여 리소스 재사용)
# ---------------------------------------------------------
@st.cache_resource
def get_system():
    # 데이터 파일이 존재하는지 확인용 가드
    if not os.path.exists(DATA_FILE):
        st.error(f"데이터 파일이 없습니다: {DATA_FILE}")
        return None
    
    st.info("RAG 엔진을 초기화 중입니다... (모델 로딩)")
    rag = ConstructionRAG(DATA_FILE, GLOSSARY_FILE)
    return rag

system = get_system()

# ---------------------------------------------------------
# UI Layout
# ---------------------------------------------------------
st.title("🏗️ Kainexa Construction Law RAG")
st.caption("건설법령/판례/의결서 하이브리드 검색 시스템")

# [사이드바] 설정 컨트롤
with st.sidebar:
    st.header("⚙️ 검색 파라미터 설정")
    
    st.subheader("1. 검색 범위")
    top_k = st.slider("검색 문서 수 (Top-k)", 1, 20, 5)
    
    st.subheader("2. 고급 기능 On/Off")
    use_glossary = st.toggle("용어집(Glossary) 확장", value=True, help="전문 용어 동의어 확장")
    use_graph = st.toggle("Graph DB(Neo4j) 연결", value=False, help="관련 조항/참조 조항 탐색")
    use_rerank = st.toggle("Re-ranker 적용", value=True, help="정밀도 향상을 위한 재정렬")
    
    st.subheader("3. 결과 필터링")
    blend = st.slider("Reranker 반영 비율", 0.0, 1.0, 0.8)

    # Config 객체 생성
    config = RAGConfig(
        top_k=top_k,
        use_glossary=use_glossary,
        use_graph_db=use_graph,
        use_reranker=use_rerank,
        alpha_blend=blend
    )
    
    if st.button("설정 적용 및 초기화"):
        st.cache_data.clear()
        st.success("설정이 적용되었습니다.")

# [메인] 채팅 인터페이스
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if "context" in msg:
            with st.expander("참고 문서 확인"):
                for idx, doc in enumerate(msg["context"], 1):
                    score_display = f"(Re-rank: {doc.get('rerank_score',0):.2f})" if use_rerank else f"(Fused: {doc.get('fused_score',0):.2f})"
                    st.markdown(f"**{idx}. [{doc.get('law_name')}] {doc.get('clause_id')}** {score_display}")
                    st.text(doc.get('text')[:200] + "...")

if prompt := st.chat_input("질문을 입력하세요 (예: 하도급 대금 지급 보증 예외 사유는?)"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    if system:
        with st.chat_message("assistant"):
            with st.spinner("법령 분석 및 검색 중..."):
                # 엔진 실행
                results, graph_info = system.run_pipeline(prompt, config)
                
                # 답변 생성 (현재는 검색 결과 요약 형태로 표시, LLM 연결 가능)
                if not results:
                    response_text = "관련된 법령 정보를 찾을 수 없습니다."
                else:
                    top_doc = results[0]
                    response_text = f"**[{top_doc.get('law_name')}] {top_doc.get('clause_id')}** 내용을 기반으로 답변합니다.\n\n"
                    response_text += top_doc.get('text')
                    
                    if graph_info:
                        response_text += f"\n\n💡 **지식그래프 참고:** {graph_info[:150]}..."

                st.markdown(response_text)
                
                # 근거 문서 표시
                with st.expander("🔍 검색된 근거 문서 (Evidence)"):
                    for i, doc in enumerate(results, 1):
                        st.markdown(f"--- \n**{i}. {doc.get('title', '문서')}**")
                        st.caption(f"Score: {doc.get('rerank_score', 0):.4f}")
                        st.write(doc.get('text'))

            # 대화 기록 저장
            st.session_state.messages.append({
                "role": "assistant", 
                "content": response_text,
                "context": results
            })
    else:
        st.error("시스템이 로드되지 않았습니다. 데이터 파일 경로를 확인해주세요.")