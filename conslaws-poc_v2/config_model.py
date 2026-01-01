from pydantic import BaseModel

class RAGConfig(BaseModel):
    # 1. 검색 제어
    top_k: int = 5
    
    # 2. 기능 활성화 (UI Toggle)
    use_glossary: bool = True    # 용어집 확장 사용 여부
    use_graph_db: bool = False   # Neo4j 지식그래프 사용 여부
    use_reranker: bool = True    # CrossEncoder 리랭킹 사용 여부
    
    # 3. 모델 가중치 (RRF) - 고급 설정
    alpha_blend: float = 0.8     # 리랭커 점수 반영 비율