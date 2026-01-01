import os
import re
from typing import List
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()

class LegalGraphRetriever:
    """
    Notebook 3.1.1 기반: Neo4j에서 관련 법령 텍스트 조회
    """
    def __init__(self):
        uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        user = os.getenv("NEO4J_USER", "neo4j")
        pwd = os.getenv("NEO4J_PASSWORD", "password")
        self.driver = None
        try:
            self.driver = GraphDatabase.driver(uri, auth=(user, pwd))
            self.driver.verify_connectivity()
            print("[Graph] Neo4j Connected Successfully.")
        except Exception as e:
            print(f"[Graph] Connection Failed: {e}")

    def close(self):
        if self.driver:
            self.driver.close()

    def expand_query_with_graph(self, query: str, limit: int = 3) -> List[str]:
        if not self.driver:
            return []

        # 정규식으로 '제N조' 패턴 추출
        patterns = re.findall(r"(제\d+조(?:의\d+)?)", query)
        if not patterns:
            return []
        
        target_articles = list(set(patterns))
        expanded_context = []

        # Notebook에 있던 Cypher 쿼리 그대로 활용
        cypher_query = """
        UNWIND $article_keys AS key
        MATCH (c:Clause)
        WHERE c.clause_id CONTAINS key
        
        // 1. 본인 노드
        OPTIONAL MATCH (c)-[:CONTAINS*0..1]-(sub)
        // 2. 참조 (Outbound)
        OPTIONAL MATCH (c)-[:REFERS_TO]->(target:Clause)
        // 3. 역참조 (Inbound)
        OPTIONAL MATCH (source:Clause)-[:REFERS_TO]->(c)
        
        RETURN 
            c.text as SelfText,
            target.text as RefText,
            source.text as CitedText
        LIMIT $limit
        """
        
        try:
            with self.driver.session() as session:
                result = session.run(cypher_query, article_keys=target_articles, limit=limit)
                for record in result:
                    for val in record.values():
                        if val and val.strip():
                            expanded_context.append(val.strip()[:300]) # 길이 제한
        except Exception as e:
            print(f"[Graph] Query Error: {e}")

        return list(set(expanded_context))