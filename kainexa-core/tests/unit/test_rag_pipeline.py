"""
tests/unit/test_rag_pipeline.py
RAG 파이프라인 단위 테스트
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime
import numpy as np

from src.core.governance.rag_pipeline import (
    RAGPipeline,
    DocumentProcessor,
    VectorStore,
    Retriever,
    Reranker,
    DocumentMetadata,
    AccessLevel,
    ChunkingStrategy
)


class TestRAGPipeline:
    """RAG 파이프라인 통합 테스트"""
    
    @pytest.fixture
    def rag_pipeline(self):
        """RAG 파이프라인 인스턴스"""
        config = {
            "vector_store": {
                "type": "qdrant",
                "host": "localhost",
                "port": 6333,
                "collection": "test_docs"
            },
            "embedding_model": {
                "name": "text-embedding-ada-002",
                "dimension": 1536
            },
            "chunking": {
                "strategy": "semantic",
                "max_chunk_size": 500,
                "overlap": 50
            },
            "retrieval": {
                "top_k": 10,
                "similarity_threshold": 0.7
            }
        }
        return RAGPipeline(config)
    
    @pytest.fixture
    def sample_documents(self):
        """샘플 문서"""
        return [
            {
                "content": "인공지능은 컴퓨터 과학의 한 분야로, 기계가 인간의 학습능력과 같은 지능적인 행동을 할 수 있도록 하는 기술입니다.",
                "metadata": {
                    "doc_id": "doc_001",
                    "title": "AI 개요",
                    "source": "wikipedia",
                    "language": "ko",
                    "access_level": AccessLevel.PUBLIC
                }
            },
            {
                "content": "머신러닝은 명시적으로 프로그래밍하지 않고도 컴퓨터가 학습할 수 있도록 하는 알고리즘과 기술을 연구하는 분야입니다.",
                "metadata": {
                    "doc_id": "doc_002",
                    "title": "머신러닝 기초",
                    "source": "textbook",
                    "language": "ko",
                    "access_level": AccessLevel.PUBLIC
                }
            },
            {
                "content": "This is a confidential document about company strategy.",
                "metadata": {
                    "doc_id": "doc_003",
                    "title": "Company Strategy",
                    "source": "internal",
                    "language": "en",
                    "access_level": AccessLevel.CONFIDENTIAL
                }
            }
        ]
    
    @pytest.mark.asyncio
    async def test_document_ingestion(self, rag_pipeline, sample_documents):
        """문서 수집 테스트"""
        # 벡터 스토어 모킹
        with patch.object(rag_pipeline.vector_store, 'add_documents') as mock_add:
            mock_add.return_value = True
            
            # 문서 추가
            for doc in sample_documents:
                result = await rag_pipeline.add_document(
                    doc["content"],
                    DocumentMetadata(**doc["metadata"])
                )
                assert result == True
            
            # 호출 확인
            assert mock_add.call_count == len(sample_documents)
    
    @pytest.mark.asyncio
    async def test_document_chunking(self, rag_pipeline):
        """문서 청킹 테스트"""
        processor = DocumentProcessor(
            strategy=ChunkingStrategy.SEMANTIC,
            max_chunk_size=100,
            overlap=20
        )
        
        long_text = "이것은 매우 긴 문서입니다. " * 50  # 긴 텍스트 생성
        
        chunks = await processor.chunk_text(long_text)
        
        assert len(chunks) > 1
        assert all(len(chunk) <= 100 for chunk in chunks)
        
        # 오버랩 확인
        for i in range(len(chunks) - 1):
            # 인접한 청크 간 일부 내용 중복
            assert any(word in chunks[i+1] for word in chunks[i].split()[-5:])
    
    @pytest.mark.asyncio
    async def test_semantic_search(self, rag_pipeline):
        """의미 검색 테스트"""
        query = "인공지능이란 무엇인가?"
        
        # 임베딩 모킹
        with patch.object(rag_pipeline, 'get_embedding') as mock_embed:
            mock_embed.return_value = np.random.randn(1536).tolist()
            
            # 벡터 검색 모킹
            with patch.object(rag_pipeline.vector_store, 'search') as mock_search:
                mock_search.return_value = [
                    {
                        "id": "doc_001",
                        "score": 0.95,
                        "content": "인공지능은 컴퓨터 과학의 한 분야...",
                        "metadata": {"title": "AI 개요"}
                    },
                    {
                        "id": "doc_002",
                        "score": 0.85,
                        "content": "머신러닝은 명시적으로...",
                        "metadata": {"title": "머신러닝 기초"}
                    }
                ]
                
                results = await rag_pipeline.search(query, k=5)
                
                assert len(results) == 2
                assert results[0]["score"] > results[1]["score"]
                assert "인공지능" in results[0]["content"]
    
    @pytest.mark.asyncio
    async def test_hybrid_search(self, rag_pipeline):
        """하이브리드 검색 테스트 (의미 + 키워드)"""
        query = "machine learning algorithms"
        
        with patch.object(rag_pipeline, 'semantic_search') as mock_semantic:
            mock_semantic.return_value = [
                {"id": "1", "score": 0.9, "content": "ML content"}
            ]
            
            with patch.object(rag_pipeline, 'keyword_search') as mock_keyword:
                mock_keyword.return_value = [
                    {"id": "2", "score": 0.85, "content": "algorithm content"}
                ]
                
                results = await rag_pipeline.hybrid_search(
                    query,
                    semantic_weight=0.7,
                    keyword_weight=0.3
                )
                
                # 두 검색 결과가 결합되어야 함
                assert len(results) >= 2
                mock_semantic.assert_called_once()
                mock_keyword.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_reranking(self, rag_pipeline):
        """재순위화 테스트"""
        query = "인공지능 응용"
        
        initial_results = [
            {"id": "1", "score": 0.8, "content": "AI 게임 응용"},
            {"id": "2", "score": 0.85, "content": "통계학 기초"},
            {"id": "3", "score": 0.75, "content": "AI 의료 응용"}
        ]
        
        reranker = Reranker(model="cross-encoder/ms-marco")
        
        with patch.object(reranker, 'score') as mock_score:
            # 재순위 점수 (쿼리와의 실제 관련성)
            mock_score.side_effect = [0.95, 0.3, 0.9]
            
            reranked = await reranker.rerank(query, initial_results)
            
            # 재순위 후 순서 확인
            assert reranked[0]["id"] == "1"  # 가장 높은 재순위 점수
            assert reranked[1]["id"] == "3"  # 두 번째
            assert reranked[2]["id"] == "2"  # 가장 낮은 점수
    
    @pytest.mark.asyncio
    async def test_access_control(self, rag_pipeline, sample_documents):
        """접근 제어 테스트"""
        # PUBLIC 권한 사용자
        public_user_context = {"access_level": AccessLevel.PUBLIC}
        
        with patch.object(rag_pipeline.vector_store, 'search') as mock_search:
            mock_search.return_value = sample_documents
            
            results = await rag_pipeline.search_with_access_control(
                "company strategy",
                user_context=public_user_context
            )
            
            # CONFIDENTIAL 문서는 필터링되어야 함
            assert all(
                doc.get("metadata", {}).get("access_level") != AccessLevel.CONFIDENTIAL
                for doc in results
            )
    
    @pytest.mark.asyncio
    async def test_multilingual_search(self, rag_pipeline):
        """다국어 검색 테스트"""
        # 한국어 쿼리로 영어 문서 검색
        query_ko = "회사 전략"
        
        with patch.object(rag_pipeline, 'detect_language') as mock_detect:
            mock_detect.return_value = "ko"
            
            with patch.object(rag_pipeline, 'translate') as mock_translate:
                mock_translate.return_value = "company strategy"
                
                with patch.object(rag_pipeline.vector_store, 'search') as mock_search:
                    mock_search.return_value = [
                        {
                            "content": "Company strategic planning...",
                            "metadata": {"language": "en"}
                        }
                    ]
                    
                    results = await rag_pipeline.multilingual_search(query_ko)
                    
                    mock_translate.assert_called_once_with(query_ko, target="en")
                    assert len(results) > 0
    
    @pytest.mark.asyncio
    async def test_document_update(self, rag_pipeline):
        """문서 업데이트 테스트"""
        doc_id = "doc_001"
        new_content = "업데이트된 내용입니다."
        
        with patch.object(rag_pipeline.vector_store, 'update_document') as mock_update:
            mock_update.return_value = True
            
            result = await rag_pipeline.update_document(doc_id, new_content)
            
            assert result == True
            mock_update.assert_called_once_with(doc_id, new_content)
    
    @pytest.mark.asyncio
    async def test_document_deletion(self, rag_pipeline):
        """문서 삭제 테스트"""
        doc_id = "doc_001"
        
        with patch.object(rag_pipeline.vector_store, 'delete_document') as mock_delete:
            mock_delete.return_value = True
            
            result = await rag_pipeline.delete_document(doc_id)
            
            assert result == True
            mock_delete.assert_called_once_with(doc_id)
    
    @pytest.mark.asyncio
    async def test_context_window_management(self, rag_pipeline):
        """컨텍스트 윈도우 관리 테스트"""
        retrieved_docs = [
            {"content": "문서 1" * 100, "score": 0.95},
            {"content": "문서 2" * 100, "score": 0.90},
            {"content": "문서 3" * 100, "score": 0.85},
            {"content": "문서 4" * 100, "score": 0.80},
        ]
        
        max_tokens = 500
        
        # 토큰 제한에 맞게 문서 선택
        selected_docs = await rag_pipeline.fit_to_context_window(
            retrieved_docs,
            max_tokens=max_tokens
        )
        
        # 선택된 문서의 총 토큰이 제한 이하
        total_tokens = sum(len(doc["content"].split()) for doc in selected_docs)
        assert total_tokens <= max_tokens
        
        # 높은 점수 문서가 우선 선택
        assert selected_docs[0]["score"] >= selected_docs[-1]["score"]


class TestDocumentProcessor:
    """문서 처리기 테스트"""
    
    @pytest.mark.asyncio
    async def test_text_cleaning(self):
        """텍스트 정제 테스트"""
        processor = DocumentProcessor()
        
        dirty_text = """
        <html>
        <body>
            This is a    test   document.
            
            With HTML tags and    extra    spaces.
        </body>
        </html>
        """
        
        cleaned = await processor.clean_text(dirty_text)
        
        assert "<html>" not in cleaned
        assert "<body>" not in cleaned
        assert "  " not in cleaned  # 중복 공백 제거
        assert "test document" in cleaned
    
    @pytest.mark.asyncio
    async def test_metadata_extraction(self):
        """메타데이터 추출 테스트"""
        processor = DocumentProcessor()
        
        document = {
            "content": "제목: AI 연구\n저자: 김철수\n날짜: 2024-01-15\n\n본문 내용...",
            "source": "pdf"
        }
        
        metadata = await processor.extract_metadata(document)
        
        assert metadata["title"] == "AI 연구"
        assert metadata["author"] == "김철수"
        assert metadata["date"] == "2024-01-15"
    
    @pytest.mark.asyncio
    async def test_sentence_splitting(self):
        """문장 분할 테스트"""
        processor = DocumentProcessor()
        
        text = "첫 번째 문장입니다. 두 번째 문장입니다! 세 번째 문장인가요? 네 번째 문장..."
        
        sentences = await processor.split_sentences(text)
        
        assert len(sentences) == 4
        assert sentences[0] == "첫 번째 문장입니다."
        assert sentences[2] == "세 번째 문장인가요?"


class TestVectorStore:
    """벡터 스토어 테스트"""
    
    @pytest.mark.asyncio
    async def test_qdrant_connection(self):
        """Qdrant 연결 테스트"""
        from src.core.governance.vector_stores import QdrantStore
        
        store = QdrantStore(
            host="localhost",
            port=6333,
            collection="test"
        )
        
        with patch('qdrant_client.QdrantClient') as MockClient:
            mock_client = MockClient.return_value
            mock_client.get_collection.return_value = {"status": "ok"}
            
            is_connected = await store.test_connection()
            assert is_connected == True
    
    @pytest.mark.asyncio
    async def test_batch_insertion(self):
        """배치 삽입 테스트"""
        from src.core.governance.vector_stores import VectorStore
        
        store = VectorStore()
        
        documents = [
            {"id": f"doc_{i}", "embedding": [0.1] * 1536, "content": f"Document {i}"}
            for i in range(100)
        ]
        
        with patch.object(store, 'batch_upsert') as mock_upsert:
            mock_upsert.return_value = True
            
            result = await store.add_batch(documents, batch_size=20)
            
            # 5번의 배치 호출 (100/20)
            assert mock_upsert.call_count == 5
            assert result == True
    
    @pytest.mark.asyncio
    async def test_similarity_metrics(self):
        """유사도 메트릭 테스트"""
        from src.core.governance.vector_stores import compute_similarity
        
        # 코사인 유사도
        vec1 = [1, 0, 0]
        vec2 = [0, 1, 0]
        vec3 = [1, 0, 0]
        
        similarity_12 = compute_similarity(vec1, vec2, metric="cosine")
        similarity_13 = compute_similarity(vec1, vec3, metric="cosine")
        
        assert similarity_12 == 0  # 직교 벡터
        assert similarity_13 == 1  # 동일 벡터


if __name__ == "__main__":
    pytest.main([__file__, "-v"])