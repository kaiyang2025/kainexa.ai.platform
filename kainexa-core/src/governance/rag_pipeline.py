# src/governance/rag_pipeline.py
"""
RAG 거버넌스 - 문서 파이프라인 및 품질 관리
"""
import hashlib
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer, CrossEncoder
import os
import tiktoken
import structlog
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = structlog.get_logger()

class DocumentStatus(Enum):
    """문서 상태"""
    PENDING = "pending"
    PROCESSING = "processing"
    INDEXED = "indexed"
    FAILED = "failed"
    EXPIRED = "expired"
    RESTRICTED = "restricted"

class AccessLevel(Enum):
    """접근 권한 레벨"""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    SECRET = "secret"

@dataclass
class DocumentMetadata:
    """문서 메타데이터"""
    doc_id: str
    title: str
    source: str
    author: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    access_level: AccessLevel = AccessLevel.PUBLIC
    tags: List[str] = field(default_factory=list)
    language: str = "ko"
    version: str = "1.0"
    checksum: Optional[str] = None
    quality_score: float = 0.0
    usage_count: int = 0
    
    def is_expired(self) -> bool:
        """만료 여부 확인"""
        if self.expires_at:
            return datetime.now() > self.expires_at
        return False
    
    def can_access(self, user_level: AccessLevel) -> bool:
        """접근 가능 여부"""
        level_hierarchy = {
            AccessLevel.PUBLIC: 0,
            AccessLevel.INTERNAL: 1,
            AccessLevel.CONFIDENTIAL: 2,
            AccessLevel.SECRET: 3
        }
        return level_hierarchy.get(user_level, 0) >= level_hierarchy.get(self.access_level, 0)

@dataclass
class ChunkMetadata:
    """청크 메타데이터"""
    chunk_id: str
    doc_id: str
    chunk_index: int
    start_char: int
    end_char: int
    tokens: int
    embedding_model: str
    
class DocumentProcessor:
    """문서 처리 파이프라인"""
    
    def __init__(self, embedding_model_name: str = "sentence-transformers/xlm-r-bert-base-nli-stsb-mean-tokens"):
        
         # ✅ 임베딩 모델은 기본 CPU로 구동 (env로 변경 가능)
        emb_device = os.getenv("KXN_EMB_DEVICE", "cpu")  # ex) "cpu" | "cuda:0"
        self.embedding_model = SentenceTransformer(embedding_model_name, device=emb_device)
        # 배치 사이즈도 환경변수로 제어
        self._emb_batch = int(os.getenv("KXN_EMB_BATCH", "16"))        
        self.tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
        
        # 청킹 전략
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
            length_function=self._token_length
        )
        
    def _token_length(self, text: str) -> int:
        """토큰 길이 계산"""
        return len(self.tokenizer.encode(text))
    
    async def process_document(self, 
                              content: str, 
                              metadata: DocumentMetadata) -> List[Dict[str, Any]]:
        """문서 처리"""
        logger.info(f"Processing document: {metadata.doc_id}")
        
        # 1. 유효성 검증
        if not await self._validate_document(content, metadata):
            raise ValueError("Document validation failed")
        
        # 2. 체크섬 계산
        metadata.checksum = self._calculate_checksum(content)
        
        # 3. 청킹
        chunks = self._chunk_document(content, metadata)
        
        # 4. 임베딩 생성
        embeddings = await self._create_embeddings(chunks)
        
        # 5. 품질 평가
        metadata.quality_score = await self._assess_quality(content, chunks)
        
        # 6. 결과 조합
        processed_chunks = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            chunk_metadata = ChunkMetadata(
                chunk_id=f"{metadata.doc_id}_chunk_{i}",
                doc_id=metadata.doc_id,
                chunk_index=i,
                start_char=chunk.get('start', 0),
                end_char=chunk.get('end', 0),
                tokens=self._token_length(chunk['text']),
                embedding_model=str(self.embedding_model.get_sentence_embedding_dimension())
            )
            
            processed_chunks.append({
                'chunk_id': chunk_metadata.chunk_id,
                'text': chunk['text'],
                'embedding': embedding,
                'metadata': chunk_metadata,
                'doc_metadata': metadata
            })
        
        logger.info(f"Document processed: {len(processed_chunks)} chunks created")
        return processed_chunks
    
    async def _validate_document(self, content: str, metadata: DocumentMetadata) -> bool:
        """문서 유효성 검증"""
        # 크기 제한
        if len(content) > 1_000_000:  # 1MB
            logger.warning("Document too large")
            return False
        
        # 만료 확인
        if metadata.is_expired():
            logger.warning("Document expired")
            return False
        
        # 언어 감지 (간단한 한글 체크)
        if metadata.language == "ko":
            korean_chars = sum(1 for c in content if '가' <= c <= '힣')
            if korean_chars / len(content) < 0.3:
                logger.warning("Document language mismatch")
                return False
        
        return True
    
    def _calculate_checksum(self, content: str) -> str:
        """체크섬 계산"""
        return hashlib.sha256(content.encode()).hexdigest()
    
    def _chunk_document(self, content: str, metadata: DocumentMetadata) -> List[Dict[str, Any]]:
        """문서 청킹"""
        chunks = self.text_splitter.split_text(content)
        
        chunk_dicts = []
        current_pos = 0
        
        for i, chunk_text in enumerate(chunks):
            start_pos = content.find(chunk_text, current_pos)
            end_pos = start_pos + len(chunk_text)
            
            chunk_dicts.append({
                'text': chunk_text,
                'start': start_pos,
                'end': end_pos,
                'index': i
            })
            
            current_pos = end_pos
        
        return chunk_dicts
    
    async def _create_embeddings(self, chunks: List[Dict[str, Any]]) -> List[np.ndarray]:
        """임베딩 생성"""
        texts = [chunk['text'] for chunk in chunks]
        embeddings = self.embedding_model.encode(
            texts,
            show_progress_bar=False,
            batch_size=self._emb_batch
        )        
        return embeddings
    
    async def _assess_quality(self, content: str, chunks: List[Dict[str, Any]]) -> float:
        """문서 품질 평가"""
        score = 1.0
        
        # 길이 체크
        if len(content) < 100:
            score *= 0.5
        
        # 청크 수 체크
        if len(chunks) < 2:
            score *= 0.7
        elif len(chunks) > 100:
            score *= 0.8
        
        # 중복 체크
        unique_chunks = set(chunk['text'] for chunk in chunks)
        duplication_ratio = 1 - (len(unique_chunks) / len(chunks))
        score *= (1 - duplication_ratio * 0.5)
        
        return min(max(score, 0.0), 1.0)

class RAGGovernance:
    """RAG 거버넌스 시스템"""
    
    def __init__(self, 
                 qdrant_host: str = "localhost",
                 qdrant_port: int = 6333,
                 collection_name: str = "kainexa_knowledge"):
        
        # Qdrant 클라이언트
        self.qdrant = QdrantClient(host=qdrant_host, port=qdrant_port)
        self.collection_name = collection_name
        
        # 문서 처리기
        self.processor = DocumentProcessor()
        
        # Cross-Encoder for reranking
        # ✅ 리랭커는 기본 CPU로 구동 (env로 변경 가능)
        rerank_device = os.getenv("KXN_RERANK_DEVICE", "cpu")  # ex) "cpu" | "cuda:0"
        self.reranker = CrossEncoder(
            'cross-encoder/ms-marco-MiniLM-L-6-v2',
            max_length=512,
            device=rerank_device
        )
        
        # 거버넌스 정책
        self.policies = {
            'max_age_days': 365,
            'min_quality_score': 0.3,
            'max_results': 10,
            'diversity_weight': 0.3
        }
        
        # 초기화
        self._init_collection()
    
    def _init_collection(self):
        """컬렉션 초기화"""
        try:
            self.qdrant.get_collection(self.collection_name)
        except:
            # 컬렉션 생성
            self.qdrant.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=768,  # XLM-RoBERTa dimension
                    distance=Distance.COSINE
                )
            )
            logger.info(f"Created collection: {self.collection_name}")
    
    async def add_document(self, 
                          content: str,
                          metadata: DocumentMetadata) -> bool:
        """문서 추가"""
        try:
            # 문서 처리
            processed_chunks = await self.processor.process_document(content, metadata)
            
            # Qdrant에 저장
            points = []
            for chunk in processed_chunks:
                point = PointStruct(
                    id=hash(chunk['chunk_id']) & 0x7FFFFFFF,  # 32-bit positive int
                    vector=chunk['embedding'].tolist(),
                    payload={
                        'chunk_id': chunk['chunk_id'],
                        'doc_id': chunk['metadata'].doc_id,
                        'text': chunk['text'],
                        'chunk_index': chunk['metadata'].chunk_index,
                        'tokens': chunk['metadata'].tokens,
                        'doc_title': metadata.title,
                        'doc_source': metadata.source,
                        'access_level': metadata.access_level.value,
                        'created_at': metadata.created_at.isoformat(),
                        'quality_score': metadata.quality_score,
                        'tags': metadata.tags
                    }
                )
                points.append(point)
            
            # 배치 업로드
            self.qdrant.upsert(
                collection_name=self.collection_name,
                points=points
            )
            
            logger.info(f"Added document {metadata.doc_id} with {len(points)} chunks")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add document: {e}")
            return False
    
    async def retrieve(self,
                      query: str,
                      k: int = 5,
                      user_access_level: AccessLevel = AccessLevel.PUBLIC,
                      filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """문서 검색 및 거버넌스 적용"""
        
        # 1. 쿼리 임베딩
        query_embedding = self.processor.embedding_model.encode(query)
        
        # 2. 필터 구성
        search_filters = self._build_filters(user_access_level, filters)
        
        # 3. 벡터 검색
        search_results = self.qdrant.search(
            collection_name=self.collection_name,
            query_vector=query_embedding.tolist(),
            limit=k * 3,  # Reranking을 위해 더 많이 검색
            query_filter=search_filters
        )
        
        # 4. 거버넌스 필터링
        filtered_results = await self._apply_governance(search_results)
        
        # 5. Re-ranking
        reranked_results = await self._rerank(query, filtered_results)
        
        # 6. 다양성 보장
        diverse_results = self._ensure_diversity(reranked_results)
        
        # 7. 메타데이터 추가
        final_results = self._enrich_results(diverse_results[:k])
        
        # 8. 사용 통계 업데이트
        await self._update_usage_stats(final_results)
        
        return final_results
    
    def _build_filters(self, 
                      user_access_level: AccessLevel,
                      custom_filters: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """검색 필터 구성"""
        filters = {
            'must': [
                {
                    'key': 'access_level',
                    'match': {
                        'any': self._get_accessible_levels(user_access_level)
                    }
                }
            ]
        }
        
        # 커스텀 필터 추가
        if custom_filters:
            for key, value in custom_filters.items():
                filters['must'].append({
                    'key': key,
                    'match': {'value': value}
                })
        
        return filters
    
    def _get_accessible_levels(self, user_level: AccessLevel) -> List[str]:
        """접근 가능한 레벨 목록"""
        level_hierarchy = {
            AccessLevel.PUBLIC: [AccessLevel.PUBLIC.value],
            AccessLevel.INTERNAL: [AccessLevel.PUBLIC.value, AccessLevel.INTERNAL.value],
            AccessLevel.CONFIDENTIAL: [AccessLevel.PUBLIC.value, AccessLevel.INTERNAL.value, 
                                      AccessLevel.CONFIDENTIAL.value],
            AccessLevel.SECRET: [level.value for level in AccessLevel]
        }
        return level_hierarchy.get(user_level, [AccessLevel.PUBLIC.value])
    
    async def _apply_governance(self, 
                               search_results: List[Any]) -> List[Dict[str, Any]]:
        """거버넌스 정책 적용"""
        filtered = []
        
        for result in search_results:
            payload = result.payload
            
            # 품질 점수 확인
            if payload.get('quality_score', 0) < self.policies['min_quality_score']:
                continue
            
            # 최신성 확인
            created_at = datetime.fromisoformat(payload['created_at'])
            age_days = (datetime.now() - created_at).days
            if age_days > self.policies['max_age_days']:
                continue
            
            filtered.append({
                'id': result.id,
                'score': result.score,
                'text': payload['text'],
                'metadata': payload
            })
        
        return filtered
    
    async def _rerank(self, 
                     query: str,
                     results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Cross-Encoder를 사용한 Re-ranking"""
        if not results:
            return results
        
        # Cross-encoder 입력 준비
        pairs = [[query, r['text']] for r in results]
        
        # Re-ranking 점수 계산
        # ✅ 배치 사이즈/프로그레스바 제어
        rerank_batch = int(os.getenv("KXN_RERANK_BATCH", "16"))
        rerank_scores = self.reranker.predict(
            pairs,
            batch_size=rerank_batch,
            show_progress_bar=False
        )  
        
        # 원본 점수와 결합 (0.7 * vector_score + 0.3 * rerank_score)
        for i, result in enumerate(results):
            combined_score = 0.7 * result['score'] + 0.3 * rerank_scores[i]
            result['rerank_score'] = rerank_scores[i]
            result['combined_score'] = combined_score
        
        # 정렬
        results.sort(key=lambda x: x['combined_score'], reverse=True)
        
        return results
    
    def _ensure_diversity(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """결과 다양성 보장"""
        if len(results) <= 2:
            return results
        
        diverse_results = [results[0]]  # 첫 번째 결과는 항상 포함
        
        for candidate in results[1:]:
            # 기존 결과와의 유사도 체크
            is_diverse = True
            for selected in diverse_results:
                # 같은 문서에서 나온 청크인지 확인
                if candidate['metadata']['doc_id'] == selected['metadata']['doc_id']:
                    # 인접 청크가 아닌 경우만 추가
                    chunk_diff = abs(candidate['metadata']['chunk_index'] - 
                                   selected['metadata']['chunk_index'])
                    if chunk_diff < 2:
                        is_diverse = False
                        break
            
            if is_diverse:
                diverse_results.append(candidate)
        
        return diverse_results
    
    def _enrich_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """결과 메타데이터 보강"""
        enriched = []
        
        for result in results:
            enriched.append({
                'text': result['text'],
                'score': result.get('combined_score', result['score']),
                'source': result['metadata']['doc_source'],
                'title': result['metadata']['doc_title'],
                'chunk_index': result['metadata']['chunk_index'],
                'created_at': result['metadata']['created_at'],
                'quality_score': result['metadata']['quality_score'],
                'tags': result['metadata'].get('tags', [])
            })
        
        return enriched
    
    async def _update_usage_stats(self, results: List[Dict[str, Any]]):
        """사용 통계 업데이트"""
        # 실제 구현에서는 데이터베이스에 기록
        for result in results:
            logger.debug(f"Document used: {result['title']}")
    
    async def delete_expired_documents(self):
        """만료된 문서 삭제"""
        # 만료 날짜 기준으로 삭제
        cutoff_date = datetime.now() - timedelta(days=self.policies['max_age_days'])
        
        # Qdrant에서 삭제 (실제 구현 필요)
        logger.info(f"Deleting documents older than {cutoff_date}")
    
    async def update_quality_scores(self):
        """품질 점수 재계산"""
        # 사용 빈도, 피드백 등을 기반으로 품질 점수 업데이트
        logger.info("Updating quality scores based on usage patterns")
    
    def get_governance_report(self) -> Dict[str, Any]:
        """거버넌스 리포트 생성"""
        return {
            'policies': self.policies,
            'collection_stats': self.qdrant.get_collection(self.collection_name).dict(),
            'governance_status': 'active'
        }