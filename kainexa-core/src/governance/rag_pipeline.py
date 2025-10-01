# src/governance/rag_pipeline.py
"""
Kainexa RAG Pipeline - 완전한 구현
벡터 DB 연동, 문서 처리, 임베딩, 검색, 재순위화, 컨텍스트 주입
"""
import asyncio
import hashlib
import json
import time
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import numpy as np
from datetime import datetime, timedelta
import structlog
import re
from types import SimpleNamespace
import tiktoken

# Vector DB imports
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue,
    SearchRequest, SearchParams
)

# Document processing
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, TextLoader, UnstructuredMarkdownLoader
import pandas as pd

logger = structlog.get_logger()

# ========== Enums ==========
class DocumentType(Enum):
    """문서 타입"""
    PDF = "pdf"
    TEXT = "text"
    MARKDOWN = "markdown"
    HTML = "html"
    CSV = "csv"
    JSON = "json"
    DOCX = "docx"

class ChunkingStrategy(Enum):
    """청킹 전략"""
    FIXED_SIZE = "fixed_size"          # 고정 크기
    SEMANTIC = "semantic"              # 의미 단위
    SENTENCE = "sentence"              # 문장 단위
    PARAGRAPH = "paragraph"            # 단락 단위
    SLIDING_WINDOW = "sliding_window"  # 슬라이딩 윈도우

class SearchStrategy(Enum):
    """검색 전략"""
    SIMILARITY = "similarity"          # 유사도
    MMR = "mmr"                       # Maximum Marginal Relevance
    HYBRID = "hybrid"                 # 하이브리드 (키워드 + 벡터)
    RERANK = "rerank"                 # 재순위화

# ========== Data Classes ==========
@dataclass
class Document:
    """문서"""
    id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    embeddings: Optional[List[float]] = None
    chunks: List['DocumentChunk'] = field(default_factory=list)
    
    @property
    def source(self) -> str:
        return self.metadata.get('source', 'unknown')
    
    @property
    def created_at(self) -> datetime:
        return self.metadata.get('created_at', datetime.now())

@dataclass
class DocumentChunk:
    """문서 청크"""
    id: str
    document_id: str
    content: str
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    position: int = 0
    overlap_with_previous: int = 0
    
    @property
    def token_count(self) -> int:
        # 간단한 토큰 추정 (실제로는 tokenizer 사용)
        return len(self.content.split())

@dataclass
class SearchQuery:
    """검색 쿼리"""
    text: str
    top_k: int = 5
    strategy: SearchStrategy = SearchStrategy.SIMILARITY
    filters: Dict[str, Any] = field(default_factory=dict)
    boost_recent: bool = True
    include_metadata: bool = True
    min_score: float = 0.5
    rerank: bool = False

@dataclass
class SearchResult:
    """검색 결과"""
    chunk: DocumentChunk
    score: float
    rerank_score: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def final_score(self) -> float:
        return self.rerank_score if self.rerank_score is not None else self.score

@dataclass
class RAGContext:
    """RAG 컨텍스트"""
    query: str
    retrieved_chunks: List[SearchResult]
    enhanced_prompt: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    processing_time_ms: float = 0

# ========== Document Processor ==========
class DocumentProcessor:
    """문서 처리기"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.splitters = self._initialize_splitters()
        self.token_encoder = tiktoken.get_encoding("cl100k_base")
        
    def _initialize_splitters(self) -> Dict[ChunkingStrategy, Any]:
        """텍스트 분할기 초기화"""
        return {
            ChunkingStrategy.FIXED_SIZE: RecursiveCharacterTextSplitter(
                chunk_size=self.config.get('chunk_size', 500),
                chunk_overlap=self.config.get('chunk_overlap', 50),
                separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
                length_function=len
            ),
            ChunkingStrategy.SEMANTIC: RecursiveCharacterTextSplitter(
                chunk_size=self.config.get('semantic_chunk_size', 800),
                chunk_overlap=self.config.get('semantic_overlap', 100),
                separators=["\n\n", "\n", ".", " "],
                length_function=self._semantic_length
            ),
            ChunkingStrategy.SENTENCE: RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=0,
                separators=[".", "!", "?", "。", "！", "？"],
                length_function=len
            ),
            ChunkingStrategy.PARAGRAPH: RecursiveCharacterTextSplitter(
                chunk_size=2000,
                chunk_overlap=200,
                separators=["\n\n", "\n"],
                length_function=len
            )
        }
    
    async def process_document(self, 
                              file_path: str,
                              doc_type: DocumentType,
                              chunking_strategy: ChunkingStrategy = ChunkingStrategy.FIXED_SIZE,
                              metadata: Optional[Dict[str, Any]] = None) -> Document:
        """문서 처리"""
        
        # 문서 로드
        content = await self._load_document(file_path, doc_type)
        
        # 문서 ID 생성
        doc_id = self._generate_document_id(file_path, content)
        
        # 메타데이터 생성
        doc_metadata = {
            'source': file_path,
            'type': doc_type.value,
            'created_at': datetime.now(),
            'chunking_strategy': chunking_strategy.value,
            'original_length': len(content),
            **(metadata or {})
        }
        
        # 문서 객체 생성
        document = Document(
            id=doc_id,
            content=content,
            metadata=doc_metadata
        )
        
        # 청킹
        chunks = await self._chunk_document(document, chunking_strategy)
        document.chunks = chunks
        
        logger.info(
            f"Processed document: {file_path}",
            chunks=len(chunks),
            strategy=chunking_strategy.value
        )
        
        return document
    
    async def _load_document(self, file_path: str, doc_type: DocumentType) -> str:
        """문서 로드"""
        
        if doc_type == DocumentType.PDF:
            loader = PyPDFLoader(file_path)
            pages = await asyncio.to_thread(loader.load)
            return "\n\n".join([page.page_content for page in pages])
            
        elif doc_type == DocumentType.TEXT:
            loader = TextLoader(file_path)
            docs = await asyncio.to_thread(loader.load)
            return docs[0].page_content if docs else ""
            
        elif doc_type == DocumentType.MARKDOWN:
            loader = UnstructuredMarkdownLoader(file_path)
            docs = await asyncio.to_thread(loader.load)
            return docs[0].page_content if docs else ""
            
        elif doc_type == DocumentType.CSV:
            df = await asyncio.to_thread(pd.read_csv, file_path)
            return df.to_string()
            
        elif doc_type == DocumentType.JSON:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return json.dumps(data, ensure_ascii=False, indent=2)
            
        else:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
    
    async def _chunk_document(self, 
                            document: Document,
                            strategy: ChunkingStrategy) -> List[DocumentChunk]:
        """문서 청킹"""
        
        splitter = self.splitters.get(strategy, self.splitters[ChunkingStrategy.FIXED_SIZE])
        
        # 텍스트 분할
        if strategy == ChunkingStrategy.SLIDING_WINDOW:
            chunks = self._sliding_window_chunk(
                document.content,
                window_size=self.config.get('window_size', 500),
                step_size=self.config.get('step_size', 250)
            )
        else:
            chunks = splitter.split_text(document.content)
        
        # DocumentChunk 객체 생성
        doc_chunks = []
        for i, chunk_text in enumerate(chunks):
            chunk = DocumentChunk(
                id=f"{document.id}_chunk_{i}",
                document_id=document.id,
                content=chunk_text,
                metadata={
                    **document.metadata,
                    'chunk_index': i,
                    'total_chunks': len(chunks),
                    'token_count': len(self.token_encoder.encode(chunk_text))
                },
                position=i
            )
            
            # 이전 청크와의 겹침 계산
            if i > 0 and strategy != ChunkingStrategy.SENTENCE:
                prev_chunk = doc_chunks[-1]
                overlap = self._calculate_overlap(prev_chunk.content, chunk.content)
                chunk.overlap_with_previous = overlap
            
            doc_chunks.append(chunk)
        
        return doc_chunks
    
    def _sliding_window_chunk(self, text: str, window_size: int, step_size: int) -> List[str]:
        """슬라이딩 윈도우 청킹"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), step_size):
            chunk = ' '.join(words[i:i + window_size])
            if chunk:
                chunks.append(chunk)
            if i + window_size >= len(words):
                break
        
        return chunks
    
    def _calculate_overlap(self, text1: str, text2: str) -> int:
        """텍스트 겹침 계산"""
        words1 = set(text1.split()[-50:])  # 마지막 50단어
        words2 = set(text2.split()[:50])   # 처음 50단어
        return len(words1.intersection(words2))
    
    def _semantic_length(self, text: str) -> int:
        """의미적 길이 계산"""
        # 토큰 수 + 문장 경계 고려
        tokens = self.token_encoder.encode(text)
        sentences = text.count('.') + text.count('!') + text.count('?')
        return len(tokens) + sentences * 10
    
    def _generate_document_id(self, file_path: str, content: str) -> str:
        """문서 ID 생성"""
        hash_input = f"{file_path}:{len(content)}:{content[:100]}"
        return hashlib.sha256(hash_input.encode()).hexdigest()[:16]

# ========== Embedding Generator ==========
class EmbeddingGenerator:
    """임베딩 생성기"""
    
    def __init__(self, model_name: str = "text-embedding-ada-002"):
        self.model_name = model_name
        self.embedding_cache = {}
        self.batch_size = 32
        
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """임베딩 생성"""
        embeddings = []
        
        # 배치 처리
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_embeddings = await self._generate_batch(batch)
            embeddings.extend(batch_embeddings)
        
        return embeddings
    
    async def generate_embedding(self, text: str) -> List[float]:
        """단일 텍스트 임베딩 생성"""
        
        # 캐시 확인
        cache_key = hashlib.md5(text.encode()).hexdigest()
        if cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]
        
        # 생성
        embedding = await self._generate_single(text)
        
        # 캐시 저장
        self.embedding_cache[cache_key] = embedding
        
        return embedding
    
    async def _generate_batch(self, texts: List[str]) -> List[List[float]]:
        """배치 임베딩 생성 (실제 구현 필요)"""
        # 실제 구현에서는 OpenAI API 또는 로컬 모델 사용
        # 여기서는 시뮬레이션
        await asyncio.sleep(0.1)
        
        embeddings = []
        for text in texts:
            # 랜덤 임베딩 (시뮬레이션)
            embedding = np.random.randn(1536).tolist()
            embeddings.append(embedding)
        
        return embeddings
    
    async def _generate_single(self, text: str) -> List[float]:
        """단일 임베딩 생성"""
        embeddings = await self._generate_batch([text])
        return embeddings[0]

# ========== Vector Store (Qdrant) ==========
class VectorStore:
    """벡터 저장소 (Qdrant)"""
    
    def __init__(self, 
                 host: str = "localhost",
                 port: int = 6333,
                 collection_name: str = "kainexa_docs"):
        self.client = QdrantClient(host=host, port=port)
        self.collection_name = collection_name
        self.embedding_generator = EmbeddingGenerator()
        self._ensure_collection()
    
    def _ensure_collection(self):
        """컬렉션 확인/생성"""
        collections = self.client.get_collections().collections
        
        if not any(c.name == self.collection_name for c in collections):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=1536,  # OpenAI embedding size
                    distance=Distance.COSINE
                )
            )
            logger.info(f"Created collection: {self.collection_name}")
    
    async def index_chunks(self, chunks: List[DocumentChunk]):
        """청크 인덱싱"""
        
        # 임베딩 생성
        texts = [chunk.content for chunk in chunks]
        embeddings = await self.embedding_generator.generate_embeddings(texts)
        
        # 포인트 생성
        points = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            chunk.embedding = embedding
            
            point = PointStruct(
                id=hashlib.md5(chunk.id.encode()).hexdigest()[:16],
                vector=embedding,
                payload={
                    'chunk_id': chunk.id,
                    'document_id': chunk.document_id,
                    'content': chunk.content,
                    'metadata': chunk.metadata,
                    'position': chunk.position,
                    'created_at': datetime.now().isoformat()
                }
            )
            points.append(point)
        
        # 배치 업로드
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        
        logger.info(f"Indexed {len(chunks)} chunks")
    
    async def search(self, query: SearchQuery) -> List[SearchResult]:
        """벡터 검색"""
        
        # 쿼리 임베딩 생성
        query_embedding = await self.embedding_generator.generate_embedding(query.text)
        
        # 필터 생성
        search_filter = self._build_filter(query.filters) if query.filters else None
        
        # 검색 파라미터
        search_params = SearchParams(
            hnsw_ef=128,
            exact=False
        )
        
        # 검색 실행
        if query.strategy == SearchStrategy.MMR:
            results = await self._search_mmr(
                query_embedding, 
                query.top_k,
                search_filter
            )
        elif query.strategy == SearchStrategy.HYBRID:
            results = await self._search_hybrid(
                query.text,
                query_embedding,
                query.top_k,
                search_filter
            )
        else:  # SIMILARITY
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=query.top_k * 2 if query.rerank else query.top_k,
                query_filter=search_filter,
                search_params=search_params
            )
        
        # SearchResult 객체로 변환
        search_results = []
        for result in results:
            if result.score >= query.min_score:
                chunk = DocumentChunk(
                    id=result.payload['chunk_id'],
                    document_id=result.payload['document_id'],
                    content=result.payload['content'],
                    metadata=result.payload['metadata'],
                    position=result.payload['position']
                )
                
                search_result = SearchResult(
                    chunk=chunk,
                    score=result.score,
                    metadata={
                        'created_at': result.payload.get('created_at'),
                        'distance': 1 - result.score  # Cosine similarity to distance
                    }
                )
                
                # 최신성 부스팅
                if query.boost_recent:
                    search_result.score = self._boost_recent_score(
                        search_result.score,
                        result.payload.get('created_at')
                    )
                
                search_results.append(search_result)
        
        # 재순위화
        if query.rerank:
            search_results = await self._rerank_results(query.text, search_results)
        
        # Top-K 선택
        search_results.sort(key=lambda x: x.final_score, reverse=True)
        return search_results[:query.top_k]
    
    async def _search_mmr(self, 
                         query_embedding: List[float],
                         top_k: int,
                         search_filter: Optional[Filter]) -> List[Any]:
        """Maximum Marginal Relevance 검색"""
        
        # 초기 후보 검색 (2배수)
        candidates = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=top_k * 3,
            query_filter=search_filter
        )
        
        if not candidates:
            return []
        
        # MMR 선택
        selected = []
        selected_embeddings = []
        lambda_param = 0.7  # 다양성 파라미터
        
        while len(selected) < top_k and candidates:
            mmr_scores = []
            
            for candidate in candidates:
                if candidate.id in [s.id for s in selected]:
                    continue
                
                # 쿼리 유사도
                query_sim = candidate.score
                
                # 선택된 문서들과의 최대 유사도
                if selected_embeddings:
                    max_sim = max(
                        self._cosine_similarity(
                            candidate.vector,
                            selected_emb
                        )
                        for selected_emb in selected_embeddings
                    )
                else:
                    max_sim = 0
                
                # MMR 점수
                mmr_score = lambda_param * query_sim - (1 - lambda_param) * max_sim
                mmr_scores.append((candidate, mmr_score))
            
            if mmr_scores:
                # 최고 MMR 점수 선택
                best = max(mmr_scores, key=lambda x: x[1])
                selected.append(best[0])
                selected_embeddings.append(best[0].vector)
                candidates.remove(best[0])
        
        return selected
    
    async def _search_hybrid(self,
                           query_text: str,
                           query_embedding: List[float],
                           top_k: int,
                           search_filter: Optional[Filter]) -> List[Any]:
        """
        하이브리드 검색 (키워드 + 벡터)
        1) 벡터 유사도로 후보군을 넉넉히 가져온 뒤
        2) 질의어 토큰의 등장 빈도로 키워드 점수를 계산
        3) 두 점수를 가중 결합하여 상위 top_k 반환
        """
        # 1) 벡터 후보 (여유 있게 N배수)
        candidate_multiplier = 4
        candidates = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=max(top_k * candidate_multiplier, top_k),
            query_filter=search_filter,
            search_params=SearchParams(hnsw_ef=128, exact=False),
        ) or []

        if not candidates:
            return []

        # 2) 키워드 토큰화 (한/영/숫자 기준)
        tokens = [t for t in re.split(r"[^0-9A-Za-z가-힣]+", query_text) if t]
        tokens = [t for t in tokens if len(t) > 1]  # 단일 문자 노이즈 제거

        # 벡터 점수 정규화 준비
        v_scores = [c.score for c in candidates]
        v_min, v_max = min(v_scores), max(v_scores)
        def norm_vec(x: float) -> float:
            return 0.0 if v_max == v_min else (x - v_min) / (v_max - v_min)

        # 3) 키워드 점수 계산 + 결합
        kw_counts = []
        for c in candidates:
            text = c.payload.get("content", "") if hasattr(c, "payload") else ""
            # 간단한 포함 횟수 기반 점수
            count = 0
            if tokens and text:
                # 대소문자/한글 그대로 match (요구 시 lower() 적용)
                for tok in tokens:
                    if tok:
                        count += text.count(tok)
            kw_counts.append(count)

        kw_max = max(kw_counts) if kw_counts else 0
        def norm_kw(x: int) -> float:
            return 0.0 if kw_max == 0 else (x / kw_max)

        alpha = 0.7  # 벡터 가중
        beta = 0.3   # 키워드 가중

        combined = []
        for c, kw in zip(candidates, kw_counts):
            combined_score = alpha * norm_vec(c.score) + beta * norm_kw(kw)
            combined.append(
                SimpleNamespace(
                    id=getattr(c, "id", None),
                    score=combined_score,
                    payload=getattr(c, "payload", {}),
                    vector=getattr(c, "vector", None),
                )
            )

        combined.sort(key=lambda x: x.score, reverse=True)
        return combined[:top_k]