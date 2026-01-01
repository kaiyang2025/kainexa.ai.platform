import json
import pandas as pd
import numpy as np
import torch
import os
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Any

# 라이브러리
from rank_bm25 import BM25Okapi
from kiwipiepy import Kiwi
from sentence_transformers import SentenceTransformer, CrossEncoder

# 로컬 모듈
from config_model import RAGConfig
from graph_engine import LegalGraphRetriever

# ---------------------------------------------------------
# 1. BM25 Searcher (Notebook 1.1.13 - ConstructionBM25)
# ---------------------------------------------------------
class ConstructionBM25:
    def __init__(self, glossary_path: str = None):
        self.kiwi = Kiwi()
        self.bm25 = None
        self.docs = []
        self.CIRCLED_TO_ARABIC = dict(zip("①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳", map(str, range(1, 21))))
        
        if glossary_path and os.path.exists(glossary_path):
            self._load_glossary(glossary_path)

    def _load_glossary(self, path):
        try:
            df = pd.read_csv(path)
            terms = df['term'].dropna().astype(str).str.strip().unique().tolist()
            for term in terms:
                self.kiwi.add_user_word(term, tag='NNG', score=10)
        except Exception as e:
            print(f"[Warn] Glossary load failed: {e}")

    def _normalize(self, text: str) -> str:
        import re
        text = re.sub(r"제\s*(\d+)\s*조\s*의\s*(\d+)", r"제\1조의\2", text)
        text = re.sub(r"제\s*(\d+)\s*조", r"제\1조", text)
        text = re.sub(r"제\s*(\d+)\s*항", r"제\1항", text)
        for char, digit in self.CIRCLED_TO_ARABIC.items():
            text = text.replace(char, digit)
        return text

    def tokenize(self, text: str) -> List[str]:
        text = self._normalize(text)
        tokens = []
        for res in self.kiwi.analyze(text):
            for token, tag, _, _ in res[0]:
                if tag.startswith('N') or tag in ('SL', 'SN', 'XPN', 'XR'):
                    tokens.append(token)
        return tokens

    def fit(self, jsonl_path: str):
        print(f"[BM25] Loading data from {jsonl_path}...")
        self.docs = [json.loads(line) for line in Path(jsonl_path).read_text(encoding="utf-8").splitlines() if line.strip()]
        corpus_tokens = [self.tokenize(f'{d.get("clause_id","")} {d.get("text","")}') for d in self.docs]
        self.bm25 = BM25Okapi(corpus_tokens, k1=1.2, b=0.75)
        print("[BM25] Indexing Done.")

    def search(self, query: str, topn: int = 10):
        if not self.bm25: return []
        tokens = self.tokenize(query)
        scores = self.bm25.get_scores(tokens)
        top_indices = np.argsort(-scores)[:topn]
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                doc = self.docs[idx].copy()
                doc["bm25_score"] = float(scores[idx])
                results.append(doc)
        return results

# ---------------------------------------------------------
# 2. Dense Retriever (Notebook 1.1.13 - DenseRetriever)
# ---------------------------------------------------------
class DenseRetriever:
    def __init__(self, model_name="BAAI/bge-m3"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[Dense] Loading model {model_name} on {self.device}...")
        self.model = SentenceTransformer(model_name, device=self.device)
        self.doc_embeddings = None
        self.docs = []

    def encode_documents(self, docs: List[Dict]):
        self.docs = docs
        texts = [d.get("text", "") for d in docs]
        if texts:
            print(f"[Dense] Encoding {len(texts)} documents...")
            self.doc_embeddings = self.model.encode(texts, batch_size=32, normalize_embeddings=True)

    def search(self, query: str, top_k: int = 5):
        if self.doc_embeddings is None: return []
        q_emb = self.model.encode([query], normalize_embeddings=True)[0]
        scores = self.doc_embeddings @ q_emb
        top_indices = np.argsort(-scores)[:top_k]
        results = []
        for idx in top_indices:
            doc = self.docs[idx].copy()
            doc["dense_score"] = float(scores[idx])
            results.append(doc)
        return results

# ---------------------------------------------------------
# 3. Main Pipeline Controller (Notebook - RAGSearchPipeline)
# ---------------------------------------------------------
class ConstructionRAG:
    def __init__(self, data_path, glossary_path):
        # 1. Load Engines
        self.bm25_engine = ConstructionBM25(glossary_path)
        self.bm25_engine.fit(data_path) # Load & Index
        
        self.dense_engine = DenseRetriever()
        # Share docs with Dense engine and encode
        self.dense_engine.encode_documents(self.bm25_engine.docs) 
        
        # 2. Reranker
        self.ce_model = CrossEncoder("BAAI/bge-reranker-v2-m3", device="cuda" if torch.cuda.is_available() else "cpu")
        
        # 3. Graph
        self.graph_retriever = LegalGraphRetriever()
        
        # 4. Glossary Data for expansion
        self.glossary_df = pd.read_csv(glossary_path) if os.path.exists(glossary_path) else None

    def _expand_glossary(self, query):
        if self.glossary_df is None: return query
        expanded = query
        for _, row in self.glossary_df.iterrows():
            if str(row['term']) in query:
                expanded += f" {row['definition']}"
        return expanded

    def _rrf_merge(self, bm25_res, dense_res, k=60):
        # ... (Notebook의 rrf_merge 로직과 동일) ...
        # 지면 관계상 핵심 로직만 간소화하여 작성
        ranks = defaultdict(float)
        docs_map = {}
        
        def process(res_list, rank_key):
            for i, doc in enumerate(res_list):
                did = doc.get('id', str(doc.get('text')[:10]))
                docs_map[did] = doc
                ranks[did] += 1 / (k + i + 1)
        
        process(bm25_res, 'bm25')
        process(dense_res, 'dense')
        
        merged = []
        for did, score in ranks.items():
            doc = docs_map[did].copy()
            doc['fused_score'] = score
            merged.append(doc)
        merged.sort(key=lambda x: -x['fused_score'])
        return merged

    def run_pipeline(self, query: str, config: RAGConfig):
        # 1. 용어집 확장
        search_query = self._expand_glossary(query) if config.use_glossary else query
        
        # 2. Graph DB 확장
        graph_context = ""
        if config.use_graph_db:
            ctxs = self.graph_retriever.expand_query_with_graph(query)
            if ctxs:
                graph_context = " ".join(ctxs)
                print(f"[Log] Graph Context Found: {len(ctxs)} items")

        # 3. Retrieval
        bm25_hits = self.bm25_engine.search(search_query, topn=config.top_k * 3)
        dense_hits = self.dense_engine.search(search_query, top_k=config.top_k * 3)
        
        # 4. RRF Merge
        candidates = self._rrf_merge(bm25_hits, dense_hits)
        
        # 5. Reranking (CrossEncoder)
        final_results = candidates
        if config.use_reranker and candidates:
            # 리랭커에 그래프 컨텍스트 주입 가능
            rerank_query = f"{query} [참고] {graph_context[:300]}" if graph_context else query
            
            top_n_cands = candidates[:50]
            pairs = [[rerank_query, d.get('text', '')] for d in top_n_cands]
            scores = self.ce_model.predict(pairs)
            
            for i, doc in enumerate(top_n_cands):
                doc['rerank_score'] = float(scores[i])
            
            top_n_cands.sort(key=lambda x: -x['rerank_score'])
            final_results = top_n_cands
            
        return final_results[:config.top_k], graph_context