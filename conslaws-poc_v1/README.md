# 건설법령 RAG POC (mini sample repo, v2)

Ubuntu + Python 기반의 **아주 킥한** POC 코드 스켈레톤입니다.  
제공된 **건설산업기본법**과 **하도급법**을 조항(조/항) 단위로 파싱해 인덱싱하고,  
OpenSearch(BM25) + FAISS(Dense) 하이브리드 검색 → (옵션) Reranker → RAG 생성까지 한 번에 시연합니다.

- 데이터 출처(POC용): 건설산업기본법(시행 2025. 8. 28.), 하도급법(시행 2025. 10. 2.).

## 0) Prerequisites
- Python 3.11+ (`sudo apt-get install python3.11-venv`)
- Docker & Docker Compose
- (선택) OpenAI API Key (생성 백엔드 테스트 시)

## 1) Quickstart
```bash
# venv & install
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# OpenSearch up
docker compose -f docker/opensearch.yml up -d

# env
cp .env.example .env
# 필요시 OPENAI_API_KEY, 모델명 등 수정

# parse → chunk (조/항)
python scripts/parse_and_chunk.py

# index (OpenSearch + FAISS)
python scripts/build_index.py

# API (검색/RAG)
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# Streamlit demo
streamlit run app/streamlit_app.py
```

## 2) Structure
```
건설법령-RAG-POC-v2/
  data_raw/      # 제공 법령 원문(txt)
  data_proc/     # 파싱/청킹 결과(jsonl)
  index/         # FAISS 인덱스/메타
  api/           # FastAPI (검색/RAG)
  app/           # Streamlit 데모
  scripts/       # 파싱/인덱싱 스크립트
  eval/          # RAGAS 평가 스켈레톤
  prompts/       # 프롬프트 템플릿
  docker/opensearch.yml
  requirements.txt
  .env.example
  README.md
  Makefile
```

## 3) What’s included
- **Parser**: `제\d+조(의\d+)?` + ①②… 항 기반 간단 청킹 (POC 최소 규칙)
- **Hybrid Retrieval**: OpenSearch BM25 + FAISS cosine → RRF → (옵션) CrossEncoder rerank
- **RAG**: 상위 컨텍스트를 인용형 프롬프트로 묶어 생성(backend: openai | dummy)
- **UI**: Streamlit 실험용 화면(검색 결과 카드, 답변, 인용 뱃지, 컨텍스트)

## 4) Evaluate (sample)
- `eval/questions_seed.jsonl` : 샘플 질의 + gold citation
- `eval/ragas_eval.py` : answer_relevancy, faithfulness, context_precision/recall 측정 스켈레톤

> 본 레포는 **POC 최소구성**으로, 실제 운영 시에는 (1) 조/항/호/목 계층 파싱 강화, (2) 버전/개정 추적, (3) 가드레일·인용 표기 강화, (4) 리포트 자동화 등을 권장합니다.
