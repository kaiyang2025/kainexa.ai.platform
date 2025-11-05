본 POC는 제공하신 법령 원문(건설산업기본법, 하도급법)을 조항(조/항) 단위로 분리해 하이브리드 검색과 RAG 파이프라인을 시연합니다.
설계·일정·평가 흐름은 내부 제안서(건설 법령 AI 가이드 RAG 시스템) 구조를 반영했습니다.

0) 레포 구조
건설법령-RAG-POC-v2/
  data_raw/           # 제공 법령 원문(txt) ← 첨부본 자동 포함
  data_proc/          # 파싱/청킹 결과(jsonl)
  index/              # FAISS 인덱스/메타
  api/                # FastAPI (검색/RAG)
  app/                # Streamlit 데모 (UI)
  scripts/            # 파싱/인덱싱 스크립트
  eval/               # RAGAS 평가 스켈레톤
  prompts/            # 프롬프트 템플릿
  docker/opensearch.yml
  requirements.txt
  .env.example
  README.md
  Makefile

1) 빠른 실행 (Ubuntu, Python 3.11+)
# 0) 압축 해제
unzip 건설법령-RAG-POC-v2.zip
cd 건설법령-RAG-POC-v2

# 1) 가상환경 & 의존성
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# 2) OpenSearch 단일 노드
docker compose -f docker/opensearch.yml up -d

# 3) .env 설정
cp .env.example .env
# (선택) OPENAI_API_KEY 넣으면 생성도 LLM 호출로 동작
# 기본은 GEN_BACKEND=dummy 로 요약형 답변 생성

# 4) 데이터 파싱/청킹 (조/항 단위 데이터셋 생성)
python scripts/parse_and_chunk.py

# 5) 인덱싱 (OpenSearch + FAISS)
python scripts/build_index.py

# 6) 검색/RAG API 서버
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# 7) 데모 UI
streamlit run app/streamlit_app.py


데모 질문 예시

“발주자의 공사대금 지급보증 의무는?” → 건설산업기본법 제22조의2 근거 검색/답변.

“원사업자의 하도급대금 지급보증 의무는?” → 하도급법 제13조의2 근거 검색/답변.

2) 단계별 체크리스트 → 코드 매핑
1단계: 데이터수집·정제·청킹

파일: scripts/parse_and_chunk.py

핵심: 제\d+조(의\d+)? + ①②… 항 기반의 간단 규칙 파서(POC 최소) → data_proc/law_clauses.jsonl 생성

메타: law_name, clause_id(예: 제22조의2-①), title, text, effective_date, source_path

데이터: 제공된 원문을 그대로 포함(건설산업기본법: [시행 2025.08.28], 하도급법: [시행 2025.10.02]).

2단계: 인덱싱·검색엔진 구축

파일: scripts/build_index.py, docker/opensearch.yml

BM25: OpenSearch 인덱스(간단 매핑: title^2, text^1)

Dense: SentenceTransformer("BAAI/bge-m3") 임베딩 → FAISS(IP/코사인)

산출물: index/faiss.index, index/meta.json, index/docs.jsonl

3단계: RAG 모델·파이프라인

파일: api/ (config.py, search_utils.py, main.py), prompts/answer_system.txt

검색: BM25 + Dense → RRF 결합 → (옵션) BAAI/bge-reranker-large CrossEncoder 재랭킹

생성: /answer에서 상위 K개 문맥을 인용형 프롬프트로 구성

GEN_BACKEND=openai + OPENAI_API_KEY 있으면 OpenAI 호출

기본은 dummy로 즉시 시연 가능

4단계: 테스트코딩·평가·데모

UI: app/streamlit_app.py — 질의 입력 → 컨텍스트 카드 → 답변 + 인용 뱃지

평가: eval/ragas_eval.py — eval/questions_seed.jsonl 기반 RAGAS(answer_relevancy/faithfulness/context_precision/recall) 스켈레톤

보고서 템플릿: RAGAS 결과 JSON 출력(템플릿 자동화는 후속 과제)

3) 품질·운영 팁

모델 교체: .env에서 EMBED_MODEL, RERANK_MODEL 변경(예: intfloat/multilingual-e5-base)

하이브리드 가중치/TopK: .env의 BM25_K / DENSE_K / FINAL_K / LAMBDA_*

데이터 추가: data_raw/에 법령/예규 txt 추가 → parse → index 재실행

Reranker 옵셔널: 최초엔 끄고(BM25+FAISS만) → 필요 시 켜서 재현성 확인

4) 체크리스트(요약)

 조항단위 데이터셋 생성(조/항까지) — parse_and_chunk.py 실행.

 하이브리드 검색엔진(BM25+FAISS+RRF, 옵션 Reranker) — build_index.py & api/search_utils.py.

 RAG 생성 API & 데모 UI — api/main.py, app/streamlit_app.py(인용형 답변).

 RAGAS 평가 스켈레톤 — eval/ragas_eval.py / questions_seed.jsonl.

 제안서 흐름 반영 — 단계/산출물/지표 구성.

필요하시면, **법령·예규 추가 파서(항/호/목 계층, 부칙/표 처리)**와 버전·개정 추적, **리포트 자동 템플릿(docx)**까지 확장한 버전도 바로 만들어 드릴게요.