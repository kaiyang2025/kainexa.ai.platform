up:
	docker compose -f docker/opensearch.yml up -d

parse:
	python scripts/parse_and_chunk.py

index:
	python scripts/build_index.py

api:
	uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

app:
	streamlit run app/streamlit_app.py

eval:
	python eval/ragas_eval.py
