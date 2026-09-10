.PHONY: install ingest seed run-api run-ui test clean

install:
	pip install -r requirements.txt

# Seed the corpus with the bundled demo documents (creates chroma_db)
seed:
	python -m app.scripts.seed_data

# Ingest your own files placed in ./data/documents
ingest:
	python -m app.scripts.ingest --dir ./data/documents

# Start the FastAPI backend
run-api:
	uvicorn app.api.main:app --host 0.0.0.0 --port 8000 --reload

# Start the Streamlit UI
run-ui:
	streamlit run app/ui/streamlit_app.py

# Run the full stack (API + UI) via the launcher
run:
	python scripts/launch.py

test:
	python -m pytest tests/ -v

clean:
	rm -rf chroma_db traces __pycache__ .pytest_cache
