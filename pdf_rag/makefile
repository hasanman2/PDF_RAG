.PHONY: setup ingest query eval api

setup:
	python -m venv .venv && . .venv/bin/activate && pip install -e .

ingest:
	rag-pdf ingest

query:
	rag-pdf query "What is reinforcement learning?"

eval:
	python -m rag_pdf.cli.evaluate run

api:
	uvicorn rag_pdf.api.app:app --reload --port 8000
