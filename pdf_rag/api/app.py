# rag_pdf/api/app.py
from fastapi import FastAPI, Query
import yaml
from pathlib import Path
from ..retrieval.hybrid import retrieve
from ..gen.llm import answer

app = FastAPI()

CFG = yaml.safe_load(Path("configs/default.yaml").read_text())

@app.get("/ask")
def ask(q: str = Query(..., min_length=3)):
    chunks = retrieve(q, CFG)
    out = answer(q, chunks, CFG["models"]["llm"])
    return {"answer": out, "chunks": [c["id"] for c in chunks]}
