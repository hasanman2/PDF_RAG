import json
import numpy as np
from pathlib import Path
from sentence_transformers import Sentencestranformer
from ..index.bm25_store import scores as sparse_scores
from ..index.faiss_store import search as dense_search

def load(chunks_path: str):
    return [json.loads(l) for l in Path(chunks_path).read_text().splitlines()]

def retrieve(query: str, cfg: dict):
    chunks_path = cfg["paths"]["chunks_path"]
    chunks = load(chunks_path)
    embed_model = Sentencestranformer(cfg["model"]["embed"])
    q_emb = embed_model.encode([query], normalize_embeddings=True)[0]
    D, I = dense_search(q_emb, cfg["paths"]["faiss_path"], cfg["retrieval"]["k_initial"])
    dense = {i:int_rank for int_rank,i in enumerate(I)}  # position-based proxy
    bm25 = sparse_scores(query, chunks_path, cfg["paths"]["bm25_path"])
    alpha = cfg["retrieval"]["alpha"]
    final = {
        i: alpha * (1 - (dense.get(i, len(chunks))/max(1,len(chunks)))) + (1-alpha) * bm25[i]
        for i in range(len(chunks))
    }
    topk = cfg["retrieval"]["k_final"]
    top = sorted(final.items(), key=lambda x: x[1], reverse=True)[:topk]
    return [chunks[i] for i,_ in top]