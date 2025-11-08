import json
import pickle
import re
from pathlib import Path
from rank_bm25 import BM25Okapi
from typing import List

TOKEN_RE = re.compile(r"[A-Za-z0-9]+")

def tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(text.lower())

def build(chunks_path: str, out_path: str) -> None:
    lines = Path(chunks_path).read_text(encoding="utf-8").splitlines()
    chunks = [json.loads(l) for l in lines]
    tokenized = [tokenize(c["content"]) for c in chunks]
    bm25 = BM25Okapi(tokenized)  # optionally: k1=1.5, b=0.75
    with open(out_path, "wb") as f:
        pickle.dump({"bm25": bm25, "ids": [c["id"] for c in chunks]}, f)

def scores(query: str, bm25_path: str):
    data = pickle.load(open(bm25_path, "rb"))
    bm25 = data["bm25"]
    return bm25.get_scores(tokenize(query)), data.get("ids")
