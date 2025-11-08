import typer
import json

import yaml
from ..retrieval.hybrid import retrieve

app = typer.Typer()

@app.command()
def run(dataset: str="data/eval/dev.jsonl", cfg_path: str="configs/default.yaml"):
    from pathlib import Path
    cfg = yaml.safe_load(Path(cfg_path).read_text())
    rows = [json.loads(l) for l in Path(dataset).read_text().splitlines()]
    hits = 0
    for row in rows:
        top = retrieve(row["question"], cfg)
        doc_ids = {c["source"] for c in top}
        if any(g in doc_ids for g in row.get("gold_docs", [])):
            hits += 1
    print(f"Recall@{cfg['retrieval']['k_final']}: {hits/len(rows):.2f}")

if __name__ == "__main__":
    app()
