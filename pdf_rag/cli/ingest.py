import typer, yaml, numpy as np
from pathlib import Path
from ..io.pdf_extractor import extract_folder
from ..chunking.splitter import chunk_folder
from ..embed.encoder import embed_chunks
from ..index.faiss_store import build as build_faiss
from ..index.bm25_store import build as build_bm25

app = typer.Typer(help="RAG PDF pipeline")

def load_cfg(path="configs/default.yaml"):
    return yaml.safe_load(Path(path).read_text())

@app.command()
def ingest(cfg_path: str = "configs/default.yaml"):
    cfg = load_cfg(cfg_path)
    c = cfg["paths"]; r = cfg["retrieval"]; ch = cfg["chunking"]; m = cfg["models"]
    n = extract_folder(c["pdf_dir"], c["text_dir"])
    typer.echo(f"Extracted {n} PDFs")
    cnt = chunk_folder(c["text_dir"], c["chunk_path"], ch["size"], ch["overlap"])
    typer.echo(f"Chunked {cnt} chunks")
    embs = embed_chunks(c["chunk_path"], m["embed"])
    build_faiss(embs, c["faiss_path"])
    build_bm25(c["chunk_path"], c["bm25_path"])
    typer.echo("✅ Indexes built")

@app.command()
def query(q: str, cfg_path: str = "configs/default.yaml"):
    from ..retrieval.hybrid import retrieve
    from ..gen.llm import answer
    cfg = load_cfg(cfg_path)
    chunks = retrieve(q, cfg)
    print(answer(q, chunks, cfg["models"]["llm"]))

if __name__ == "__main__":
    app()
