from sentence_transformers import SentenceTransformer
from pathlib import Path
from numpy import np, json

def embed_chunks(chunks_path: str, model_name: str) -> np.ndarray:
    model = SentenceTransformer(model_name)
    chunks = [json.loads(l) for l in Path(chunks_path).read_text().splitlines()]
    texts = [c["content"] for c in chunks]
    embs = model.encode(texts, batchsize = 32, normalize_embeddings = True, show_progress_bar = True)
    np.save(Path(chunks_path).with_suffix(".npy"), embs)
    return embs
