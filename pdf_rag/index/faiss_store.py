import faiss
import numpy as np
from typing import Tuple

def build(embs: np.ndarray, out_path: str) -> None:
    """Build a flat inner-product FAISS index and save it to disk."""
    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs.astype(np.float32))
    faiss.write_index(index, out_path)

def search(query_emb: np.ndarray, index_path: str, k: int = 30) -> Tuple[np.ndarray, np.ndarray]:
    """Search top-k most similar vectors.

    Returns:
        sims: (k,) similarity scores (inner product)
        idx:  (k,) indices of the matches in the original embedding matrix
    """
    index = faiss.read_index(index_path)
    sims_2d, idx_2d = index.search(query_emb[None, :].astype(np.float32), k)
    sims, idx = sims_2d[0], idx_2d[0]
    return sims, idx
