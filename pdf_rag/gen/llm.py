import subprocess

def answer(query: str, chunks: list, model_name: str="mistral") -> str:
    context = "\n\n".join([f"[{c['id']}] {c['content']}" for c in chunks])
    prompt = f"""Use only the context to answer. If insufficient, say so.
Question: {query}

Context:
{context}

Cite as [chunk_id]."""
    res = subprocess.run(["ollama","run",model_name], input=prompt, text=True, capture_output=True)
    return res.stdout.strip()
