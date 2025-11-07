from pathlib import Path 
import json

def simple_chunks(text: str, size = 600, overlap = 100):
    i, n = 0, len(text)
    while i < n:
        yield text[i : i + size]
        i += size - overlap

def chunk_folder(text_dir : str, out_path: str, size = 600, overlap = 100) -> int:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cnt = 0
    with open(out_path, "w") as w:
        for txt in Path(text_dir).glob("*.txt"):
            content = txt.read_text()
            for j, chunk in enumerate(simple_chunks(content, size, overlap)):
                obj = {"id": f"{txt.stem}_{j}", "source": txt.stem, "content": chunk}
                w.write(json.dumps(obj, ensure_ascii=False) + "\n")
                cnt += 1
    return cnt