#!/usr/bin/env python3
"""FastAPI service for AEM embedding inference.

Serves POST /embed with { text, mode: "full" | "trajectory" }.
Returns { embedding, trajectory?, neighbors? }.

Requires a trained model at models/aem/ or set AEM_MODEL_PATH.
For neighbors, run index_corpus.py first (creates data/aem_index/).
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="GPM AEM")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_model = None
_index_embeddings = None
_index_metadata = None


def _chunk_poem(text: str, chunk_size: int = 4) -> list[str]:
    """Split poem into stanza-like chunks of ~chunk_size lines."""
    lines = [l.strip() for l in text.split("\n") if l.strip() and not l.strip().startswith("#")]
    chunks = []
    for i in range(0, len(lines), chunk_size):
        chunk = "\n".join(lines[i : i + chunk_size])
        if chunk:
            chunks.append(chunk)
    if not chunks:
        chunks = [text] if text.strip() else []
    return chunks


def get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        path = os.environ.get("AEM_MODEL_PATH") or str(ROOT / "models" / "aem")
        if not Path(path).exists():
            path = "sentence-transformers/all-mpnet-base-v2"
        _model = SentenceTransformer(path)
    return _model


def get_index():
    global _index_embeddings, _index_metadata
    if _index_embeddings is None:
        index_path = Path(os.environ.get("AEM_INDEX_PATH", str(ROOT / "data" / "aem_index")))
        emb_path = index_path / "embeddings.npy"
        meta_path = index_path / "metadata.json"
        if not emb_path.exists() or not meta_path.exists():
            return None, None
        try:
            import numpy as np
            _index_embeddings = np.load(emb_path)
            _index_metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            return None, None
    return _index_embeddings, _index_metadata


class EmbedRequest(BaseModel):
    text: str
    mode: str = "full"  # "full" | "trajectory"
    k: int = 10


@app.post("/embed")
def embed(req: EmbedRequest):
    text = (req.text or "").strip()
    if not text:
        return {"embedding": [], "trajectory": [], "neighbors": [], "neighbor_embeddings": []}

    model = get_model()
    embedding = model.encode(text, convert_to_numpy=True)

    trajectory = []
    if req.mode == "trajectory":
        chunks = _chunk_poem(text)
        if chunks:
            trajectory = model.encode(chunks, convert_to_numpy=True).tolist()

    embedding_list = embedding.tolist()

    neighbors = []
    neighbor_embeddings = []
    emb, meta = get_index()
    if emb is not None and meta is not None and len(meta) > 0:
        import numpy as np
        q = embedding.astype(np.float32).reshape(1, -1)
        q_norm = q / (np.linalg.norm(q) or 1)
        scores = np.dot(emb, q_norm.T).flatten()
        k = min(req.k, 50, len(scores))
        top_indices = np.argsort(scores)[::-1][:k]
        for idx in top_indices:
            m = meta[int(idx)]
            neighbors.append({
                "poem_id": m.get("id", ""),
                "title": m.get("title", ""),
                "author": m.get("author", ""),
                "score": round(float(scores[idx]), 4),
            })
            neighbor_embeddings.append(emb[int(idx)].tolist())

    return {
        "embedding": embedding_list,
        "trajectory": trajectory,
        "neighbors": neighbors,
        "neighbor_embeddings": neighbor_embeddings,
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("AEM_PORT", 8001))
    uvicorn.run(app, host="0.0.0.0", port=port)
