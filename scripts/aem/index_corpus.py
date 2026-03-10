#!/usr/bin/env python3
"""Index poem corpus for AEM neighbor lookup (numpy-based, no Chroma).

Usage:
  python index_corpus.py --corpus data/raw/good/combined_clean.json --output data/aem_index
"""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=ROOT / "data" / "raw" / "good" / "combined_clean.json")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "aem_index")
    parser.add_argument("--limit", type=int, default=0, help="Max poems to index (0 = all)")
    args = parser.parse_args()

    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
    except ImportError:
        print("pip install sentence-transformers numpy", file=sys.stderr)
        sys.exit(1)

    model_path = os.environ.get("AEM_MODEL_PATH") or str(ROOT / "models" / "aem")
    model = SentenceTransformer(model_path) if Path(model_path).exists() else SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

    poems = []
    if args.corpus.exists():
        data = json.loads(args.corpus.read_text())
        for i, p in enumerate(data):
            if isinstance(p, dict) and p.get("poem"):
                text = p["poem"].strip() if isinstance(p["poem"], str) else ""
                if len(text) > 30:
                    poems.append({
                        "id": f"p{i}",
                        "title": p.get("title", ""),
                        "author": p.get("author", ""),
                        "text": text,
                    })
            if args.limit and len(poems) >= args.limit:
                break

    if not poems:
        print("No poems to index", file=sys.stderr)
        sys.exit(1)

    args.output.mkdir(parents=True, exist_ok=True)

    batch_size = 64
    all_embeddings = []
    metadata = []

    for i in range(0, len(poems), batch_size):
        batch = poems[i : i + batch_size]
        texts = [p["text"] for p in batch]
        emb = model.encode(texts, convert_to_numpy=True)
        all_embeddings.append(emb)
        metadata.extend([{"id": p["id"], "title": p["title"], "author": p["author"]} for p in batch])
        print(f"Indexed {min(i + batch_size, len(poems))}/{len(poems)}", file=sys.stderr)

    embeddings = np.vstack(all_embeddings).astype(np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    embeddings_norm = embeddings / norms

    np.save(args.output / "embeddings.npy", embeddings_norm)
    (args.output / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    print(f"Indexed {len(poems)} poems to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
