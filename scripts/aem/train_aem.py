#!/usr/bin/env python3
"""Fine-tune sentence-transformers for aesthetic embeddings using triplet loss.

Usage:
  python train_aem.py --triplets data/aem_triplets.jsonl --output models/aem --epochs 5
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--triplets", type=Path, default=ROOT / "data" / "aem_triplets.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "models" / "aem")
    parser.add_argument("--backbone", default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--margin", type=float, default=0.2)
    args = parser.parse_args()

    try:
        from sentence_transformers import SentenceTransformer, InputExample, losses
        from torch.utils.data import DataLoader
    except ImportError:
        print("pip install sentence-transformers torch", file=sys.stderr)
        sys.exit(1)

    triplets: list[dict] = []
    with open(args.triplets) as f:
        for line in f:
            line = line.strip()
            if line:
                triplets.append(json.loads(line))

    if len(triplets) < 100:
        print(f"Need at least 100 triplets, got {len(triplets)}", file=sys.stderr)
        sys.exit(1)

    examples = [
        InputExample(texts=[t["anchor"], t["positive"], t["negative"]])
        for t in triplets
    ]

    model = SentenceTransformer(args.backbone)
    train_dataloader = DataLoader(examples, shuffle=True, batch_size=args.batch)
    train_loss = losses.TripletLoss(model=model, distance_metric=losses.TripletDistanceMetric.COSINE, triplet_margin=args.margin)

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=args.epochs,
        warmup_steps=100,
        output_path=str(args.output),
        show_progress_bar=True,
    )
    print(f"Saved to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
