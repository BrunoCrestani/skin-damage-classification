#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from skin_lesions.config import ExperimentPaths
from skin_lesions.data import load_splits
from skin_lesions.image_embeddings import MODEL_CONFIGS, extract_embeddings


def parse_args():
    parser = argparse.ArgumentParser(description="Extract CNN embeddings for HAM10000 images.")
    parser.add_argument("--splits", type=Path, default=ExperimentPaths().splits_csv)
    parser.add_argument("--output-dir", type=Path, default=ExperimentPaths().processed_dir / "embeddings")
    parser.add_argument("--model", choices=sorted(MODEL_CONFIGS), default="resnet50")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = load_splits(args.splits)
    if not df["has_image"].all():
        missing = int((~df["has_image"]).sum())
        print(f"Warning: ignoring {missing} rows without image files.")

    for split in ["train", "val", "test"]:
        split_df = df[df["split"] == split]
        output_path = args.output_dir / args.model / f"{split}.npz"
        extract_embeddings(
            split_df,
            output_path=output_path,
            model_name=args.model,
            batch_size=args.batch_size,
            image_size=args.image_size,
            num_workers=args.num_workers,
        )
        print(f"Saved {split} embeddings: {output_path}")


if __name__ == "__main__":
    main()

