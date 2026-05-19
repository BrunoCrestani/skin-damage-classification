#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
import numpy as np

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
    parser.add_argument(
        "--augment-train",
        action="store_true",
        help="Extract and concatenate augmented features for the training split.",
    )
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
        
        if split == "train" and args.augment_train:
            print("--- Extracting base training embeddings ---")
            temp_normal_path = output_path.with_name("train_normal.npz")
            extract_embeddings(
                split_df,
                output_path=temp_normal_path,
                model_name=args.model,
                batch_size=args.batch_size,
                image_size=args.image_size,
                num_workers=args.num_workers,
                augment=False,
            )
            print("--- Extracting augmented training embeddings (Alternative B) ---")
            temp_aug_path = output_path.with_name("train_augmented.npz")
            extract_embeddings(
                split_df,
                output_path=temp_aug_path,
                model_name=args.model,
                batch_size=args.batch_size,
                image_size=args.image_size,
                num_workers=args.num_workers,
                augment=True,
            )
            
            # Load, merge and save
            norm_data = np.load(temp_normal_path, allow_pickle=False)
            aug_data = np.load(temp_aug_path, allow_pickle=False)
            
            merged_features = np.vstack([norm_data["features"], aug_data["features"]])
            merged_labels = np.concatenate([norm_data["labels"], aug_data["labels"]])
            merged_ids = np.concatenate([norm_data["image_ids"], aug_data["image_ids"]])
            
            np.savez_compressed(
                output_path,
                features=merged_features,
                labels=merged_labels,
                image_ids=merged_ids,
                model_name=norm_data["model_name"],
            )
            
            # Clean up
            temp_normal_path.unlink(missing_ok=True)
            temp_aug_path.unlink(missing_ok=True)
            print(f"Saved doubled train embeddings with Data Augmentation: {output_path}")
        else:
            extract_embeddings(
                split_df,
                output_path=output_path,
                model_name=args.model,
                batch_size=args.batch_size,
                image_size=args.image_size,
                num_workers=args.num_workers,
                augment=False,
            )
            print(f"Saved {split} embeddings: {output_path}")


if __name__ == "__main__":
    main()

