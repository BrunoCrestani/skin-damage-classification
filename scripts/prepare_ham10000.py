#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from skin_lesions.config import ExperimentPaths
from skin_lesions.data import load_ham10000, make_splits


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare stratified HAM10000 splits.")
    parser.add_argument("--raw-dir", type=Path, default=ExperimentPaths().raw_dir)
    parser.add_argument("--output", type=Path, default=ExperimentPaths().splits_csv)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-size", type=float, default=0.70)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Optional stratified sample size for quick local experiments.",
    )
    return parser.parse_args()


def stratified_sample(df: pd.DataFrame, sample_size: int, seed: int) -> pd.DataFrame:
    if sample_size >= len(df):
        return df
    fractions = sample_size / len(df)
    return (
        df.groupby("dx", group_keys=False)
        .apply(lambda group: group.sample(max(1, int(round(len(group) * fractions))), random_state=seed))
        .reset_index(drop=True)
    )


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    try:
        df = load_ham10000(args.raw_dir)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"Erro ao carregar o HAM10000: {exc}") from exc
    if args.sample_size:
        df = stratified_sample(df, args.sample_size, args.seed)

    splits = make_splits(
        df,
        train_size=args.train_size,
        val_size=args.val_size,
        seed=args.seed,
    )
    splits.to_csv(args.output, index=False)

    print(f"Saved splits: {args.output}")
    print(f"Rows: {len(splits)}")
    print("Split distribution:")
    print(pd.crosstab(splits["split"], splits["dx"]))
    missing_images = int((~splits["has_image"]).sum())
    if missing_images:
        print(f"Warning: {missing_images} rows do not have a matching image file.")


if __name__ == "__main__":
    main()
