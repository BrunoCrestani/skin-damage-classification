#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from skin_lesions.config import ExperimentPaths
from skin_lesions.data import load_splits
from skin_lesions.metrics import evaluate_predictions
from skin_lesions.tabular import build_tabular_models, split_tabular


def parse_args():
    parser = argparse.ArgumentParser(description="Train tabular HAM10000 baselines.")
    parser.add_argument("--splits", type=Path, default=ExperimentPaths().splits_csv)
    parser.add_argument("--output-dir", type=Path, default=ExperimentPaths().output_dir / "tabular")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = load_splits(args.splits)
    matrices = split_tabular(df)
    models = build_tabular_models(random_state=args.seed)
    summary = []

    for name, model in models.items():
        model_dir = args.output_dir / name
        model_dir.mkdir(parents=True, exist_ok=True)

        model.fit(matrices.x_train, matrices.y_train)
        val_pred = model.predict(matrices.x_val)
        test_pred = model.predict(matrices.x_test)

        val_metrics = evaluate_predictions(matrices.y_val, val_pred, model_dir, "val")
        test_metrics = evaluate_predictions(matrices.y_test, test_pred, model_dir, "test")
        joblib.dump(model, model_dir / "model.joblib")

        row = {"model": name}
        row.update({f"val_{k}": v for k, v in val_metrics.items()})
        row.update({f"test_{k}": v for k, v in test_metrics.items()})
        summary.append(row)
        print(f"{name}: val_f1_macro={val_metrics['f1_macro']:.4f} test_f1_macro={test_metrics['f1_macro']:.4f}")

    pd.DataFrame(summary).sort_values("val_f1_macro", ascending=False).to_csv(
        args.output_dir / "summary.csv",
        index=False,
    )
    print(f"Saved summary: {args.output_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()

