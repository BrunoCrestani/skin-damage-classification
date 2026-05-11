#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from skin_lesions.config import ExperimentPaths
from skin_lesions.image_models import build_embedding_models, load_embedding_npz
from skin_lesions.metrics import evaluate_predictions


def parse_args():
    parser = argparse.ArgumentParser(description="Train classifiers on CNN embeddings.")
    parser.add_argument("--model", default="resnet50")
    parser.add_argument("--embedding-dir", type=Path, default=ExperimentPaths().processed_dir / "embeddings")
    parser.add_argument("--output-dir", type=Path, default=ExperimentPaths().output_dir / "image_embeddings")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    embedding_dir = args.embedding_dir / args.model
    output_dir = args.output_dir / args.model
    output_dir.mkdir(parents=True, exist_ok=True)

    x_train, y_train, _ = load_embedding_npz(embedding_dir / "train.npz")
    x_val, y_val, _ = load_embedding_npz(embedding_dir / "val.npz")
    x_test, y_test, _ = load_embedding_npz(embedding_dir / "test.npz")

    models = build_embedding_models(random_state=args.seed)
    summary = []
    for name, clf in models.items():
        model_dir = output_dir / name
        model_dir.mkdir(parents=True, exist_ok=True)

        clf.fit(x_train, y_train)
        val_pred = clf.predict(x_val)
        test_pred = clf.predict(x_test)
        val_metrics = evaluate_predictions(y_val, val_pred, model_dir, "val")
        test_metrics = evaluate_predictions(y_test, test_pred, model_dir, "test")
        joblib.dump(clf, model_dir / "model.joblib")

        row = {"embedding_model": args.model, "classifier": name}
        row.update({f"val_{k}": v for k, v in val_metrics.items()})
        row.update({f"test_{k}": v for k, v in test_metrics.items()})
        summary.append(row)
        print(f"{name}: val_f1_macro={val_metrics['f1_macro']:.4f} test_f1_macro={test_metrics['f1_macro']:.4f}")

    pd.DataFrame(summary).sort_values("val_f1_macro", ascending=False).to_csv(
        output_dir / "summary.csv",
        index=False,
    )
    print(f"Saved summary: {output_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()

