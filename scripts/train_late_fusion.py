#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from skin_lesions.config import CLASSES, ExperimentPaths
from skin_lesions.data import load_splits
from skin_lesions.image_models import load_embedding_npz
from skin_lesions.metrics import evaluate_predictions
from skin_lesions.tabular import build_tabular_models, split_tabular


def parse_args():
    parser = argparse.ArgumentParser(description="Late fusion of tabular and image probabilities.")
    parser.add_argument("--model", default="resnet50", help="CNN used to generate embeddings.")
    parser.add_argument("--embedding-dir", type=Path, default=ExperimentPaths().processed_dir / "embeddings")
    parser.add_argument("--output-dir", type=Path, default=ExperimentPaths().output_dir / "late_fusion")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--image-model-path",
        type=Path,
        default=None,
        help="Optional trained image classifier. Defaults to image logistic regression.",
    )
    return parser.parse_args()


def aligned_tabular(df: pd.DataFrame, image_ids: np.ndarray, split: str) -> pd.DataFrame:
    split_df = df[df["split"] == split].set_index("image_id")
    return split_df.loc[list(image_ids)].reset_index()


def average_probabilities(tab_probs, image_probs, tab_classes, image_classes, image_weight):
    tab_aligned = np.zeros((tab_probs.shape[0], len(CLASSES)))
    img_aligned = np.zeros((image_probs.shape[0], len(CLASSES)))
    for i, klass in enumerate(CLASSES):
        if klass in tab_classes:
            tab_aligned[:, i] = tab_probs[:, list(tab_classes).index(klass)]
        if klass in image_classes:
            img_aligned[:, i] = image_probs[:, list(image_classes).index(klass)]
    return (1 - image_weight) * tab_aligned + image_weight * img_aligned


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir / args.model
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_splits(ExperimentPaths().splits_csv)
    embedding_dir = args.embedding_dir / args.model
    x_train, y_train, train_ids = load_embedding_npz(embedding_dir / "train.npz")
    x_val, y_val, val_ids = load_embedding_npz(embedding_dir / "val.npz")
    x_test, y_test, test_ids = load_embedding_npz(embedding_dir / "test.npz")

    tab_df = pd.concat(
        [
            aligned_tabular(df, train_ids, "train"),
            aligned_tabular(df, val_ids, "val"),
            aligned_tabular(df, test_ids, "test"),
        ],
        ignore_index=True,
    )
    tabular_model = build_tabular_models(random_state=args.seed)["logistic_regression"]
    tabular = split_tabular(tab_df)
    tabular_model.fit(tabular.x_train, tabular.y_train)

    if args.image_model_path:
        image_model_path = args.image_model_path
    else:
        image_model_path = (
            ExperimentPaths().output_dir
            / "image_embeddings"
            / args.model
            / "logistic_regression"
            / "model.joblib"
        )
    image_model = joblib.load(image_model_path)

    val_tab_probs = tabular_model.predict_proba(tabular.x_val)
    test_tab_probs = tabular_model.predict_proba(tabular.x_test)
    val_img_probs = image_model.predict_proba(x_val)
    test_img_probs = image_model.predict_proba(x_test)

    summary = []
    best_weight = 0.5
    best_score = -1.0
    for image_weight in np.linspace(0.0, 1.0, 11):
        val_probs = average_probabilities(
            val_tab_probs,
            val_img_probs,
            tabular_model.classes_,
            image_model.classes_,
            image_weight,
        )
        val_pred = np.asarray(CLASSES)[val_probs.argmax(axis=1)]
        metrics = evaluate_predictions(y_val, val_pred, output_dir / f"weight_{image_weight:.1f}", "val")
        summary.append({"image_weight": image_weight, **{f"val_{k}": v for k, v in metrics.items()}})
        if metrics["f1_macro"] > best_score:
            best_score = metrics["f1_macro"]
            best_weight = float(image_weight)

    test_probs = average_probabilities(
        test_tab_probs,
        test_img_probs,
        tabular_model.classes_,
        image_model.classes_,
        best_weight,
    )
    test_pred = np.asarray(CLASSES)[test_probs.argmax(axis=1)]
    test_metrics = evaluate_predictions(y_test, test_pred, output_dir, "test")
    pd.DataFrame(summary).to_csv(output_dir / "weight_search.csv", index=False)
    pd.DataFrame([{"best_image_weight": best_weight, **test_metrics}]).to_csv(
        output_dir / "summary.csv",
        index=False,
    )
    joblib.dump(tabular_model, output_dir / "tabular_model.joblib")
    print(f"Best image weight on validation: {best_weight:.1f}")
    print(f"Test f1_macro={test_metrics['f1_macro']:.4f}")


if __name__ == "__main__":
    main()

