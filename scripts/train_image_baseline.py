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
from sklearn.model_selection import GridSearchCV


def parse_args():
    parser = argparse.ArgumentParser(description="Train tuned classifiers on CNN embeddings.")
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
    
    param_grids = {
        "logistic_regression": {
            "model__C": [0.01, 0.1, 1.0, 10.0]
        },
        "knn": {
            "model__n_neighbors": [3, 5, 7, 11]
        },
        "random_forest": {
            "n_estimators": [100, 300, 500],
            "min_samples_leaf": [1, 2, 4]
        }
    }

    summary = []
    for name, clf in models.items():
        model_dir = output_dir / name
        model_dir.mkdir(parents=True, exist_ok=True)

        print(f"Otimizando classificador {name} sobre embeddings...")
        grid_search = GridSearchCV(
            estimator=clf,
            param_grid=param_grids[name],
            cv=5,
            scoring="f1_macro",
            n_jobs=-1
        )
        
        grid_search.fit(x_train, y_train)
        best_model = grid_search.best_estimator_
        print(f"Melhores parâmetros para {name}: {grid_search.best_params_}")

        val_pred = best_model.predict(x_val)
        val_prob = best_model.predict_proba(x_val)
        
        test_pred = best_model.predict(x_test)
        test_prob = best_model.predict_proba(x_test)
        
        val_metrics = evaluate_predictions(y_val, val_pred, model_dir, "val", y_prob=val_prob)
        test_metrics = evaluate_predictions(y_test, test_pred, model_dir, "test", y_prob=test_prob)
        
        joblib.dump(best_model, model_dir / "model.joblib")

        row = {"embedding_model": args.model, "classifier": name, "best_params": str(grid_search.best_params_)}
        row.update({f"val_{k}": v for k, v in val_metrics.items()})
        row.update({f"test_{k}": v for k, v in test_metrics.items()})
        summary.append(row)
        
        print(f"{name} Tunado: val_f1_macro={val_metrics['f1_macro']:.4f} val_auc_macro={val_metrics.get('roc_auc_macro', 0.0):.4f}")

    pd.DataFrame(summary).sort_values("val_f1_macro", ascending=False).to_csv(
        output_dir / "summary.csv",
        index=False,
    )
    print(f"Saved summary: {output_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
