#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd
import numpy as np

# Prevent matplotlib from requiring a GUI backend
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from skin_lesions.config import ExperimentPaths
from skin_lesions.data import load_splits
from skin_lesions.metrics import evaluate_predictions
from skin_lesions.tabular import (
    build_tabular_models,
    split_tabular,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
)
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def parse_args():
    parser = argparse.ArgumentParser(description="Train tabular HAM10000 baselines with tuning.")
    parser.add_argument("--splits", type=Path, default=ExperimentPaths().splits_csv)
    parser.add_argument("--output-dir", type=Path, default=ExperimentPaths().output_dir / "tabular")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def random_over_sample(x: pd.DataFrame, y: pd.Series, seed: int = 42) -> tuple[pd.DataFrame, pd.Series]:
    df_temp = pd.concat([x, y], axis=1)
    target_col = y.name
    max_size = df_temp[target_col].value_counts().max()
    
    resampled_groups = []
    for class_name, group in df_temp.groupby(target_col):
        resampled_group = group.sample(max_size, replace=True, random_state=seed)
        resampled_groups.append(resampled_group)
        
    df_resampled = pd.concat(resampled_groups, ignore_index=True)
    return df_resampled.drop(columns=[target_col]), df_resampled[target_col]


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = load_splits(args.splits)
    matrices = split_tabular(df)
    
    # 1. ESTUDO COMPARATIVO DE DESBALANCEAMENTO DE DADOS (Exigência Acadêmica)
    print("=== Estudo Comparativo de Desbalanceamento (Random Forest) ===")
    imbalance_results = []
    
    # Estratégia A: Baseline Sem Correção
    rf_baseline = build_tabular_models(random_state=args.seed)["random_forest"]
    rf_baseline.named_steps["model"].set_params(class_weight=None)
    rf_baseline.fit(matrices.x_train, matrices.y_train)
    bl_val_pred = rf_baseline.predict(matrices.x_val)
    bl_metrics = evaluate_predictions(matrices.y_val, bl_val_pred, args.output_dir / "imbalance_study" / "baseline", "val")
    imbalance_results.append({"Estratégia": "Sem Correção (Baseline)", "val_f1_macro": bl_metrics["f1_macro"], "val_accuracy": bl_metrics["accuracy"]})
    
    # Estratégia B: Classificação Sensível ao Custo (Pesos)
    rf_weighted = build_tabular_models(random_state=args.seed)["random_forest"]
    rf_weighted.fit(matrices.x_train, matrices.y_train)
    w_val_pred = rf_weighted.predict(matrices.x_val)
    w_metrics = evaluate_predictions(matrices.y_val, w_val_pred, args.output_dir / "imbalance_study" / "weighted", "val")
    imbalance_results.append({"Estratégia": "Sensível ao Custo (Pesos)", "val_f1_macro": w_metrics["f1_macro"], "val_accuracy": w_metrics["accuracy"]})
    
    # Estratégia C: Sobreamostragem (Random Over-sampling manual)
    x_resampled, y_resampled = random_over_sample(matrices.x_train, matrices.y_train, seed=args.seed)
    rf_ros = build_tabular_models(random_state=args.seed)["random_forest"]
    rf_ros.named_steps["model"].set_params(class_weight=None)
    rf_ros.fit(x_resampled, y_resampled)
    ros_val_pred = rf_ros.predict(matrices.x_val)
    ros_metrics = evaluate_predictions(matrices.y_val, ros_val_pred, args.output_dir / "imbalance_study" / "ros", "val")
    imbalance_results.append({"Estratégia": "Sobreamostragem (ROS)", "val_f1_macro": ros_metrics["f1_macro"], "val_accuracy": ros_metrics["accuracy"]})
    
    # Salvar e mostrar comparação
    df_imbalance = pd.DataFrame(imbalance_results)
    df_imbalance.to_csv(args.output_dir / "imbalance_comparison.csv", index=False)
    print(df_imbalance.to_string(index=False))
    print(f"Estudo comparativo de desbalanceamento salvo em: {args.output_dir / 'imbalance_comparison.csv'}\n")

    # 2. OTIMIZAÇÃO DE HIPERPARÂMETROS (GridSearchCV)
    print("=== Ajuste de Hiperparâmetros (GridSearchCV) ===")
    models = build_tabular_models(random_state=args.seed)
    
    param_grids = {
        "logistic_regression": {
            "model__C": [0.01, 0.1, 1.0, 10.0]
        },
        "random_forest": {
            "model__n_estimators": [100, 300, 500],
            "model__max_depth": [None, 10, 20],
            "model__min_samples_leaf": [1, 2, 4]
        }
    }

    summary = []

    for name, pipeline in models.items():
        model_dir = args.output_dir / name
        model_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Otimizando {name}...")
        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grids[name],
            cv=5,
            scoring="f1_macro",
            n_jobs=-1
        )
        
        grid_search.fit(matrices.x_train, matrices.y_train)
        best_model = grid_search.best_estimator_
        print(f"Melhores parâmetros para {name}: {grid_search.best_params_}")
        
        # Predições e Probabilidades
        val_pred = best_model.predict(matrices.x_val)
        val_prob = best_model.predict_proba(matrices.x_val)
        
        test_pred = best_model.predict(matrices.x_test)
        test_prob = best_model.predict_proba(matrices.x_test)

        # Avaliação com as probabilidades para AUC-ROC
        val_metrics = evaluate_predictions(matrices.y_val, val_pred, model_dir, "val", y_prob=val_prob)
        test_metrics = evaluate_predictions(matrices.y_test, test_pred, model_dir, "test", y_prob=test_prob)
        
        # Salvar melhor modelo tunado
        joblib.dump(best_model, model_dir / "model.joblib")

        row = {"model": name, "best_params": str(grid_search.best_params_)}
        row.update({f"val_{k}": v for k, v in val_metrics.items()})
        row.update({f"test_{k}": v for k, v in test_metrics.items()})
        summary.append(row)
        
        print(f"[{name}] Tunado: val_f1_macro={val_metrics['f1_macro']:.4f} val_auc_macro={val_metrics.get('roc_auc_macro', 0.0):.4f}")

        # 3. EXPLICABILIDADE & IMPORTÂNCIA DE ATRIBUTOS
        preprocessor = best_model.named_steps["preprocessor"]
        feature_names = NUMERIC_FEATURES + list(
            preprocessor.named_transformers_["cat"]
            .named_steps["one_hot"]
            .get_feature_names_out(CATEGORICAL_FEATURES)
        )

        if name == "random_forest":
            print(f"[{name}] Gerando importância intrínseca de características...")
            importances = best_model.named_steps["model"].feature_importances_
            indices = np.argsort(importances)
            
            plt.figure(figsize=(10, 6))
            plt.title("Importância das Características - Random Forest")
            plt.barh(range(len(indices)), importances[indices], align="center")
            plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
            plt.xlabel("Importância Relativa")
            plt.tight_layout()
            plt.savefig(model_dir / "feature_importance.png", dpi=160)
            plt.close()
            print(f"[{name}] Gráfico de importância salvo em: {model_dir / 'feature_importance.png'}")
            
            # Integração com SHAP
            try:
                import shap
                print(f"[{name}] Gerando valores SHAP...")
                x_val_proc = preprocessor.transform(matrices.x_val)
                explainer = shap.TreeExplainer(best_model.named_steps["model"])
                shap_values = explainer.shap_values(x_val_proc)
                
                plt.figure(figsize=(10, 6))
                shap.summary_plot(shap_values, x_val_proc, feature_names=feature_names, show=False)
                plt.tight_layout()
                plt.savefig(model_dir / "shap_summary.png", dpi=160)
                plt.close()
                print(f"[{name}] Gráfico SHAP salvo em: {model_dir / 'shap_summary.png'}")
            except Exception as e:
                print(f"[{name}] Ignorando SHAP: {e}")

    # Salvar resumo dos baselines tunados
    pd.DataFrame(summary).sort_values("val_f1_macro", ascending=False).to_csv(
        args.output_dir / "summary.csv",
        index=False,
    )
    print(f"\nResumo final dos baselines de metadados tabulares salvo em: {args.output_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
