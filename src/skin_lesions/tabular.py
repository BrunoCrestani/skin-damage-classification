from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TABULAR_FEATURES = ["age", "sex", "localization", "dx_type"]
NUMERIC_FEATURES = ["age"]
CATEGORICAL_FEATURES = ["sex", "localization", "dx_type"]


@dataclass(frozen=True)
class DatasetMatrices:
    x_train: pd.DataFrame
    y_train: pd.Series
    x_val: pd.DataFrame
    y_val: pd.Series
    x_test: pd.DataFrame
    y_test: pd.Series


def split_tabular(df: pd.DataFrame) -> DatasetMatrices:
    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    test_df = df[df["split"] == "test"]
    return DatasetMatrices(
        x_train=train_df[TABULAR_FEATURES],
        y_train=train_df["dx"],
        x_val=val_df[TABULAR_FEATURES],
        y_val=val_df["dx"],
        x_test=test_df[TABULAR_FEATURES],
        y_test=test_df["dx"],
    )


def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def make_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("one_hot", make_one_hot_encoder()),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )


def build_tabular_models(random_state: int = 42) -> dict[str, Pipeline]:
    preprocessor = make_preprocessor()
    models = {
        "logistic_regression": LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            random_state=random_state,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=500,
            class_weight="balanced_subsample",
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=random_state,
        ),
    }
    return {
        name: Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
        for name, model in models.items()
    }

