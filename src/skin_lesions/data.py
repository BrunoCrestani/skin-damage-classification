from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import CLASSES


REQUIRED_METADATA_COLUMNS = {
    "lesion_id",
    "image_id",
    "dx",
    "dx_type",
    "age",
    "sex",
    "localization",
}


def find_metadata_csv(raw_dir: Path) -> Path:
    candidates = sorted(raw_dir.rglob("HAM10000_metadata.csv"))
    if not candidates:
        raise FileNotFoundError(
            f"HAM10000_metadata.csv not found under {raw_dir}. "
            "Download/extract the HAM10000 dataset first."
        )
    return candidates[0]


def index_images(raw_dir: Path) -> dict[str, Path]:
    image_paths = {}
    for path in raw_dir.rglob("*.jpg"):
        image_paths[path.stem] = path
    return image_paths


def load_ham10000(raw_dir: Path) -> pd.DataFrame:
    metadata_path = find_metadata_csv(raw_dir)
    df = pd.read_csv(metadata_path)

    missing = REQUIRED_METADATA_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"Missing metadata columns: {sorted(missing)}")

    image_paths = index_images(raw_dir)
    df["image_path"] = df["image_id"].map(lambda image_id: image_paths.get(str(image_id)))
    df = df[df["dx"].isin(CLASSES)].copy()
    df["has_image"] = df["image_path"].notna()
    df["image_path"] = df["image_path"].map(lambda p: str(p) if p is not None else "")
    return df


def make_splits(
    df: pd.DataFrame,
    train_size: float = 0.70,
    val_size: float = 0.15,
    seed: int = 42,
) -> pd.DataFrame:
    if not 0 < train_size < 1:
        raise ValueError("train_size must be between 0 and 1.")
    if not 0 < val_size < 1:
        raise ValueError("val_size must be between 0 and 1.")
    if train_size + val_size >= 1:
        raise ValueError("train_size + val_size must be lower than 1.")

    train_df, temp_df = train_test_split(
        df,
        train_size=train_size,
        stratify=df["dx"],
        random_state=seed,
    )
    relative_val_size = val_size / (1.0 - train_size)
    val_df, test_df = train_test_split(
        temp_df,
        train_size=relative_val_size,
        stratify=temp_df["dx"],
        random_state=seed,
    )

    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()
    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"
    return pd.concat([train_df, val_df, test_df], ignore_index=True)


def load_splits(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Split file not found: {path}. Run scripts/prepare_ham10000.py first."
        )
    return pd.read_csv(path)

