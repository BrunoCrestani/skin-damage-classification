from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import (
    EfficientNet_B0_Weights,
    MobileNet_V3_Small_Weights,
    ResNet50_Weights,
    efficientnet_b0,
    mobilenet_v3_small,
    resnet50,
)
from tqdm import tqdm


MODEL_CONFIGS = {
    "resnet50": (resnet50, ResNet50_Weights.IMAGENET1K_V2, "fc"),
    "mobilenet_v3_small": (
        mobilenet_v3_small,
        MobileNet_V3_Small_Weights.IMAGENET1K_V1,
        "classifier",
    ),
    "efficientnet_b0": (efficientnet_b0, EfficientNet_B0_Weights.IMAGENET1K_V1, "classifier"),
}


class HamImageDataset(Dataset):
    def __init__(self, df: pd.DataFrame, image_size: int = 224, augment: bool = False):
        self.df = df[df["has_image"]].reset_index(drop=True)
        
        transform_list = []
        if augment:
            transform_list.extend([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
            ])
        transform_list.extend([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
        
        self.transform = transforms.Compose(transform_list)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int):
        row = self.df.iloc[index]
        image = Image.open(row["image_path"]).convert("RGB")
        return self.transform(image), row["dx"], row["image_id"]


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_feature_extractor(model_name: str, device: torch.device) -> nn.Module:
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model {model_name}. Options: {sorted(MODEL_CONFIGS)}")

    builder, weights, head_name = MODEL_CONFIGS[model_name]
    model = builder(weights=weights)
    setattr(model, head_name, nn.Identity())
    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model


def extract_embeddings(
    df: pd.DataFrame,
    output_path: Path,
    model_name: str = "resnet50",
    batch_size: int = 32,
    image_size: int = 224,
    num_workers: int = 2,
    augment: bool = False,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset = HamImageDataset(df, image_size=image_size, augment=augment)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    device = get_device()
    model = build_feature_extractor(model_name, device)

    features = []
    labels = []
    image_ids = []
    with torch.no_grad():
        for images, batch_labels, batch_image_ids in tqdm(loader, desc=f"Extracting {model_name}"):
            images = images.to(device)
            batch_features = model(images).detach().cpu().numpy()
            features.append(batch_features)
            labels.extend(batch_labels)
            image_ids.extend(batch_image_ids)

    if not features:
        raise ValueError("No image features were extracted. Check image paths in the split file.")

    np.savez_compressed(
        output_path,
        features=np.vstack(features),
        labels=np.asarray(labels),
        image_ids=np.asarray(image_ids),
        model_name=np.asarray([model_name]),
    )

