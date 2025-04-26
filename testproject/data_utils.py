# -----------------------------------------------------------------------------
# data_utils.py – download FER2013 and prepare PyTorch datasets
# -----------------------------------------------------------------------------
import os, subprocess, csv, io, zipfile, contextlib
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

DATA_DIR = Path("data/fer2013")
CSV_FILE = DATA_DIR / "fer2013.csv"


def download_fer2013():
    """Download FER2013 using Kaggle API if not already present."""
    if CSV_FILE.exists():
        print("[data_utils] FER2013 already downloaded.")
        return
    print("[data_utils] Downloading FER2013 …")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    # Uses kaggle API – requires KAGGLE_USERNAME / KAGGLE_KEY env vars
    subprocess.check_call(
        [
            "kaggle",
            "datasets",
            "download",
            "-d",
            "msambare/fer2013",
            "-p",
            str(DATA_DIR),
        ]
    )
    # unzip
    zfile = next(DATA_DIR.glob("*.zip"))
    with zipfile.ZipFile(zfile, "r") as z:
        z.extractall(DATA_DIR)
    print("[data_utils] FER2013 downloaded & extracted.")


class FERDataset(Dataset):
    """PyTorch Dataset for FER2013. Returns (image, happiness_score)."""

    def __init__(self, csv_path: Path, split="Training", transform=None):
        df = pd.read_csv(csv_path)
        df = df[df["Usage"] == split]
        self.pixels = df["pixels"].tolist()
        self.labels = df["emotion"].tolist()  # 3 == happy
        self.transform = transform or transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((64, 64)),
                transforms.Grayscale(num_output_channels=3),  # duplicate channel
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        pixels = np.fromstring(self.pixels[idx], sep=" ", dtype="uint8").reshape(48, 48)
        img = self.transform(pixels)
        is_happy = 1 if self.labels[idx] == 3 else 0  # 3 = happy class
        # Score target: happy → 10, others → 1 (train as regression; later calibrate)
        target = torch.tensor(float(10 if is_happy else 1))
        return img, target