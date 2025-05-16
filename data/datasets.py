# ============================================================
# 🗂️  data/datasets.py
# ============================================================
"""Dataset utilities: balanced sampling + ``torch.utils.data.Dataset``."""
from collections import defaultdict
import os
import re
from typing import List

import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import WeightedRandomSampler, DataLoader

from .transforms import train_transform, val_transform
from config import DATA_ROOT, RANDOM_STATE, IMG_SIZE

__all__ = [
    "BalancedDataset",
    "CattleDataset",
    "build_dataloaders",
]

class BalancedDataset:
    """Walk a folder tree and compute class‑balanced train/val splits.

    Produces ``train_files`` / ``val_files`` and per‑class sampling weights
    so that the rarest class is sampled as frequently as the most common
    one.  The heavy lifting (counting images) happens **once** during
    ``__init__``.
    """

    def __init__(self, root_dir: str | os.PathLike = DATA_ROOT, *, test_size: float = 0.2):
        self.class_samples = defaultdict(list)
        root_dir = os.fspath(root_dir)

        for class_dir in os.listdir(root_dir):
            class_path = os.path.join(root_dir, class_dir)
            if not os.path.isdir(class_path):
                continue  # ignore stray files
            for img in os.listdir(class_path):
                self.class_samples[class_dir].append(os.path.join(class_path, img))

        # ---- compute sampling weights ----------------------------------
        sample_counts = [len(v) for v in self.class_samples.values()]
        max_count     = max(sample_counts)
        self.class_weights = [max_count / c for c in sample_counts]

        # ---- train/val split ------------------------------------------
        self.train_files: List[str] = []
        self.val_files:   List[str] = []
        for cls, files in self.class_samples.items():
            train, val = train_test_split(files, test_size=test_size, random_state=RANDOM_STATE)
            self.train_files.extend(train)
            self.val_files.extend(val)

class CattleDataset(torch.utils.data.Dataset):
    """Torch Dataset that infers the label from the parent folder name."""
    def __init__(self, file_list: List[str], *, train: bool):
        self.file_list = file_list
        self.transform = train_transform if train else val_transform
        classes        = sorted({os.path.basename(os.path.dirname(f)) for f in file_list})
        self.class_to_idx = {cls: i for i, cls in enumerate(classes)}

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        image    = Image.open(img_path).convert("RGB")
        class_id = self.class_to_idx[os.path.basename(os.path.dirname(img_path))]
        image    = self.transform(image)
        return image, class_id

# ------------------------------------------------------------
# Helper: build dataloaders in one function (for train.py)
# ------------------------------------------------------------

# def build_dataloaders(batch_size: int, num_workers: int = 8):
#     ds        = BalancedDataset()
#     train_ds  = CattleDataset(ds.train_files, train=True)
#     val_ds    = CattleDataset(ds.val_files,   train=False)
#
#     sample_weights = [
#         ds.class_weights[train_ds.class_to_idx[os.path.basename(os.path.dirname(f))]]
#         for f in ds.train_files
#     ]
#     sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
#
#     train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=num_workers)
#     val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,  num_workers=num_workers)
#     return train_loader, val_loader, train_ds.class_to_idx

def build_dataloaders(batch_size: int, num_workers: int = 8):
    ds        = BalancedDataset()
    train_ds  = CattleDataset(ds.train_files, train=True)
    val_ds    = CattleDataset(ds.val_files,   train=False)

    # 采样权重
    sample_weights = [
        ds.class_weights[train_ds.class_to_idx[os.path.basename(os.path.dirname(f))]]
        for f in ds.train_files
    ]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,  num_workers=num_workers)

    # ---------------------------------------
    # 🐄 数据统计：类别 → 照片数 & 牛的唯一数量
    # ---------------------------------------
    def extract_cow_id(file_path):
        """假设文件名如 cow123_img01.jpg → 提取 cow123"""
        fname = os.path.basename(file_path)
        match = re.match(r"(cow\d+)", fname.lower())
        return match.group(1) if match else "unknown"

    def summarize(file_list):
        class_to_imgs = defaultdict(list)
        class_to_cows = defaultdict(set)
        for f in file_list:
            cls = os.path.basename(os.path.dirname(f))
            cow_id = extract_cow_id(f)
            class_to_imgs[cls].append(f)
            class_to_cows[cls].add(cow_id)
        summary = {
            cls: {
                "photos": len(imgs),
                "unique_cows": len(cows)
            }
            for cls, (imgs, cows) in zip(class_to_imgs.keys(), zip(class_to_imgs.values(), class_to_cows.values()))
        }
        return summary

    # print("📊 Dataset split summary:")
    # print("🔹 Train set:", summarize(ds.train_files))
    # print("🔹 Val set:  ", summarize(ds.val_files))

    return train_loader, val_loader, train_ds.class_to_idx
