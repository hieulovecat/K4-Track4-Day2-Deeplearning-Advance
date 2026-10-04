"""dataset.py - đọc DeepWeeds, kiểm tra chia dữ liệu, transform, DataLoader.

Quy tắc chia dữ liệu bắt buộc (S1-S6) nằm ở README.md, mục 2.1.

Giao diện:
    load_split(labels_dir, fold=0)            -> (train_df, val_df, test_df)
    check_split(train_df, val_df, test_df, images_dir) -> dict  (số liệu để ghi báo cáo)
    build_transforms(train, img_size, aug)    -> torchvision transform
    DeepWeedsDataset[i]                       -> (image_tensor, label:int, filename:str)
    make_loader(df, images_dir, transform, batch_size, train, sampler, num_workers)
"""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms as T

NUM_CLASSES = 9
TOTAL_IMAGES = 17509
# Thứ tự lớp theo cột `Label` của labels.csv (0 = Chinee Apple ... 7 = Snake Weed, 8 = Negative).
CLASS_NAMES = [
    "Chinee Apple", "Lantana", "Parkinsonia", "Parthenium", "Prickly Acacia",
    "Rubber Vine", "Siam Weed", "Snake Weed", "Negatives",
]
IMAGENET_MEAN = (0.485, 0.456, 0.406)  # run() lấy mean/std đúng theo pretrained_cfg của backbone
IMAGENET_STD = (0.229, 0.224, 0.225)
RAW_SIZE = 256  # ảnh gốc DeepWeeds là 256x256


# --------------------------------------------------------------------------- #
# Đọc và kiểm tra split
# --------------------------------------------------------------------------- #
def load_split(labels_dir: str | Path, fold: int = 0):
    """Đọc train/val/test_subset{fold}.csv (S1). Không sửa, lọc hay chia lại."""
    d = Path(labels_dir)
    out = []
    for part in ("train", "val", "test"):
        p = d / f"{part}_subset{fold}.csv"
        if not p.exists():
            raise FileNotFoundError(f"không thấy {p}; hãy tải CSV nguyên bản của tác giả (README mục 2)")
        df = pd.read_csv(p)
        if not {"Filename", "Label"} <= set(df.columns):
            raise ValueError(f"{p}: cần cột Filename và Label, thấy {list(df.columns)}")
        out.append(df)
    return tuple(out)


def check_split(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
                images_dir: str | Path, verbose: bool = True) -> dict:
    """Kiểm tra bắt buộc trước khi train (README.md, mục 2.1); lỗi thì raise ngay."""
    dfs = {"train": train_df, "val": val_df, "test": test_df}
    n = {k: len(v) for k, v in dfs.items()}
    total = sum(n.values())
    per_class = {k: v["Label"].value_counts().reindex(range(NUM_CLASSES), fill_value=0).tolist()
                 for k, v in dfs.items()}
    ratio = {k: n[k] / total for k in n}

    sets = {k: set(v["Filename"]) for k, v in dfs.items()}
    for k, v in dfs.items():
        if len(sets[k]) != len(v):
            raise AssertionError(f"{k}: có Filename trùng lặp")
    overlap = {"train∩val": len(sets["train"] & sets["val"]),
               "train∩test": len(sets["train"] & sets["test"]),
               "val∩test": len(sets["val"] & sets["test"])}
    union = len(sets["train"] | sets["val"] | sets["test"])

    existing = set(os.listdir(images_dir)) if Path(images_dir).is_dir() else set()
    missing = sorted(f for s in sets.values() for f in s if f not in existing)

    if verbose:
        print(f"Số ảnh: {n}  (tổng {total}; tỉ lệ " + ", ".join(f"{k} {r:.1%}" for k, r in ratio.items()) + ")")
        tbl = pd.DataFrame(per_class, index=CLASS_NAMES).T
        print("Số ảnh mỗi lớp:\n" + tbl.to_string())
        print("Giao các cặp tập:", overlap, "| hợp:", union, "| thiếu file:", len(missing))

    if any(overlap.values()):
        raise AssertionError(f"các tập giao nhau: {overlap}")
    if union != TOTAL_IMAGES:
        raise AssertionError(f"hợp ba tập = {union}, kỳ vọng {TOTAL_IMAGES}")
    if missing:
        raise AssertionError(f"{len(missing)} file trong CSV không có trong {images_dir}, ví dụ {missing[:3]}")
    for k, target in (("train", 0.6), ("val", 0.2), ("test", 0.2)):
        if abs(ratio[k] - target) > 0.01:
            raise AssertionError(f"tỉ lệ {k} = {ratio[k]:.3f} lệch quá 1 điểm % so với {target}; báo giảng viên")
    return {"n": n, "ratio": ratio, "per_class": per_class, "overlap": overlap,
            "union": union, "missing": len(missing)}


# --------------------------------------------------------------------------- #
# Transform
# --------------------------------------------------------------------------- #
def build_transforms(train: bool, img_size: int = 224, aug: str = "basic",
                     mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """Tạo transform.

    Train: RandomResizedCrop(img_size, scale=(0.25, 1)) + lật ngang, rồi (tuỳ `aug`):
      - "basic"  : chỉ vậy
      - "color"  : + ColorJitter (độ sáng/tương phản/bão hoà/sắc độ)
      - "trivial": + TrivialAugmentWide
      - "randaug": + RandAugment(2, 9)
    - "flipv"  : + lật dọc (ảnh chụp từ trên xuống nên lật dọc vẫn hợp lệ về mặt vật lý)
    Val/test: ảnh gốc 256 -> CenterCrop(img_size) nếu img_size < 256, Resize(img_size) nếu >= 256
    (img_size = 256 nghĩa là dùng nguyên ảnh). KHÔNG augmentation ngẫu nhiên.
    """
    norm = [T.ToTensor(), T.Normalize(mean, std)]
    if not train:
        if img_size < RAW_SIZE:
            return T.Compose([T.Resize(RAW_SIZE), T.CenterCrop(img_size)] + norm)
        return T.Compose([T.Resize((img_size, img_size))] + norm)

    ops = [T.RandomResizedCrop(img_size, scale=(0.25, 1.0)), T.RandomHorizontalFlip()]
    if aug == "basic":
        pass
    elif aug == "flipv":
        ops.append(T.RandomVerticalFlip())
    elif aug == "color":
        ops.append(T.ColorJitter(0.3, 0.3, 0.3, 0.05))
    elif aug == "trivial":
        ops.append(T.TrivialAugmentWide())
    elif aug == "randaug":
        ops.append(T.RandAugment(num_ops=2, magnitude=9))
    else:
        raise ValueError(f"aug không hợp lệ: {aug!r} (basic|flipv|color|trivial|randaug)")
    return T.Compose(ops + norm)


# --------------------------------------------------------------------------- #
# Dataset, DataLoader
# --------------------------------------------------------------------------- #
class DeepWeedsDataset(Dataset):
    """Đọc ảnh từ `images_dir` theo DataFrame (Filename, Label); trả (tensor, int label, filename)."""

    def __init__(self, df: pd.DataFrame, images_dir: str | Path, transform=None):
        self.filenames = df["Filename"].tolist()
        self.labels = df["Label"].astype(int).tolist()
        self.images_dir = Path(images_dir)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, i: int):
        name = self.filenames[i]
        with Image.open(self.images_dir / name) as im:
            img = im.convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, self.labels[i], name


def _worker_init(worker_id: int):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def make_loader(df: pd.DataFrame, images_dir: str | Path, transform, batch_size: int,
                train: bool, sampler: str | None = None, num_workers: int = 2, seed: int = 0):
    """Tạo DataLoader.

    train=True: shuffle (hoặc sampler cân bằng), drop_last; train=False: giữ nguyên thứ tự df.
    sampler: None | "balanced" (WeightedRandomSampler, trọng số 1/số ảnh của lớp).
    """
    ds = DeepWeedsDataset(df, images_dir, transform)
    g = torch.Generator()
    g.manual_seed(seed)
    kw = dict(num_workers=num_workers, pin_memory=torch.cuda.is_available(),
              worker_init_fn=_worker_init, generator=g,
              persistent_workers=num_workers > 0)
    if not train:
        return DataLoader(ds, batch_size=batch_size, shuffle=False, **kw)
    if sampler == "balanced":
        counts = df["Label"].value_counts()
        w = df["Label"].map(lambda c: 1.0 / counts[c]).to_numpy()
        smp = WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), num_samples=len(ds),
                                    replacement=True, generator=g)
        return DataLoader(ds, batch_size=batch_size, sampler=smp, drop_last=True, **kw)
    if sampler is not None:
        raise ValueError(f"sampler không hợp lệ: {sampler!r} (None|balanced)")
    return DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True, **kw)
