"""checks.py - kiểm tra pipeline trước khi chạy thật (GUIDE.md mục 1.3, slide trang 59)."""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn

import dataset as D
import model as M
import train as TR


def initial_loss(backbone: str, loader, device, n_batches: int = 3) -> float:
    """Loss CE ban đầu của head mới; phải xấp xỉ ln 9 = 2.197."""
    TR.set_seed(0)
    model = M.build_model(backbone, pretrained=True).to(device).eval()
    ce, losses = nn.CrossEntropyLoss(), []
    with torch.no_grad():
        for i, (x, y, _) in enumerate(loader):
            if i >= n_batches:
                break
            losses.append(ce(model(x.to(device)).float(), y.to(device)).item())
    val = float(np.mean(losses))
    print(f"loss ban đầu = {val:.3f} (kỳ vọng ≈ ln 9 = {math.log(9):.3f})")
    return val


def overfit_one_batch(backbone: str, loader, device, steps: int = 60, lr: float = 1e-3) -> float:
    """Huấn luyện lặp lại trên MỘT batch nhỏ (16 ảnh) tới loss gần 0. Dùng BN ở eval để ổn định."""
    TR.set_seed(0)
    model = M.build_model(backbone, pretrained=True).to(device)
    x, y, _ = next(iter(loader))
    x, y = x[:16].to(device), y[:16].to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    ce = nn.CrossEntropyLoss()
    model.eval()                      # tắt dropout/BN-train: kiểm tra khả năng ghi nhớ, không phải khái quát
    last = None
    for s in range(steps):
        opt.zero_grad()
        loss = ce(model(x).float(), y)
        loss.backward()
        opt.step()
        last = loss.item()
        if s % 10 == 0 or s == steps - 1:
            print(f"  bước {s:02d}: loss {last:.4f}")
    return last


def show_augmented(df, images_dir, aug: str = "basic", n: int = 8, img_size: int = 224, path=None):
    """Vẽ ảnh sau augmentation (đã giải chuẩn hoá) cùng nhãn để chắc ảnh và nhãn khớp nhau."""
    import matplotlib.pyplot as plt

    tf = D.build_transforms(True, img_size, aug)
    ds = D.DeepWeedsDataset(df.sample(n, random_state=0), images_dir, tf)
    mean = torch.tensor(D.IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(D.IMAGENET_STD).view(3, 1, 1)
    fig, axes = plt.subplots(2, n // 2, figsize=(2.6 * n // 2, 5.6))
    for ax, i in zip(axes.ravel(), range(n)):
        x, y, name = ds[i]
        ax.imshow((x * std + mean).clamp(0, 1).permute(1, 2, 0).numpy())
        ax.set_title(f"{D.CLASS_NAMES[y]}", fontsize=9)
        ax.axis("off")
    fig.suptitle(f"Sau augmentation (aug={aug})")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=120)
    plt.show()
