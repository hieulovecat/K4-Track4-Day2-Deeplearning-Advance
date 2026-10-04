"""losses.py - các hàm loss và trộn mẫu (Mixup, CutMix).

Liên hệ slide Day 2: label smoothing (trang 56), focal loss (trang 57), Mixup/CutMix (trang 48).

Giao diện:
    build_criterion(kind, **kw)                 -> callable(logits, target) -> loss scalar
    class_weights(counts, beta)                 -> tensor trọng số lớp
    mix_batch(x, y, alpha, mode)                -> (x_mixed, (y_a, y_b, lam))
    mixed_loss(criterion, logits, targets)      -> loss scalar
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingCE(nn.Module):
    """q'(k) = (1 - eps) * 1[k == y] + eps / K.  Tự cài đặt: loss = (1-eps)*NLL + eps*mean_k(-log p_k).

    eps = 0 cho đúng cross-entropy (có test).
    """

    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, logits, target):
        logp = F.log_softmax(logits.float(), dim=-1)
        nll = -logp.gather(1, target[:, None]).squeeze(1)
        smooth = -logp.mean(dim=-1)
        return ((1 - self.smoothing) * nll + self.smoothing * smooth).mean()


class FocalLoss(nn.Module):
    """FL = -alpha_t * (1 - p_t)^gamma * log(p_t), trung bình theo batch. gamma = 0, alpha=None ≡ CE."""

    def __init__(self, gamma: float = 2.0, alpha=None):
        super().__init__()
        self.gamma = gamma
        if alpha is not None:
            alpha = torch.as_tensor(alpha, dtype=torch.float32)
        self.register_buffer("alpha", alpha)

    def forward(self, logits, target):
        logp = F.log_softmax(logits.float(), dim=-1).gather(1, target[:, None]).squeeze(1)
        pt = logp.exp()
        loss = -((1 - pt) ** self.gamma) * logp
        if self.alpha is not None:
            loss = loss * self.alpha[target]
        return loss.mean()


def build_criterion(kind: str = "ce", **kw):
    """kind: "ce" | "ls" (smoothing=) | "focal" (gamma=, alpha=) | "ce_weighted" (weight=)."""
    if kind == "ce":
        return nn.CrossEntropyLoss()
    if kind == "ls":
        return LabelSmoothingCE(kw.get("smoothing", 0.1))
    if kind == "focal":
        return FocalLoss(kw.get("gamma", 2.0), kw.get("alpha"))
    if kind == "ce_weighted":
        w = kw.get("weight")
        if w is None:
            raise ValueError("ce_weighted cần weight=")
        return nn.CrossEntropyLoss(weight=torch.as_tensor(w, dtype=torch.float32))
    raise ValueError(f"loss không hợp lệ: {kind!r} (ce|ls|focal|ce_weighted)")


def class_weights(counts, beta: float = 0.0):
    """Trọng số theo lớp từ số ảnh mỗi lớp của TRAIN.

    beta = 0: w_c = 1/n_c, chuẩn hoá về trung bình 1.
    beta > 0: w_c = (1-beta)/(1-beta^n_c) (Cui et al.), chuẩn hoá tổng = số lớp.
    """
    n = np.asarray(counts, dtype=np.float64)
    if beta == 0:
        w = 1.0 / n
    else:
        w = (1.0 - beta) / (1.0 - np.power(beta, n))
    w = w / w.sum() * len(n)
    return torch.as_tensor(w, dtype=torch.float32)


def mix_batch(x, y, alpha: float = 1.0, mode: str = "cutmix"):
    """Mixup / CutMix một batch. Trả về (x_mix, (y_a, y_b, lam)); lam của CutMix tính theo diện tích thực."""
    lam = float(np.random.beta(alpha, alpha))
    perm = torch.randperm(x.size(0), device=x.device)
    y_a, y_b = y, y[perm]
    if mode == "mixup":
        return lam * x + (1 - lam) * x[perm], (y_a, y_b, lam)
    if mode != "cutmix":
        raise ValueError(f"mode không hợp lệ: {mode!r} (mixup|cutmix)")
    _, _, h, w = x.shape
    cut = np.sqrt(1.0 - lam)
    ch, cw = int(h * cut), int(w * cut)
    cy, cx = np.random.randint(h), np.random.randint(w)
    y1, y2 = np.clip(cy - ch // 2, 0, h), np.clip(cy + ch // 2, 0, h)
    x1, x2 = np.clip(cx - cw // 2, 0, w), np.clip(cx + cw // 2, 0, w)
    x_mix = x.clone()
    x_mix[:, :, y1:y2, x1:x2] = x[perm][:, :, y1:y2, x1:x2]
    lam = 1.0 - float((y2 - y1) * (x2 - x1)) / (h * w)
    return x_mix, (y_a, y_b, lam)


def mixed_loss(criterion, logits, targets):
    """lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)."""
    y_a, y_b, lam = targets
    return lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)
