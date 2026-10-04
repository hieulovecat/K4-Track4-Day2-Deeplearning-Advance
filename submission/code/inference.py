"""inference.py - các phương pháp suy luận (Bước 3 của GUIDE.md).

Liên hệ slide Day 2: TTA (trang 62-66, 75), ensemble/EMA/soup (trang 67), độ phân giải kiểm tra
(trang 68), temperature scaling (trang 69), gộp BatchNorm (trang 71).

Mọi hàm chạy ở chế độ eval, không gradient. Chọn phương pháp CHỈ dựa trên val;
nhiệt độ T khớp trên VAL rồi áp dụng sang test (README.md, S2 và S4).

Giao diện:
    predict_logits(model, loader, device, view=None)      -> (filenames, y_true, logits[N, 9])
    predict_views(model, loader, device, view_fn)         -> (filenames, y_true, [logits_k[N, 9]])
    aggregate_views(list_of_logits, space)                -> probs[N, 9]
    fit_temperature(val_logits, val_labels)               -> float T
    apply_temperature(logits, T)                          -> probs
    ensemble_probs(list_of_probs)                         -> probs
    fuse_conv_bn(model)                                   -> model (BN đã gộp vào conv)
"""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


@torch.inference_mode()
def predict_views(model, loader, device, view_fn=None, amp: bool = False):
    """Một lượt qua loader, mỗi batch biến thành K view; trả về (names, y_true, [logits_k])."""
    model.eval()
    names, ys, per_view = [], [], None
    for x, y, fn in loader:
        x = x.to(device, non_blocking=True)
        views = view_fn(x) if view_fn is not None else [x]
        if per_view is None:
            per_view = [[] for _ in views]
        for k, v in enumerate(views):
            with torch.autocast(device.type, dtype=torch.float16, enabled=amp and device.type == "cuda"):
                out = model(v)
            per_view[k].append(out.float().cpu().numpy())
        names += list(fn)
        ys.append(np.asarray(y))
    return names, np.concatenate(ys), [np.concatenate(v) for v in per_view]


def predict_logits(model, loader, device, view=None, amp: bool = False):
    """Chạy model trên loader, gom logit theo đúng thứ tự file. `view` biến đổi batch (hoặc None)."""
    fn = (lambda x: [view(x)]) if view is not None else None
    names, y, lg = predict_views(model, loader, device, fn, amp)
    return names, y, lg[0]


def view_identity(x):
    return x


def view_hflip(x):
    """Lật ngang batch (N, C, H, W) (slide trang 75)."""
    return torch.flip(x, dims=[-1])


def views_multicrop(x, crop: int, flip: bool = False):
    """5 crop (4 góc + giữa) kích thước `crop`; flip=True thêm bản lật của từng crop (10 view)."""
    h, w = x.shape[-2:]
    if crop > min(h, w):
        raise ValueError(f"crop {crop} lớn hơn ảnh {h}x{w}")
    ys = [0, 0, h - crop, h - crop, (h - crop) // 2]
    xs = [0, w - crop, 0, w - crop, (w - crop) // 2]
    views = [x[..., y:y + crop, c:c + crop] for y, c in zip(ys, xs)]
    if flip:
        views += [torch.flip(v, dims=[-1]) for v in views]
    return views


def views_multiscale(x, sizes):
    """Resize batch về từng kích thước trong `sizes`. CNN có global pooling chạy được mọi kích thước;
    ViT/Swin của timm yêu cầu đúng kích thước lúc train nên sẽ báo lỗi (ghi giới hạn này vào báo cáo)."""
    out = []
    for s in sizes:
        out.append(x if x.shape[-1] == s else
                   F.interpolate(x, size=(s, s), mode="bilinear", align_corners=False, antialias=s < x.shape[-1]))
    return out


def aggregate_views(logits_per_view, space: str = "prob"):
    """Gộp K view: space="prob" (trung bình softmax) hoặc "logit" (trung bình logit rồi softmax)."""
    if space == "prob":
        return np.mean([_softmax(l) for l in logits_per_view], axis=0)
    if space == "logit":
        return _softmax(np.mean(logits_per_view, axis=0))
    raise ValueError(f"space không hợp lệ: {space!r} (prob|logit)")


def ensemble_probs(list_of_probs):
    """Trung bình xác suất của nhiều mô hình trên CÙNG tập ảnh, cùng thứ tự file."""
    shapes = {p.shape for p in list_of_probs}
    if len(shapes) != 1:
        raise ValueError(f"các mô hình có kích thước dự đoán khác nhau: {shapes}")
    return np.mean(list_of_probs, axis=0)


def _nll(logits: np.ndarray, labels: np.ndarray, T: float) -> float:
    z = logits / T
    z = z - z.max(1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(1, keepdims=True))
    return float(-logp[np.arange(len(labels)), labels].mean())


def fit_temperature(val_logits, val_labels) -> float:
    """T > 0 cực tiểu NLL trên VAL (p = softmax(logit / T)). Tìm kiếm golden-section trên log T:
    NLL là hàm lồi theo 1/T nên chỉ có một cực tiểu. Không khớp trên test."""
    lg = np.asarray(val_logits, dtype=np.float64)
    y = np.asarray(val_labels)
    lo, hi = np.log(0.05), np.log(20.0)
    phi = (np.sqrt(5) - 1) / 2
    a, b = lo, hi
    c, d = b - phi * (b - a), a + phi * (b - a)
    fc, fd = _nll(lg, y, np.exp(c)), _nll(lg, y, np.exp(d))
    for _ in range(60):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - phi * (b - a)
            fc = _nll(lg, y, np.exp(c))
        else:
            a, c, fc = c, d, fd
            d = a + phi * (b - a)
            fd = _nll(lg, y, np.exp(d))
    return float(np.exp((a + b) / 2))


def apply_temperature(logits, T: float):
    """softmax(logits / T)."""
    return _softmax(np.asarray(logits, dtype=np.float64) / T)


# --------------------------------------------------------------------------- #
# Gộp BatchNorm vào Conv
# --------------------------------------------------------------------------- #
def _fuse_pair(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> nn.Conv2d:
    """w' = gamma * w / sqrt(var + eps);  b' = beta + gamma * (b - mean) / sqrt(var + eps)."""
    gamma = bn.weight if bn.weight is not None else torch.ones_like(bn.running_mean)
    beta = bn.bias if bn.bias is not None else torch.zeros_like(bn.running_mean)
    scale = gamma / torch.sqrt(bn.running_var + bn.eps)
    fused = nn.Conv2d(conv.in_channels, conv.out_channels, conv.kernel_size, conv.stride,
                      conv.padding, conv.dilation, conv.groups, bias=True,
                      padding_mode=conv.padding_mode).to(conv.weight.device)
    bias = conv.bias if conv.bias is not None else torch.zeros_like(bn.running_mean)
    with torch.no_grad():
        fused.weight.copy_(conv.weight * scale.reshape(-1, 1, 1, 1))
        fused.bias.copy_(beta + (bias - bn.running_mean) * scale)
    return fused


def _fuse_recursive(module: nn.Module) -> int:
    n_fused = 0
    names = list(module._modules.keys())
    i = 0
    while i < len(names) - 1:
        a, b = module._modules[names[i]], module._modules[names[i + 1]]
        if type(a) is nn.Conv2d and isinstance(b, nn.BatchNorm2d) and b.running_mean is not None \
                and b.num_features == a.out_channels:
            module._modules[names[i]] = _fuse_pair(a, b)
            # timm BatchNormAct2d còn chứa kích hoạt: giữ lại phần act, bỏ phần chuẩn hoá
            module._modules[names[i + 1]] = getattr(b, "act", None) or nn.Identity()
            n_fused += 1
            i += 2
        else:
            i += 1
    for child in module.children():
        n_fused += _fuse_recursive(child)
    return n_fused


def fuse_conv_bn(model, check_size: int = 224, verbose: bool = True):
    """Trả về BẢN SAO model đã gộp mọi cặp (Conv2d, BatchNorm2d) liền kề; model gốc không đổi.

    Kiến trúc không có BN (ViT, Swin; ConvNeXt dùng LayerNorm) trả về bản sao nguyên vẹn: không áp dụng.
    In sai số lớn nhất giữa đầu ra trước/sau gộp và raise nếu > 1e-3.
    """
    ref = copy.deepcopy(model).eval()
    fused = copy.deepcopy(model).eval()
    n = _fuse_recursive(fused)
    if n == 0:
        if verbose:
            print("fuse_conv_bn: không có cặp Conv-BN nào (kiến trúc không dùng BN): không áp dụng")
        return fused
    p = next(fused.parameters())
    x = torch.randn(2, 3, check_size, check_size, device=p.device, dtype=p.dtype)
    with torch.no_grad():
        err = (ref(x) - fused(x)).abs().max().item()
    if verbose:
        print(f"fuse_conv_bn: gộp {n} cặp Conv-BN, sai số đầu ra lớn nhất = {err:.2e}")
    if err > 1e-3:
        raise RuntimeError(f"gộp BN sai (lệch {err:.2e}); kiểm tra cấu trúc mạng")
    return fused
