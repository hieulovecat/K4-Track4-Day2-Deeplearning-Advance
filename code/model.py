"""model.py - tạo backbone, đóng băng, nhóm tham số, đếm params/GMAC.

Giao diện:
    build_model(name, pretrained, num_classes, drop_rate, init) -> nn.Module
    freeze_backbone(model)                                        -> None
    set_train_mode(model)                                         -> None (giữ BN backbone ở eval nếu đóng băng)
    param_groups(model, lr_backbone, lr_head, weight_decay)       -> list[dict] cho optimizer
    count_params(model) -> float (triệu)     count_gmacs(model, img_size) -> float
"""
from __future__ import annotations

import timm
import torch
import torch.nn as nn

SUGGESTED_BACKBONES = {
    "resnet50": "resnet50",
    "resnext50": "resnext50_32x4d",
    "convnext_tiny": "convnext_tiny",
    "deit_small": "deit_small_patch16_224",
    "swin_tiny": "swin_tiny_patch4_window7_224",
    "efficientnet_b0": "efficientnet_b0",        # mạng nhẹ
    "mobilenetv3": "mobilenetv3_large_100",      # mạng nhẹ
}


def pretrained_tag(model) -> str:
    """Tag trọng số timm thực sự được dùng (ghi vào results.xlsx)."""
    cfg = getattr(model, "pretrained_cfg", None) or {}
    return cfg.get("tag") or cfg.get("hf_hub_id") or cfg.get("url") or "none"


def build_model(name: str, pretrained: bool = True, num_classes: int = 9,
                drop_rate: float = 0.0, init: str = "finetune"):
    """Tạo model phân loại 9 lớp. init: "scratch" | "frozen" | "finetune" (trục A)."""
    if init not in ("scratch", "frozen", "finetune"):
        raise ValueError(f"init không hợp lệ: {init!r}")
    use_pre = pretrained and init != "scratch"
    model = timm.create_model(name, pretrained=use_pre, num_classes=num_classes, drop_rate=drop_rate)
    model.weights_tag = pretrained_tag(model) if use_pre else "scratch"
    model._frozen_backbone = False
    if init == "frozen":
        freeze_backbone(model)
    return model


def _head_params(model) -> list[nn.Parameter]:
    return list(model.get_classifier().parameters())


def freeze_backbone(model) -> None:
    """requires_grad=False cho mọi tham số trừ head. Dùng set_train_mode() để giữ BN backbone ở eval."""
    head_ids = {id(p) for p in _head_params(model)}
    for p in model.parameters():
        p.requires_grad = id(p) in head_ids
    model._frozen_backbone = True


def set_train_mode(model) -> None:
    """model.train(), nhưng nếu backbone đóng băng thì mọi lớp ngoài head về eval.

    Nếu để BatchNorm ở train mode, thống kê chạy (running mean/var) của backbone vẫn bị cập nhật
    dù trọng số đã đóng băng, làm "backbone đóng băng" thật ra thay đổi theo dữ liệu mới.
    """
    model.train()
    if getattr(model, "_frozen_backbone", False):
        head_mods = set(model.get_classifier().modules())
        for m in model.modules():
            if m not in head_mods:
                m.eval()
        for m in head_mods:
            m.train()


def param_groups(model, lr_backbone: float, lr_head: float, weight_decay: float):
    """3 nhóm tham số (slide Day 2, trang 52): backbone (có wd), norm/bias backbone (wd=0), head."""
    head_ids = {id(p) for p in _head_params(model)}
    decay, no_decay, head = [], [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        if id(p) in head_ids:
            head.append(p)
        elif p.ndim <= 1:
            no_decay.append(p)
        else:
            decay.append(p)
    groups = [
        {"params": decay, "lr": lr_backbone, "weight_decay": weight_decay, "name": "backbone"},
        {"params": no_decay, "lr": lr_backbone, "weight_decay": 0.0, "name": "backbone_norm_bias"},
        {"params": head, "lr": lr_head, "weight_decay": weight_decay, "name": "head"},
    ]
    return [g for g in groups if g["params"]]


def count_params(model) -> float:
    """Số tham số (triệu), gồm cả tham số đóng băng."""
    return sum(p.numel() for p in model.parameters()) / 1e6


def count_gmacs(model, img_size: int = 224) -> float:
    """GMAC cho một ảnh 3 x img_size x img_size, đếm bằng torch.utils.flop_counter.FlopCounterMode.

    Công cụ đếm FLOPs (conv, matmul, attention) rồi chia 2 để ra MAC; bỏ qua các phép phi tuyến và
    chuẩn hoá nên lệch vài phần trăm so với fvcore/ptflops.
    """
    from torch.utils.flop_counter import FlopCounterMode

    was_training = model.training
    model.eval()
    p = next(model.parameters())
    x = torch.zeros(1, 3, img_size, img_size, device=p.device, dtype=p.dtype)
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        model(x)
    model.train(was_training)
    return fc.get_total_flops() / 2 / 1e9
