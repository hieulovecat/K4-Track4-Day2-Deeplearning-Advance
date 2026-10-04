"""benchmark.py - đo độ trễ suy luận đúng cách (slide Day 2, trang 73 và 75; GUIDE.md mục 4.1).

Quy tắc đo: warmup >= 10 lần; torch.cuda.synchronize() TRƯỚC và SAU đoạn cần đo; >= 50 lần đo;
báo cáo p50/p95/p99. Độ trễ ở đây là thời gian FORWARD của model (không gồm đọc ảnh và tiền xử lý).
"""
from __future__ import annotations

import copy
import time

import numpy as np
import torch


def bench(fn, warmup: int = 10, iters: int = 100, sync=None) -> dict:
    """Đo thời gian một hàm `fn()` (mili-giây). `sync` là hàm đồng bộ (torch.cuda.synchronize) hoặc None."""
    if iters < 50:
        raise ValueError("cần >= 50 lần đo (GUIDE mục 4.1)")
    for _ in range(warmup):
        fn()
    if sync:
        sync()
    times = []
    for _ in range(iters):
        if sync:
            sync()
        t0 = time.perf_counter()
        fn()
        if sync:
            sync()
        times.append((time.perf_counter() - t0) * 1000.0)
    t = np.asarray(times)
    return {"p50": float(np.percentile(t, 50)), "p95": float(np.percentile(t, 95)),
            "p99": float(np.percentile(t, 99)), "mean": float(t.mean()), "n": iters}


def _prepare(model, dtype: str, device: str):
    m = copy.deepcopy(model).to(device).eval()
    if dtype == "fp16":
        m = m.half()
    return m


def _forward_fn(m, x, dtype: str, device: str, n_forward: int = 1):
    amp = dtype == "amp" and device == "cuda"

    def fn():
        with torch.inference_mode(), torch.autocast(device, dtype=torch.float16, enabled=amp):
            for _ in range(n_forward):
                m(x)
    return fn


def latency_report(model, batch_size: int, img_size: int, dtype: str = "fp32", device: str = "cuda",
                   warmup: int = 10, iters: int = 100, fused_bn: bool = False) -> dict:
    """Độ trễ forward với đầu vào ngẫu nhiên (batch_size, 3, img_size, img_size).

    dtype: "fp32" | "amp" (autocast fp16) | "fp16" (model.half()). Dùng deepcopy nên model gốc không đổi.
    `fused_bn` chỉ để GHI vào kết quả (hãy truyền model đã gọi inference.fuse_conv_bn).
    """
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    m = _prepare(model, dtype, device)
    x = torch.randn(batch_size, 3, img_size, img_size, device=device,
                    dtype=torch.float16 if dtype == "fp16" else torch.float32)
    sync = torch.cuda.synchronize if device == "cuda" else None
    r = bench(_forward_fn(m, x, dtype, device), warmup, iters, sync)
    return {"gpu": torch.cuda.get_device_name(0) if device == "cuda" else "cpu", "dtype": dtype,
            "batch": batch_size, "img_size": img_size, "fused_bn": fused_bn,
            "p50": r["p50"], "p95": r["p95"], "p99": r["p99"],
            "images_per_s": batch_size / (r["p50"] / 1000.0), "torch": torch.__version__}


def tta_latency(model, k_views: int, **kw) -> dict:
    """Độ trễ của TTA K view (K forward liên tiếp), so với K * p50 của một lượt chạy."""
    batch_size, img_size = kw.get("batch_size", 1), kw["img_size"]
    dtype, device = kw.get("dtype", "fp32"), kw.get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    single = latency_report(model, **{**kw, "device": device})
    m = _prepare(model, dtype, device)
    x = torch.randn(batch_size, 3, img_size, img_size, device=device,
                    dtype=torch.float16 if dtype == "fp16" else torch.float32)
    sync = torch.cuda.synchronize if device == "cuda" else None
    r = bench(_forward_fn(m, x, dtype, device, k_views), kw.get("warmup", 10), kw.get("iters", 100), sync)
    return {**single, "k_views": k_views, "p50": r["p50"], "p95": r["p95"], "p99": r["p99"],
            "images_per_s": batch_size / (r["p50"] / 1000.0),
            "single_p50": single["p50"], "k_times_single_p50": k_views * single["p50"]}
