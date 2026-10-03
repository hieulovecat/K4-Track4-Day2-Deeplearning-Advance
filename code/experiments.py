"""experiments.py - danh sách thí nghiệm, nghiên cứu suy luận, chung kết và tạo results.xlsx.

Notebook gọi các hàm ở đây để mọi cấu hình đều đi qua `train.run(Config(...))`.
Mã thí nghiệm (GUIDE mục 6.1): B01.. backbone · T00.. công thức huấn luyện · I00.. suy luận · F01.. chung kết.
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import benchmark as BM
import dataset as D
import inference as INF
import model as M
import train as TR
from train import Config

BACKBONES = [  # (exp_id, tên timm) - ResNet, ResNeXt, ConvNeXt, DeiT, Swin, 2 mạng nhẹ
    ("B01", "resnet50"),
    ("B02", "resnext50_32x4d"),
    ("B03", "convnext_tiny"),
    ("B04", "deit_small_patch16_224"),
    ("B05", "swin_tiny_patch4_window7_224"),
    ("B06", "efficientnet_b0"),
    ("B07", "mobilenetv3_large_100"),
]


def backbone_configs(**common) -> list[Config]:
    """Công thức nền T00 cho mọi backbone (chỉ đổi backbone). `common` = đường dẫn, epochs, seed..."""
    return [Config(exp_id=eid, backbone=name, note="công thức nền T00", **common) for eid, name in BACKBONES]


def training_ablations(backbone: str, **common) -> list[tuple[Config, str, str]]:
    """Mỗi cấu hình chỉ khác T00 đúng MỘT yếu tố. Trả về [(Config, trục, mô tả khác T00)].

    `common` là công thức nền hiện tại (có thể đã đưa yếu tố thắng vào nền nếu bạn dùng cách tham lam;
    khi đó hãy ghi rõ trong báo cáo). T00 phải được chạy với đúng `common` này.
    """
    spec = [  # (exp_id, trục, mô tả, override)
        ("T01", "A. khởi tạo", "đóng băng backbone, chỉ train head", dict(init="frozen")),
        ("T02", "A. khởi tạo", "huấn luyện từ đầu (không ImageNet)", dict(init="scratch")),
        ("T03", "B. augmentation", "thêm ColorJitter", dict(aug="color")),
        ("T04", "B. augmentation", "TrivialAugmentWide", dict(aug="trivial")),
        ("T05", "B. augmentation", "CutMix (alpha=1)", dict(mix="cutmix")),
        ("T06", "B. augmentation", "Mixup (alpha=1)", dict(mix="mixup")),
        ("T07", "C. loss", "label smoothing 0.1", dict(loss="ls", label_smoothing=0.1)),
        ("T08", "C. loss", "focal loss gamma=2", dict(loss="focal", focal_gamma=2.0)),
        ("T09", "C. loss", "CE trọng số lớp (class-balanced, beta=0.999)",
         dict(loss="ce_weighted", class_weight_beta=0.999)),
        ("T10", "D. sampler", "sampler cân bằng lớp", dict(sampler="balanced")),
        ("T11", "E. LR", "LR head = LR backbone (1e-4)", dict(lr_head=1e-4)),
        ("T12", "F. chính quy hoá", "EMA trọng số (0.999)", dict(ema_decay=0.999)),
        ("T13", "G. độ phân giải", "train và test ở 256", dict(img_size=256)),
    ]
    out = []
    for eid, axis, desc, ov in spec:
        kw = {**dict(backbone=backbone), **common, **ov, "exp_id": eid, "note": desc}
        out.append((Config(**kw), axis, desc))
    return out


def baseline_config(backbone: str, **common) -> Config:
    return Config(exp_id="T00", backbone=backbone, note="công thức nền", **common)


# --------------------------------------------------------------------------- #
# Thu thập kết quả các lần chạy
# --------------------------------------------------------------------------- #
def collect_runs(out_dir: str = "runs") -> pd.DataFrame:
    rows = []
    for f in sorted(Path(out_dir).glob("*/seed*/summary.json")):
        s = json.loads(f.read_text(encoding="utf-8"))
        s["val_f1_chinee"], s["val_f1_snake"] = s["val_f1"][0], s["val_f1"][7]
        s["val_recall_chinee"], s["val_recall_snake"] = s["val_recall"][0], s["val_recall"][7]
        rows.append({k: v for k, v in s.items() if k not in ("val_f1", "val_recall")})
    return pd.DataFrame(rows)


def mean_std_table(df: pd.DataFrame, cols=("val_macro_f1", "val_top1")) -> pd.DataFrame:
    g = df.groupby("exp_id")[list(cols)].agg(["mean", "std", "count"])
    g.columns = ["_".join(c) for c in g.columns]
    return g


def load_cfg(exp_id: str, seed: int, out_dir: str = "runs", **overrides) -> Config:
    """Dựng lại Config của một lần chạy đã xong từ config.json (đổi đường dẫn bằng **overrides)."""
    p = Path(out_dir) / exp_id / f"seed{seed}" / "config.json"
    d = json.loads(p.read_text(encoding="utf-8"))["cfg"]
    names = {f.name for f in dataclasses.fields(Config)}
    d = {k: v for k, v in d.items() if k in names}
    d.update(overrides, out_dir=out_dir)
    return Config(**d)


# --------------------------------------------------------------------------- #
# Độ trễ
# --------------------------------------------------------------------------- #
def backbone_latency(names: list[str], img_size: int = 224, device: str = "cuda") -> dict:
    """Độ trễ batch 1 (FP32) của từng backbone; trả về {tên: bản ghi latency_report}."""
    out = {}
    for n in names:
        m = M.build_model(n, pretrained=False, init="scratch")
        out[n] = BM.latency_report(m, 1, img_size, "fp32", device)
        print(f"{n:32s} p50 {out[n]['p50']:.2f}  p95 {out[n]['p95']:.2f}  p99 {out[n]['p99']:.2f} ms")
    return out


# --------------------------------------------------------------------------- #
# Nghiên cứu suy luận (Bước 3) - chỉ trên VAL
# --------------------------------------------------------------------------- #
def _center(x, size):
    h, w = x.shape[-2:]
    y0, x0 = (h - size) // 2, (w - size) // 2
    return x[..., y0:y0 + size, x0:x0 + size]


def _full_val_loader(cfg: Config, val_df, mean, std):
    """Val ở 256 (ảnh gốc, không crop); các view tự crop/resize từ ảnh này."""
    tf = D.build_transforms(False, 256, mean=mean, std=std)
    return D.make_loader(val_df, cfg.images_dir, tf, cfg.batch_size, train=False, num_workers=cfg.num_workers)


def _metrics(ev, probs, y):
    m = ev.compute_metrics(y, probs.argmax(1), probs)
    return m["macro_f1"], m["top1"], m["ece"]


VIEW_FNS = {
    "none": lambda x: [_center(x, 224)],
    "hflip": lambda x: [_center(x, 224), INF.view_hflip(_center(x, 224))],
    "crop5": lambda x: INF.views_multicrop(x, 224),
    "crop10": lambda x: INF.views_multicrop(x, 224, flip=True),
}


def inference_study(cfg: Config, device=None, with_latency: bool = True, lat_iters: int = 100,
                    ensemble_with: list[Config] | None = None, resolutions=(224, 256, 288, 320)) -> pd.DataFrame:
    """I00..I08 trên val cho một lần chạy (cfg, đã train xong). Trả về DataFrame cho sheet `Inference`."""
    ev = TR._eval_mod()
    device = device or TR.get_device()
    dev = device.type
    _, val_df, _ = D.load_split(cfg.labels_dir, cfg.fold)
    model = TR.load_best_model(cfg, device)
    mean, std = TR.model_norm(TR.timm_pretrained_stub(cfg))
    loader = _full_val_loader(cfg, val_df, mean, std)
    rows = []

    def lat(img, k=1, fused=None, dtype="fp32"):
        if not with_latency:
            return {}
        m = fused if fused is not None else model
        kw = dict(batch_size=1, img_size=img, dtype=dtype, device=dev, iters=lat_iters)
        r = BM.tta_latency(m, k, **kw) if k > 1 else BM.latency_report(m, **kw, fused_bn=fused is not None)
        return {"p50_ms": r["p50"], "p95_ms": r["p95"], "p99_ms": r["p99"], "images_per_s": r["images_per_s"]}

    def add(eid, method, ckpt, k, probs, y, latency, note=""):
        f1, acc, ece = _metrics(ev, probs, y)
        rows.append({"exp_id": eid, "method": method, "model": ckpt, "K": k, "val_macro_f1": f1,
                     "val_top1": acc, "val_ece": ece, **latency, "note": note})

    ck = f"{cfg.exp_id}/seed{cfg.seed}"
    # I00 - 1 view (mốc)
    _, y, [lg0] = INF.predict_views(model, loader, device, VIEW_FNS["none"])
    add("I00", "1 view (center crop 224)", ck, 1, INF.aggregate_views([lg0]), y, lat(224))
    # I01 / I03 - TTA lật; gộp xác suất vs logit
    _, _, lgs = INF.predict_views(model, loader, device, VIEW_FNS["hflip"])
    add("I01", "TTA lật ngang, gộp xác suất", ck, 2, INF.aggregate_views(lgs, "prob"), y, lat(224, 2))
    add("I03", "TTA lật ngang, gộp logit", ck, 2, INF.aggregate_views(lgs, "logit"), y, lat(224, 2))
    # I02 - multi-crop
    _, _, lgs5 = INF.predict_views(model, loader, device, VIEW_FNS["crop5"])
    add("I02a", "5-crop 224", ck, 5, INF.aggregate_views(lgs5, "prob"), y, lat(224, 5))
    _, _, lgs10 = INF.predict_views(model, loader, device, VIEW_FNS["crop10"])
    add("I02b", "5-crop + lật (10 view)", ck, 10, INF.aggregate_views(lgs10, "prob"), y, lat(224, 10))
    # I04 - độ phân giải kiểm tra (CNN; ViT/Swin sẽ báo lỗi kích thước)
    for r in resolutions:
        if r == 224:
            continue
        fn = (lambda s: (lambda x: INF.views_multiscale(x, [s])))(r)
        try:
            _, _, [lg] = INF.predict_views(model, loader, device, fn)
            add(f"I04_{r}", f"độ phân giải test {r}", ck, 1, INF.aggregate_views([lg]), y, lat(r),
                note="phóng to từ ảnh gốc 256" if r > 256 else "")
        except Exception as e:  # noqa: BLE001
            print(f"  bỏ qua độ phân giải {r}: {type(e).__name__}: {str(e)[:80]}")
    # I05 - ensemble các run khác (trung bình xác suất của logit đã lưu 1-view)
    if ensemble_with:
        probs = [INF.aggregate_views([np.load(TR.run_dir(c) / "val_logits.npy")]) for c in [cfg, *ensemble_with]]
        k = len(probs)
        l1 = lat(224)
        l_ens = {kk: v * k if kk != "images_per_s" else v / k for kk, v in l1.items()}
        add("I05", f"ensemble {k} mô hình", ", ".join(f"{c.exp_id}/seed{c.seed}" for c in [cfg, *ensemble_with]),
            k, INF.ensemble_probs(probs), y, l_ens, note="độ trễ = K × một mô hình (ước lượng)")
    # I07 - temperature scaling (T khớp trên VAL, ECE sau là trong mẫu)
    T = INF.fit_temperature(lg0, y)
    add("I07", f"temperature scaling T={T:.3f}", ck, 1, INF.apply_temperature(lg0, T), y, lat(224),
        note="T khớp trên val; ECE val sau TS là trong mẫu (kiểm chứng thật trên test ở Bước 4)")
    # I08 - gộp BN và FP16/AMP
    fused = INF.fuse_conv_bn(model)
    _, _, [lgf] = INF.predict_views(fused, loader, device, VIEW_FNS["none"])
    add("I08a", "gộp BN vào Conv (FP32)", ck, 1, INF.aggregate_views([lgf]), y, lat(224, fused=fused))
    if dev == "cuda":
        _, _, [lg16] = INF.predict_views(model, loader, device, VIEW_FNS["none"], amp=True)
        add("I08b", "AMP (autocast fp16)", ck, 1, INF.aggregate_views([lg16]), y, lat(224, dtype="amp"))
        add("I08c", "FP16 (model.half())", ck, 1, INF.aggregate_views([lg0]), y, lat(224, dtype="fp16"),
            note="độ chính xác lấy theo FP32 (chưa chạy lại bằng half); độ trễ đo bằng half")
    df = pd.DataFrame(rows)
    if with_latency and "p50_ms" in df:
        base = df.loc[df.exp_id == "I00", "p50_ms"].iloc[0]
        df["cost_vs_I00"] = df["p50_ms"] / base
    return df


# --------------------------------------------------------------------------- #
# Chung kết (Bước 4)
# --------------------------------------------------------------------------- #
def _views_logits(model, loader, device, views: str):
    names, y, lgs = INF.predict_views(model, loader, device, VIEW_FNS[views])
    return names, y, lgs


def predict_final(cfg: Config, views: str = "hflip", agg: str = "logit", temperature: bool = True,
                  device=None) -> dict:
    """Suy luận cuối cho MỘT seed của cấu hình chung kết và ghi:
        <pred_dir>/<exp_id>_seed<k>_test.csv         (đã áp dụng phương pháp, có temperature scaling nếu bật)
        <pred_dir>/<exp_id>_uncal_seed<k>_test.csv   (cùng views, chưa temperature scaling; chấm I4a)
        <pred_dir>/<exp_id>_seed<k>_val.csv          (val của chung kết; chấm I4b)
    T khớp trên VAL của chính seed đó. Test chạy đúng một lần: file test đã có thì KHÔNG ghi đè.
    """
    ev = TR._eval_mod()
    device = device or TR.get_device()
    _, val_df, test_df = D.load_split(cfg.labels_dir, cfg.fold)
    test_path = TR.pred_path(cfg, "test")
    if test_path.exists():
        print(f"{test_path} đã tồn tại -> bỏ qua (test chỉ chạy một lần mỗi seed)")
        return {"skipped": True}
    model = TR.load_best_model(cfg, device)
    mean, std = TR.model_norm(TR.timm_pretrained_stub(cfg))
    loader = lambda df: _full_val_loader(cfg, df, mean, std)  # noqa: E731
    nv, yv, lgv = _views_logits(model, loader(val_df), device, views)
    T = INF.fit_temperature(np.mean(lgv, axis=0), yv) if temperature else 1.0

    def probs(lgs, t):
        return INF.aggregate_views([l / t for l in lgs], agg)

    nt, yt, lgt = _views_logits(model, loader(test_df), device, views)
    ev.save_predictions(TR.pred_path(cfg, "val"), nv, yv, probs(lgv, T))
    ev.save_predictions(test_path, nt, yt, probs(lgt, T))
    uncal = Path(cfg.pred_dir) / f"{cfg.exp_id}_uncal_seed{cfg.seed}_test.csv"
    ev.save_predictions(uncal, nt, yt, probs(lgt, 1.0))
    print(f"[{cfg.exp_id} seed{cfg.seed}] views={views} agg={agg} T={T:.3f} -> {test_path.name}")
    return {"T": T, "views": views, "agg": agg}


def run_final(exp_id: str, seeds=(0, 1, 2), views="hflip", agg="logit", temperature=True, **cfg_kw) -> list:
    """Huấn luyện cấu hình chung kết với nhiều seed rồi ghi predictions (test một lần mỗi seed)."""
    out = []
    for s in seeds:
        cfg = Config(exp_id=exp_id, seed=s, save_test_predictions=False, **cfg_kw)
        TR.run(cfg)
        out.append(predict_final(cfg, views, agg, temperature))
    return out


def run_baseline(seeds=(0, 1, 2), **cfg_kw) -> None:
    """Mốc T00 + I00 (1 view, không TS): test ghi trực tiếp bởi train.run."""
    for s in seeds:
        TR.run(Config(exp_id="T00", seed=s, save_test_predictions=True, **cfg_kw))


# --------------------------------------------------------------------------- #
# results.xlsx
# --------------------------------------------------------------------------- #
def _group_rows(ev, exp_id, pattern, labels, test_csv=None, what="test"):
    names = ev.load_names(labels)
    g = ev.load_group(pattern, test_csv, ref_what=what)
    rows = [{"exp_id": exp_id, "seed": p.seed, "top1": m["top1"], "macro_f1": m["macro_f1"],
             "balanced_acc": m["balanced_acc"], "ece": m["ece"]} for p, m in zip(g.preds, g.metrics)]
    summ = {"exp_id": exp_id, "seed": "mean ± std"}
    for k in ("top1", "macro_f1", "balanced_acc", "ece"):
        summ[k] = ev.fmt(*g.summary[k])
    pc = []
    for i, n in enumerate(names):
        pc.append({"exp_id": exp_id, "class": n, "n_test": int(g.metrics[0]["support"][i]),
                   "precision": g.summary["precision"][0][i], "recall": g.summary["recall"][0][i],
                   "f1": g.summary["f1"][0][i]})
    return rows, summ, pc


def build_results_xlsx(path: str, runs: pd.DataFrame, inference_df: pd.DataFrame | None,
                       latency_df: pd.DataFrame | None, finals: dict, labels_csv: str, test_csv: str,
                       backbone_lat: dict | None = None, pred_dir: str = "predictions") -> Path:
    """Ghi results.xlsx với các sheet Backbones, Training, Inference, Final, PerClass, Latency, Summary.

    finals: {"F01": "mô tả cấu hình", "T00": "mốc (công thức nền + 1 view)"}; đọc predictions/<id>_seed*_test.csv.
    backbone_lat: {tên backbone: bản ghi latency_report} (từ experiments.backbone_latency).
    """
    ev = TR._eval_mod()
    runs = runs.copy()
    sheets = {}
    # Backbones
    b = runs[runs.exp_id.str.startswith("B")].copy()
    if backbone_lat:
        b["latency_b1_p50_ms"] = b.backbone.map(lambda n: backbone_lat.get(n, {}).get("p50"))
    sheets["Backbones"] = b.rename(columns={"params_M": "params_M", "gmacs": "GMAC"})
    # Training (Δ so với T00 cùng seed)
    t = runs[runs.exp_id.str.startswith("T")].copy()
    base = t[t.exp_id == "T00"].set_index("seed")["val_macro_f1"]
    t["delta_vs_T00"] = t.apply(lambda r: r.val_macro_f1 - base.get(r.seed, np.nan), axis=1)
    sheets["Training"] = t
    if inference_df is not None:
        sheets["Inference"] = inference_df
    final_rows, per_class, summ_rows = [], [], []
    for eid, desc in finals.items():
        pat = f"{pred_dir}/{eid}_seed*_test.csv"
        rows, summ, pc = _group_rows(ev, eid, pat, labels_csv, test_csv)
        vpat = f"{pred_dir}/{eid}_seed*_val.csv"
        try:
            vg = ev.load_group(vpat, None, ref_what="val")
            vmap = {p.seed: m["macro_f1"] for p, m in zip(vg.preds, vg.metrics)}
        except FileNotFoundError:
            vmap = {}
        for r in rows:
            r["config"], r["macro_f1_val"] = desc, vmap.get(r["seed"])
        final_rows += rows + [{**summ, "config": desc}]
        per_class += pc
    sheets["Final"] = pd.DataFrame(final_rows)
    sheets["PerClass"] = pd.DataFrame(per_class)
    if latency_df is not None:
        sheets["Latency"] = latency_df
    # Summary: top 10 theo macro-F1 val trong B, T, I
    cand = []
    for _, r in pd.concat([b, t]).iterrows():
        cand.append({"exp_id": r.exp_id, "config": r.get("note") or r.backbone, "val_macro_f1": r.val_macro_f1,
                     "val_top1": r.val_top1, "cost": f"{r.train_time_per_epoch_s:.0f} s/epoch (train)"})
    if inference_df is not None:
        for _, r in inference_df.iterrows():
            cand.append({"exp_id": r.exp_id, "config": r.method, "val_macro_f1": r.val_macro_f1,
                         "val_top1": r.val_top1, "cost": f"{r.get('cost_vs_I00', float('nan')):.2f}× I00"})
    sheets["Summary"] = pd.DataFrame(cand).sort_values("val_macro_f1", ascending=False).head(10)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        for name, df in sheets.items():
            df.to_excel(xw, sheet_name=name, index=False, freeze_panes=(1, 0), float_format="%.4f")
            ws = xw.sheets[name]
            for col in ws.columns:
                ws.column_dimensions[col[0].column_letter].width = min(
                    45, max(10, max(len(str(c.value)) if c.value is not None else 0 for c in col) + 2))
    return path
