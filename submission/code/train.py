"""train.py - vòng huấn luyện cho mọi thí nghiệm (B, T, F).

Một hàm `run(cfg)` dùng chung cho mọi cấu hình: đổi thí nghiệm chỉ bằng cách đổi `Config`.

Chạy một thí nghiệm từ dòng lệnh:
    python train.py --set exp_id=B01 backbone=resnet50 seed=0
Chỉ số chọn checkpoint (macro-F1 val) tính bằng eval.compute_metrics của repo gốc, để cùng
định nghĩa với lúc chấm.
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

import dataset as D
import losses as L
import model as M


for _stream in (sys.stdout, sys.stderr):      # console Windows cp1252 không in được tiếng Việt
    try:
        _stream.reconfigure(errors="replace")
    except Exception:  # noqa: BLE001
        pass


def _eval_mod():
    """Import eval.py của repo gốc (nằm ở thư mục cha của code/, hoặc đặt EVAL_DIR)."""
    here = Path(__file__).resolve().parent
    for p in [os.environ.get("EVAL_DIR"), here.parent, here, Path.cwd()]:
        if p and (Path(p) / "eval.py").exists():
            if str(p) not in sys.path:
                sys.path.insert(0, str(p))
            break
    import eval as ev  # noqa: WPS433
    return ev


@dataclass
class Config:
    # --- định danh ---
    exp_id: str = "T00"
    seed: int = 0
    fold: int = 0
    # --- mô hình ---
    backbone: str = "resnet50"
    init: str = "finetune"            # scratch | frozen | finetune
    drop_rate: float = 0.0
    # --- dữ liệu / augmentation ---
    img_size: int = 224
    aug: str = "basic"                # basic | flipv | color | trivial | randaug
    sampler: str | None = None        # None | balanced
    mix: str | None = None            # None | mixup | cutmix
    mix_alpha: float = 1.0
    # --- loss ---
    loss: str = "ce"                  # ce | ls | focal | ce_weighted
    label_smoothing: float = 0.0      # dùng khi loss == "ls" (nếu 0 thì lấy 0.1)
    focal_gamma: float = 2.0
    class_weight_beta: float | None = None   # dùng khi loss == "ce_weighted" (None -> 0.0)
    # --- tối ưu (công thức nền, GUIDE.md mục 1.4) ---
    epochs: int = 12
    batch_size: int = 64
    lr_backbone: float = 1e-4
    lr_head: float = 1e-3
    weight_decay: float = 0.05
    warmup_epochs: float = 1.0
    ema_decay: float | None = None
    amp: bool = True
    grad_clip: float = 1.0
    num_workers: int = 2
    # --- đường dẫn ---
    images_dir: str = "data/images"
    labels_dir: str = "data/labels"
    out_dir: str = "runs"             # config.json, history.csv, checkpoint, logit của từng lần chạy
    pred_dir: str = "predictions"     # file dự đoán đúng định dạng eval.py (nộp cùng bài)
    curves_dir: str = "curves"        # ảnh biểu đồ training
    note: str = ""                    # ghi chú: khác T00 ở điểm nào
    # --- chỉ bật ở Bước 4 (chung kết): ghi predictions trên TEST. Mặc định TẮT (quy tắc S4). ---
    save_test_predictions: bool = False


# Các field không ảnh hưởng kết quả huấn luyện: bỏ qua khi so sánh để quyết định có chạy lại hay không.
_NOT_TRAINING_FIELDS = {"save_test_predictions", "num_workers", "images_dir", "labels_dir",
                        "out_dir", "pred_dir", "curves_dir", "note"}


def run_dir(cfg: Config) -> Path:
    """Thư mục kết quả của một lần chạy: <out_dir>/<exp_id>/seed<k>/ ."""
    return Path(cfg.out_dir) / cfg.exp_id / f"seed{cfg.seed}"


def pred_path(cfg: Config, split: str) -> Path:
    """Đường dẫn chuẩn của file dự đoán: <pred_dir>/<exp_id>_seed<k>_<split>.csv (split = val | test)."""
    return Path(cfg.pred_dir) / f"{cfg.exp_id}_seed{cfg.seed}_{split}.csv"


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    """Cố định random, numpy, torch (CPU, CUDA). Worker của DataLoader được seed trong dataset.py."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_optimizer(model, cfg: Config):
    """AdamW với 3 nhóm tham số (xem model.param_groups); norm/bias không weight decay."""
    groups = M.param_groups(model, cfg.lr_backbone, cfg.lr_head, cfg.weight_decay)
    return torch.optim.AdamW(groups)


def lr_lambda_factory(total_steps: int, warmup_steps: int):
    def fn(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return fn


def build_scheduler(optimizer, cfg: Config, steps_per_epoch: int):
    """Warmup tuyến tính rồi cosine về 0, cập nhật THEO BƯỚC (iteration)."""
    total = cfg.epochs * steps_per_epoch
    warm = int(round(cfg.warmup_epochs * steps_per_epoch))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda_factory(total, warm))


class EMA:
    """W_ema <- d * W_ema + (1 - d) * W  (slide trang 56).

    Áp dụng cho mọi tensor số thực của state_dict, gồm cả running_mean/var của BatchNorm
    (đây là cách đơn giản và đủ tốt ở đây); tensor nguyên (num_batches_tracked) được sao chép.
    d được "khởi động": d_t = min(decay, (1 + t) / (10 + t)) để EMA không bị kéo về trọng số khởi tạo.
    """

    def __init__(self, model, decay: float):
        self.decay = decay
        self.steps = 0
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model) -> None:
        self.steps += 1
        d = min(self.decay, (1 + self.steps) / (10 + self.steps))
        msd = model.state_dict()
        for k, v in self.module.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(d).add_(msd[k].detach(), alpha=1 - d)
            else:
                v.copy_(msd[k])


def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, cfg: Config,
                    device, ema: EMA | None = None) -> dict:
    """Một epoch huấn luyện. Trả về {"train_loss", "train_acc", "lr", "lr_trace"}.

    train_acc chỉ có nghĩa khi không dùng Mixup/CutMix (khi đó ghi NaN).
    """
    M.set_train_mode(model)
    use_amp = scaler.is_enabled()
    total_loss, n, correct = 0.0, 0, 0
    lr_trace = []
    for x, y, _ in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        targets = y
        if cfg.mix:
            x, targets = L.mix_batch(x, y, cfg.mix_alpha, cfg.mix)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(x)
        loss = L.mixed_loss(criterion, logits, targets) if cfg.mix else criterion(logits.float(), y)
        scaler.scale(loss).backward()
        if cfg.grad_clip:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        lr_trace.append(scheduler.get_last_lr()[-1])
        if ema is not None:
            ema.update(model)
        bs = y.size(0)
        total_loss += loss.item() * bs
        n += bs
        if not cfg.mix:
            correct += (logits.argmax(1) == y).sum().item()
    return {"train_loss": total_loss / n,
            "train_acc": float("nan") if cfg.mix else correct / n,
            "lr": lr_trace[-1], "lr_trace": lr_trace}


@torch.inference_mode()
def evaluate(model, loader, criterion, device, amp: bool = False):
    """Chạy model ở chế độ eval. Trả về (filenames, y_true[N], logits[N, 9], loss)."""
    model.eval()
    names, ys, outs = [], [], []
    total, n = 0.0, 0
    for x, y, fn in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp and device.type == "cuda"):
            logits = model(x)
        logits = logits.float()
        total += criterion(logits, y).item() * y.size(0)
        n += y.size(0)
        names += list(fn)
        ys.append(y.cpu().numpy())
        outs.append(logits.cpu().numpy())
    return names, np.concatenate(ys), np.concatenate(outs), total / n


def softmax_np(z: np.ndarray) -> np.ndarray:
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def plot_curves(history: list[dict], path: str | Path, title: str, lr_trace=None) -> None:
    """Loss train/val, macro-F1 val (và top-1 val) theo epoch, LR theo bước -> ảnh .png."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    h = pd.DataFrame(history)
    ncols = 3 if lr_trace else 2
    fig, ax = plt.subplots(1, ncols, figsize=(5 * ncols, 3.8))
    ax[0].plot(h["epoch"], h["train_loss"], "o-", label="train loss")
    ax[0].plot(h["epoch"], h["val_loss"], "s-", label="val loss (CE)")
    ax[0].set(xlabel="epoch", ylabel="loss", title="Loss")
    ax[1].plot(h["epoch"], h["val_macro_f1"], "s-", label="val macro-F1")
    ax[1].plot(h["epoch"], h["val_top1"], "^--", label="val top-1")
    best = int(h["val_macro_f1"].values.argmax())
    ax[1].axvline(h["epoch"].iloc[best], color="gray", ls=":", label=f"best epoch {h['epoch'].iloc[best]}")
    ax[1].set(xlabel="epoch", ylabel="score", title="Chỉ số val")
    if lr_trace:
        ax[2].plot(lr_trace)
        ax[2].set(xlabel="bước", ylabel="LR (nhóm head)", title="LR theo bước")
    for a in ax:
        a.grid(alpha=0.3)
    for a in ax[:2]:
        a.legend()
    fig.suptitle(title)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _build_criterion(cfg: Config, train_df: pd.DataFrame, device):
    kw = {}
    if cfg.loss == "ls":
        kw["smoothing"] = cfg.label_smoothing or 0.1
    elif cfg.loss == "focal":
        kw["gamma"] = cfg.focal_gamma
    elif cfg.loss == "ce_weighted":
        counts = train_df["Label"].value_counts().reindex(range(D.NUM_CLASSES), fill_value=0).to_numpy()
        kw["weight"] = L.class_weights(counts, cfg.class_weight_beta or 0.0)
    return L.build_criterion(cfg.loss, **kw).to(device)


def _eval_loader(cfg: Config, df, mean, std):
    tf = D.build_transforms(False, cfg.img_size, mean=mean, std=std)
    return D.make_loader(df, cfg.images_dir, tf, cfg.batch_size, train=False, num_workers=cfg.num_workers)


def load_best_model(cfg: Config, device=None):
    """Dựng lại model và nạp checkpoint tốt nhất của một lần chạy (dùng cho Bước 3 và 4)."""
    device = device or get_device()
    model = M.build_model(cfg.backbone, pretrained=False, init="scratch")
    state = torch.load(run_dir(cfg) / "best.pt", map_location="cpu")
    model.load_state_dict(state)
    return model.to(device).eval()


def model_norm(model):
    """(mean, std) đúng theo pretrained_cfg của backbone."""
    pc = getattr(model, "pretrained_cfg", None) or {}
    return tuple(pc.get("mean", D.IMAGENET_MEAN)), tuple(pc.get("std", D.IMAGENET_STD))


def _save_split_outputs(cfg, ev, split, names, y, logits, rd):
    np.save(rd / f"{split}_logits.npy", logits)
    ev.save_predictions(pred_path(cfg, split), names, y, softmax_np(logits))


def run(cfg: Config, verbose: bool = True) -> dict:
    """Huấn luyện một cấu hình và lưu mọi thứ cần thiết. Trả về dict kết quả tóm tắt.

    Có thể chạy lại an toàn khi Colab bị ngắt: nếu config.json, best.pt, summary.json đã có và cấu hình
    huấn luyện không đổi thì bỏ qua phần train; chỉ sinh các file dự đoán còn thiếu.
    Test chỉ được ghi khi cfg.save_test_predictions = True và file test của run này chưa tồn tại
    (không ghi đè, để tuân thủ "test một lần mỗi seed").
    """
    ev = _eval_mod()
    device = get_device()
    rd = run_dir(cfg)
    rd.mkdir(parents=True, exist_ok=True)
    cfg_path, best_path, sum_path = rd / "config.json", rd / "best.pt", rd / "summary.json"
    key = {k: v for k, v in dataclasses.asdict(cfg).items() if k not in _NOT_TRAINING_FIELDS}
    key = json.loads(json.dumps(key))

    train_df, val_df, test_df = D.load_split(cfg.labels_dir, cfg.fold)
    D.check_split(train_df, val_df, test_df, cfg.images_dir, verbose=False)

    done = False
    if cfg_path.exists() and best_path.exists() and sum_path.exists():
        old = json.loads(cfg_path.read_text(encoding="utf-8"))["cfg"]
        done = {k: old.get(k) for k in key} == key

    if done:
        summary = json.loads(sum_path.read_text(encoding="utf-8"))
        if verbose:
            print(f"[{cfg.exp_id} seed{cfg.seed}] đã có kết quả, bỏ qua huấn luyện "
                  f"(val macro-F1 {summary['val_macro_f1']:.4f}).")
        model = load_best_model(cfg, device)
        mean, std = model_norm(timm_pretrained_stub(cfg))
        val_loader = _eval_loader(cfg, val_df, mean, std)
    else:
        for split in ("val", "test"):                      # file cũ của cấu hình khác -> xoá
            if pred_path(cfg, split).exists():
                pred_path(cfg, split).unlink()
                print(f"  (xoá {pred_path(cfg, split)} của cấu hình cũ)")
        set_seed(cfg.seed)
        model = M.build_model(cfg.backbone, pretrained=True, drop_rate=cfg.drop_rate, init=cfg.init)
        model.to(device)
        mean, std = model_norm(model)
        weights_tag = model.weights_tag
        params_m, gmacs = M.count_params(model), M.count_gmacs(model, cfg.img_size)
        env = {"torch": torch.__version__, "device": str(device),
               "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"}
        try:
            import timm
            env["timm"] = timm.__version__
        except Exception:
            pass
        cfg_path.write_text(json.dumps({"cfg": dataclasses.asdict(cfg), "env": env,
                                        "weights_tag": weights_tag}, indent=2, ensure_ascii=False),
                            encoding="utf-8")

        train_loader = D.make_loader(train_df, cfg.images_dir,
                                     D.build_transforms(True, cfg.img_size, cfg.aug, mean, std),
                                     cfg.batch_size, train=True, sampler=cfg.sampler,
                                     num_workers=cfg.num_workers, seed=cfg.seed)
        val_loader = _eval_loader(cfg, val_df, mean, std)
        criterion = _build_criterion(cfg, train_df, device)
        val_criterion = nn.CrossEntropyLoss()              # CE chung cho mọi cấu hình -> val loss so sánh được
        optimizer = build_optimizer(model, cfg)
        scheduler = build_scheduler(optimizer, cfg, len(train_loader))
        use_amp = cfg.amp and device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        ema = EMA(model, cfg.ema_decay) if cfg.ema_decay else None

        history, lr_trace, best_f1, best_epoch, best_state = [], [], -1.0, -1, None
        t_train = []
        for epoch in range(1, cfg.epochs + 1):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            tr = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, scaler, cfg, device, ema)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_train.append(time.perf_counter() - t0)
            lr_trace += tr.pop("lr_trace")
            eval_model = ema.module if ema is not None else model
            _, y_v, lg_v, val_loss = evaluate(eval_model, val_loader, val_criterion, device, amp=use_amp)
            met = ev.compute_metrics(y_v, lg_v.argmax(1), softmax_np(lg_v))
            row = {"epoch": epoch, **tr, "val_loss": val_loss, "val_macro_f1": met["macro_f1"],
                   "val_top1": met["top1"], "epoch_time_s": t_train[-1]}
            history.append(row)
            if met["macro_f1"] > best_f1:                  # strict '>' : hòa thì giữ epoch sớm hơn
                best_f1, best_epoch = met["macro_f1"], epoch
                best_state = {k: v.detach().cpu().clone() for k, v in eval_model.state_dict().items()}
                torch.save(best_state, best_path)
            if verbose:
                print(f"[{cfg.exp_id} s{cfg.seed}] ep{epoch:02d}/{cfg.epochs} "
                      f"train_loss {tr['train_loss']:.4f} val_loss {val_loss:.4f} "
                      f"val_F1 {met['macro_f1']:.4f} val_acc {met['top1']:.4f} ({t_train[-1]:.0f}s)")
            pd.DataFrame(history).to_csv(rd / "history.csv", index=False)

        curve = f"{cfg.exp_id}_{cfg.backbone}" + (f"_seed{cfg.seed}" if cfg.seed else "") + ".png"
        plot_curves(history, Path(cfg.curves_dir) / curve,
                    f"{cfg.exp_id} · {cfg.backbone} · seed {cfg.seed}", lr_trace)

        # nạp checkpoint tốt nhất vào model trần để đánh giá cuối
        model = M.build_model(cfg.backbone, pretrained=False, init="scratch")
        model.load_state_dict(best_state)
        model.to(device)
        names, y_v, lg_v, _ = evaluate(model, val_loader, val_criterion, device, amp=use_amp)
        _save_split_outputs(cfg, ev, "val", names, y_v, lg_v, rd)
        met = ev.compute_metrics(y_v, lg_v.argmax(1), softmax_np(lg_v))
        summary = {"exp_id": cfg.exp_id, "seed": cfg.seed, "backbone": cfg.backbone,
                   "weights_tag": weights_tag,
                   "init": cfg.init, "params_M": params_m, "gmacs": gmacs, "img_size": cfg.img_size,
                   "epochs": cfg.epochs, "best_epoch": best_epoch,
                   "val_macro_f1": met["macro_f1"], "val_top1": met["top1"], "val_ece": met["ece"],
                   "val_recall": [float(r) for r in met["recall"]], "val_f1": [float(r) for r in met["f1"]],
                   "train_time_per_epoch_s": float(np.mean(t_train)),
                   "total_train_time_s": float(np.sum(t_train)), "note": cfg.note}
        sum_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        pd.DataFrame(history).to_csv(rd / "history.csv", index=False)

    if not pred_path(cfg, "val").exists():
        names, y_v, lg_v, _ = evaluate(model, val_loader, nn.CrossEntropyLoss(), device, amp=False)
        _save_split_outputs(cfg, ev, "val", names, y_v, lg_v, rd)

    if cfg.save_test_predictions:
        if pred_path(cfg, "test").exists():
            print(f"  {pred_path(cfg, 'test')} đã tồn tại -> KHÔNG chạy lại test (S4).")
        else:
            names, y_t, lg_t, _ = evaluate(model, _eval_loader(cfg, test_df, mean, std),
                                           nn.CrossEntropyLoss(), device, amp=False)
            _save_split_outputs(cfg, ev, "test", names, y_t, lg_t, rd)
            if verbose:
                print(f"  đã ghi {pred_path(cfg, 'test')} (test chạy 1 lần cho seed {cfg.seed}).")
    return summary


def timm_pretrained_stub(cfg: Config):
    """Lấy pretrained_cfg (mean/std) mà không tải trọng số: tạo model rỗng pretrained=False."""
    import timm
    return timm.create_model(cfg.backbone, pretrained=False, num_classes=D.NUM_CLASSES)


def parse_overrides(pairs: list[str]) -> dict:
    """['seed=1', 'loss=focal', 'ema_decay=none'] -> dict, ép kiểu theo field của Config."""
    fields = {f.name: f for f in dataclasses.fields(Config)}
    out = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"'{pair}' phải có dạng KEY=VALUE")
        k, v = pair.split("=", 1)
        if k not in fields:
            raise KeyError(f"'{k}' không phải field của Config (có: {sorted(fields)})")
        ann = str(fields[k].type)
        nullable = "None" in ann
        if v.lower() in ("none", "null"):
            if not nullable:
                raise ValueError(f"{k} không nhận None")
            out[k] = None
        elif "bool" in ann:
            if v.lower() not in ("true", "false", "1", "0"):
                raise ValueError(f"{k}: '{v}' không phải bool")
            out[k] = v.lower() in ("true", "1")
        elif "float" in ann:
            out[k] = float(v)
        elif "int" in ann:
            out[k] = int(v)
        else:
            out[k] = v
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Huấn luyện một cấu hình DeepWeeds")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = ap.parse_args()
    cfg = Config(**parse_overrides(args.set))
    res = run(cfg)
    print(json.dumps({k: v for k, v in res.items() if k not in ("val_recall", "val_f1")},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
