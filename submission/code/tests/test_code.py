"""Test phần code hoàn chỉnh trong code/ (chạy trên CPU, không cần mạng hay dữ liệu thật).

Chạy từ thư mục gốc repo:
    python -m unittest discover -s code/tests -v
(Không chạy chung với `-s tests` vì starter/ và code/ có cùng tên module `train`.)
"""
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

CODE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE.parent))

import benchmark as BM  # noqa: E402
import dataset as D  # noqa: E402
import eval as ev  # noqa: E402
import inference as INF  # noqa: E402
import losses as L  # noqa: E402
import model as M  # noqa: E402
import experiments as EX  # noqa: E402
import train as TR  # noqa: E402


class TestLosses(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.logits = torch.randn(32, 9)
        self.y = torch.randint(0, 9, (32,))

    def test_focal_gamma0_equals_ce(self):
        ce = F.cross_entropy(self.logits, self.y)
        self.assertAlmostEqual(L.FocalLoss(0.0)(self.logits, self.y).item(), ce.item(), places=6)

    def test_label_smoothing_eps0_equals_ce(self):
        ce = F.cross_entropy(self.logits, self.y)
        self.assertAlmostEqual(L.LabelSmoothingCE(0.0)(self.logits, self.y).item(), ce.item(), places=6)

    def test_label_smoothing_matches_torch(self):
        ref = F.cross_entropy(self.logits, self.y, label_smoothing=0.1)
        self.assertAlmostEqual(L.LabelSmoothingCE(0.1)(self.logits, self.y).item(), ref.item(), places=5)

    def test_class_weights(self):
        counts = [1000, 100, 100, 100, 100, 100, 100, 100, 9000]
        for beta in (0.0, 0.999):
            w = L.class_weights(counts, beta)
            self.assertAlmostEqual(w.sum().item(), 9.0, places=4)
            self.assertGreater(w[1].item(), w[8].item())   # lớp hiếm có trọng số cao hơn
        self.assertAlmostEqual(L.class_weights(counts, 0.0).mean().item(), 1.0, places=5)

    def test_cutmix_lambda_is_real_area(self):
        np.random.seed(1)
        x = torch.arange(8 * 3 * 32 * 32, dtype=torch.float32).reshape(8, 3, 32, 32)
        y = torch.arange(8)
        for _ in range(20):
            _, (_, _, lam) = L.mix_batch(x, y, 1.0, "cutmix")
            self.assertTrue(0.0 <= lam <= 1.0)
        # kiểm tra chặt trên một ảnh: lam = 1 - diện tích dán / tổng
        np.random.seed(2)
        xm, (ya, yb, lam) = L.mix_batch(x, y, 1.0, "cutmix")
        perm_idx = [(yb[i].item()) for i in range(8)]
        i = next(j for j in range(8) if perm_idx[j] != j)
        diff = (xm[i] != x[i]).any(dim=0).float().mean().item()
        self.assertAlmostEqual(1 - lam, diff, places=6)

    def test_mixup_and_mixed_loss(self):
        np.random.seed(0)
        torch.manual_seed(0)
        x = torch.randn(4, 3, 8, 8)
        y = torch.arange(4)
        xm, (ya, yb, lam) = L.mix_batch(x, y, 0.4, "mixup")
        self.assertEqual(xm.shape, x.shape)
        crit = torch.nn.CrossEntropyLoss()
        lg = torch.randn(4, 9)
        want = lam * crit(lg, ya) + (1 - lam) * crit(lg, yb)
        self.assertAlmostEqual(L.mixed_loss(crit, lg, (ya, yb, lam)).item(), want.item(), places=6)

    def test_build_criterion(self):
        for kind, kw in [("ce", {}), ("ls", {"smoothing": 0.1}), ("focal", {"gamma": 2.0}),
                         ("ce_weighted", {"weight": torch.ones(9)})]:
            self.assertTrue(math.isfinite(L.build_criterion(kind, **kw)(self.logits, self.y).item()))
        with self.assertRaises(ValueError):
            L.build_criterion("nope")


class TestModelUtils(unittest.TestCase):
    def test_param_groups_no_wd_for_norm_bias(self):
        m = M.build_model("resnet18", pretrained=False, init="scratch")
        groups = M.param_groups(m, 1e-4, 1e-3, 0.05)
        by = {g["name"]: g for g in groups}
        self.assertEqual(by["backbone_norm_bias"]["weight_decay"], 0.0)
        self.assertEqual(by["backbone"]["weight_decay"], 0.05)
        self.assertEqual(by["head"]["lr"], 1e-3)
        n = sum(len(g["params"]) for g in groups)
        self.assertEqual(n, len(list(m.parameters())))
        self.assertTrue(all(p.ndim <= 1 for p in by["backbone_norm_bias"]["params"]))

    def test_freeze_keeps_bn_eval(self):
        m = M.build_model("resnet18", pretrained=False, init="scratch")
        M.freeze_backbone(m)
        M.set_train_mode(m)
        self.assertTrue(m.fc.training)
        self.assertFalse(m.bn1.training)
        self.assertFalse(m.layer1[0].bn1.training)
        trainable = [n for n, p in m.named_parameters() if p.requires_grad]
        self.assertTrue(all(n.startswith("fc") for n in trainable))
        self.assertTrue(all(g["name"] == "head" for g in M.param_groups(m, 1e-4, 1e-3, 0.05)))

    def test_count_params_and_gmacs(self):
        m = M.build_model("resnet18", pretrained=False, init="scratch")
        self.assertAlmostEqual(M.count_params(m), 11.18, delta=0.1)
        self.assertAlmostEqual(M.count_gmacs(m, 224), 1.82, delta=0.1)   # resnet18 ~ 1.8 GMAC

    def test_initial_loss_near_ln9(self):
        torch.manual_seed(0)
        m = M.build_model("resnet18", pretrained=False, init="scratch").eval()
        x = torch.randn(64, 3, 64, 64)
        y = torch.randint(0, 9, (64,))
        self.assertLess(abs(F.cross_entropy(m(x), y).item() - math.log(9)), 0.6)


class TestInference(unittest.TestCase):
    def _randomize_bn(self, m):
        for mod in m.modules():
            if isinstance(mod, torch.nn.BatchNorm2d):
                torch.nn.init.uniform_(mod.running_mean, -0.5, 0.5)
                torch.nn.init.uniform_(mod.running_var, 0.5, 2.0)
                torch.nn.init.uniform_(mod.weight, 0.5, 1.5)
                torch.nn.init.uniform_(mod.bias, -0.5, 0.5)

    def test_fuse_conv_bn_resnet(self):
        torch.manual_seed(0)
        m = M.build_model("resnet18", pretrained=False, init="scratch")
        self._randomize_bn(m)
        fused = INF.fuse_conv_bn(m, check_size=64, verbose=False)
        x = torch.randn(2, 3, 64, 64)
        with torch.no_grad():
            self.assertLess((m.eval()(x) - fused(x)).abs().max().item(), 1e-4)
        self.assertFalse(any(isinstance(mod, torch.nn.BatchNorm2d) for mod in fused.modules()))
        self.assertTrue(any(isinstance(mod, torch.nn.BatchNorm2d) for mod in m.modules()))   # bản gốc không đổi

    def test_fuse_conv_bn_mobilenet_batchnormact(self):
        torch.manual_seed(0)
        m = M.build_model("mobilenetv3_large_100", pretrained=False, init="scratch")
        self._randomize_bn(m)
        fused = INF.fuse_conv_bn(m, check_size=64, verbose=False)
        x = torch.randn(2, 3, 64, 64)
        with torch.no_grad():
            self.assertLess((m.eval()(x) - fused(x)).abs().max().item(), 1e-3)

    def test_fuse_noop_without_bn(self):
        m = M.build_model("deit_small_patch16_224", pretrained=False, init="scratch")
        out = INF.fuse_conv_bn(m, verbose=False)
        self.assertIsNotNone(out)

    def test_temperature_recovers_scale(self):
        rng = np.random.default_rng(0)
        z = rng.normal(size=(4000, 9)) * 2
        p = np.exp(z - z.max(1, keepdims=True))
        p /= p.sum(1, keepdims=True)
        y = np.array([rng.choice(9, p=pi) for pi in p])
        T = INF.fit_temperature(z * 3.0, y)            # mô hình quá tự tin 3 lần -> T ≈ 3
        self.assertAlmostEqual(T, 3.0, delta=0.25)
        before = ev.ece_score(INF.apply_temperature(z * 3.0, 1.0), y)
        after = ev.ece_score(INF.apply_temperature(z * 3.0, T), y)
        self.assertLess(after, before)
        np.testing.assert_array_equal(INF.apply_temperature(z, T).argmax(1), z.argmax(1))

    def test_aggregate_and_ensemble(self):
        a, b = np.random.randn(5, 9), np.random.randn(5, 9)
        for space in ("prob", "logit"):
            np.testing.assert_allclose(INF.aggregate_views([a, b], space).sum(1), 1.0, atol=1e-9)
        pe = INF.ensemble_probs([INF.aggregate_views([a]), INF.aggregate_views([b])])
        np.testing.assert_allclose(pe, INF.aggregate_views([a, b], "prob"))

    def test_views(self):
        x = torch.arange(2 * 3 * 8 * 8, dtype=torch.float32).reshape(2, 3, 8, 8)
        self.assertTrue(torch.equal(INF.view_hflip(INF.view_hflip(x)), x))
        self.assertEqual(len(INF.views_multicrop(x, 4)), 5)
        self.assertEqual(len(INF.views_multicrop(x, 4, flip=True)), 10)
        self.assertEqual(INF.views_multiscale(x, [4, 16])[1].shape[-1], 16)


class TestTrainParts(unittest.TestCase):
    def test_lr_schedule_warmup_cosine(self):
        fn = TR.lr_lambda_factory(total_steps=100, warmup_steps=10)
        vals = [fn(s) for s in range(100)]
        self.assertAlmostEqual(vals[0], 0.1)
        self.assertAlmostEqual(vals[9], 1.0)
        self.assertTrue(all(vals[i] >= vals[i + 1] - 1e-9 for i in range(10, 99)))
        self.assertLess(vals[-1], 0.01)

    def test_ema_tracks_model(self):
        m = M.build_model("resnet18", pretrained=False, init="scratch")
        ema = TR.EMA(m, 0.9)
        with torch.no_grad():
            for p in m.parameters():
                p.add_(1.0)
        for _ in range(200):
            ema.update(m)
        diff = max((a - b).abs().max().item() for a, b in zip(m.parameters(), ema.module.parameters()))
        self.assertLess(diff, 1e-3)

    def test_parse_overrides(self):
        o = TR.parse_overrides(["seed=3", "loss=focal", "ema_decay=none", "lr_head=0.002", "amp=false",
                                "mix=cutmix"])
        self.assertEqual(o, {"seed": 3, "loss": "focal", "ema_decay": None, "lr_head": 0.002,
                             "amp": False, "mix": "cutmix"})
        with self.assertRaises(KeyError):
            TR.parse_overrides(["nope=1"])
        with self.assertRaises(ValueError):
            TR.parse_overrides(["seed=none"])

    def test_bench(self):
        r = BM.bench(lambda: sum(range(1000)), warmup=3, iters=60)
        self.assertLessEqual(r["p50"], r["p95"])
        self.assertLessEqual(r["p95"], r["p99"])
        with self.assertRaises(ValueError):
            BM.bench(lambda: None, iters=10)
        m = M.build_model("resnet18", pretrained=False, init="scratch")
        rep = BM.latency_report(m, 1, 64, "fp32", "cpu", warmup=2, iters=50)
        self.assertEqual(rep["batch"], 1)
        tta = BM.tta_latency(m, 2, batch_size=1, img_size=64, dtype="fp32", device="cpu", warmup=2, iters=50)
        self.assertGreater(tta["p50"], rep["p50"])


class TestEndToEnd(unittest.TestCase):
    """Chạy train.run trên 90 ảnh giả (9 lớp x 10), 2 epoch, resnet18 từ đầu, CPU."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        (root / "images").mkdir()
        (root / "labels").mkdir()
        rng = np.random.default_rng(0)
        rows = []
        for c in range(9):
            for i in range(10):
                name = f"c{c}_{i}.jpg"
                base = np.full((256, 256, 3), 20 * c + 10, dtype=np.uint8)
                arr = np.clip(base + rng.integers(0, 30, base.shape), 0, 255).astype(np.uint8)
                Image.fromarray(arr).save(root / "images" / name)
                rows.append({"Filename": name, "Label": c, "Species": f"S{c}"})
        df = pd.DataFrame(rows)
        # mỗi lớp: 6 train, 2 val, 2 test -> 54/18/18 = 60/20/20
        parts = {"train": [], "val": [], "test": []}
        for c in range(9):
            sub = df[df.Label == c]
            parts["train"].append(sub.iloc[:6])
            parts["val"].append(sub.iloc[6:8])
            parts["test"].append(sub.iloc[8:])
        for k, v in parts.items():
            pd.concat(v).to_csv(root / "labels" / f"{k}_subset0.csv", index=False)
        df.to_csv(root / "labels" / "labels.csv", index=False)
        cls.root = root
        cls._old_total = D.TOTAL_IMAGES
        D.TOTAL_IMAGES = 90

    @classmethod
    def tearDownClass(cls):
        D.TOTAL_IMAGES = cls._old_total
        cls.tmp.cleanup()

    def cfg(self, **kw):
        r = self.root
        base = dict(exp_id="E2E", backbone="resnet18", init="scratch", epochs=2, batch_size=16,
                    amp=False, num_workers=0, images_dir=str(r / "images"), labels_dir=str(r / "labels"),
                    out_dir=str(r / "runs"), pred_dir=str(r / "predictions"), curves_dir=str(r / "curves"))
        base.update(kw)
        return TR.Config(**base)

    def test_check_split_detects_problems(self):
        tr, va, te = D.load_split(self.root / "labels")
        info = D.check_split(tr, va, te, self.root / "images", verbose=False)
        self.assertEqual(info["n"], {"train": 54, "val": 18, "test": 18})
        with self.assertRaises(AssertionError):                       # giao khác rỗng
            D.check_split(pd.concat([tr, va.iloc[:1]]), va, te, self.root / "images", verbose=False)
        with self.assertRaises(AssertionError):                       # thiếu file
            D.check_split(tr, va, te, self.root / "labels", verbose=False)

    def test_run_resume_and_test_once(self):
        cfg = self.cfg(seed=0, mix="cutmix", ema_decay=0.9, loss="ls", sampler="balanced", aug="color")
        res = TR.run(cfg, verbose=False)
        rd = TR.run_dir(cfg)
        for f in ("config.json", "history.csv", "summary.json", "best.pt", "val_logits.npy"):
            self.assertTrue((rd / f).exists(), f)
        self.assertTrue(list((self.root / "curves").glob("E2E_resnet18*.png")))
        self.assertTrue(TR.pred_path(cfg, "val").exists())
        self.assertFalse(TR.pred_path(cfg, "test").exists())           # test mặc định TẮT
        hist = pd.read_csv(rd / "history.csv")
        self.assertEqual(len(hist), 2)
        # checkpoint = epoch có macro-F1 val cao nhất (hòa lấy sớm hơn)
        self.assertEqual(res["best_epoch"], int(hist["val_macro_f1"].values.argmax()) + 1)
        pred = ev.read_pred(str(TR.pred_path(cfg, "val")))
        ev.check_against_csv(pred, str(self.root / "labels" / "val_subset0.csv"), "val")
        # chạy lại: bỏ qua huấn luyện, thêm test
        mtime = (rd / "best.pt").stat().st_mtime
        cfg_t = self.cfg(seed=0, mix="cutmix", ema_decay=0.9, loss="ls", sampler="balanced", aug="color",
                         save_test_predictions=True)
        TR.run(cfg_t, verbose=False)
        self.assertEqual((rd / "best.pt").stat().st_mtime, mtime)
        tp = TR.pred_path(cfg_t, "test")
        self.assertTrue(tp.exists())
        ev.check_against_csv(ev.read_pred(str(tp)), str(self.root / "labels" / "test_subset0.csv"), "test")
        t_mtime = tp.stat().st_mtime
        TR.run(cfg_t, verbose=False)                                   # test KHÔNG chạy lại
        self.assertEqual(tp.stat().st_mtime, t_mtime)
        # load_best_model cho cùng logit với file đã lưu
        model = TR.load_best_model(cfg)
        self.assertFalse(model.training)

    def test_experiments_pipeline(self):
        """inference_study -> run_baseline/run_final -> eval.py grade -> results.xlsx."""
        r = self.root
        common = dict(backbone="resnet18", init="scratch", epochs=1, batch_size=16, amp=False, num_workers=0,
                      images_dir=str(r / "images"), labels_dir=str(r / "labels"), out_dir=str(r / "runs2"),
                      pred_dir=str(r / "pred2"), curves_dir=str(r / "curves2"))
        EX.run_baseline(seeds=(0, 1, 2), **common)
        EX.run_final("F01", seeds=(0, 1, 2), views="hflip", agg="logit", temperature=True, **common)
        for k in range(3):
            for name in (f"T00_seed{k}_test", f"F01_seed{k}_test", f"F01_seed{k}_val", f"F01_uncal_seed{k}_test"):
                self.assertTrue((r / "pred2" / f"{name}.csv").exists(), name)
        # test chạy một lần: gọi lại predict_final không ghi đè
        cfg0 = TR.Config(exp_id="F01", seed=0, **common)
        mt = TR.pred_path(cfg0, "test").stat().st_mtime
        self.assertTrue(EX.predict_final(cfg0)["skipped"])
        self.assertEqual(TR.pred_path(cfg0, "test").stat().st_mtime, mt)
        # eval.py chấm được
        g = ev.load_group(str(r / "pred2" / "F01_seed*_test.csv"), str(r / "labels" / "test_subset0.csv"))
        self.assertEqual(len(g.preds), 3)
        ug = ev.load_group(str(r / "pred2" / "F01_uncal_seed*_test.csv"), None)
        self.assertEqual(len(ug.preds), 3)
        # nghiên cứu suy luận trên val
        runs = EX.collect_runs(str(r / "runs2"))
        self.assertEqual(set(runs.exp_id), {"T00", "F01"})
        df = EX.inference_study(TR.Config(exp_id="T00", seed=0, **common), device=torch.device("cpu"),
                                with_latency=True, lat_iters=50, resolutions=(224, 288),
                                ensemble_with=[TR.Config(exp_id="T00", seed=1, **common)])
        self.assertTrue({"I00", "I01", "I03", "I02a", "I02b", "I04_288", "I05", "I07", "I08a"} <= set(df.exp_id))
        self.assertTrue((df["val_macro_f1"].between(0, 1)).all())
        self.assertAlmostEqual(df.loc[df.exp_id == "I00", "cost_vs_I00"].iloc[0], 1.0)
        i07 = df[df.exp_id == "I07"].iloc[0]
        self.assertEqual(i07.val_top1, df.loc[df.exp_id == "I00", "val_top1"].iloc[0])  # TS không đổi accuracy
        # results.xlsx
        out = EX.build_results_xlsx(str(r / "results.xlsx"), runs, df, None,
                                    {"F01": "test F01", "T00": "mốc"}, str(r / "labels" / "labels.csv"),
                                    str(r / "labels" / "test_subset0.csv"), pred_dir=str(r / "pred2"))
        sheets = pd.read_excel(out, sheet_name=None)
        self.assertTrue({"Backbones", "Training", "Inference", "Final", "PerClass", "Summary"} <= set(sheets))
        self.assertIn("mean ± std", set(sheets["Final"]["seed"].astype(str)))

    def test_frozen_run_changes_only_head(self):
        cfg = self.cfg(exp_id="E2Efrozen", init="frozen", epochs=1)
        # init=frozen cần trọng số ImageNet (cần mạng); kiểm tra logic bằng model thủ công
        m = M.build_model("resnet18", pretrained=False, init="scratch")
        M.freeze_backbone(m)
        before = {k: v.clone() for k, v in m.state_dict().items()}
        opt = TR.build_optimizer(m, cfg)
        M.set_train_mode(m)
        x, y = torch.randn(8, 3, 64, 64), torch.randint(0, 9, (8,))
        F.cross_entropy(m(x), y).backward()
        opt.step()
        after = m.state_dict()
        changed = [k for k in before if not torch.equal(before[k], after[k])]
        self.assertTrue(changed and all(k.startswith("fc.") for k in changed), changed)

    def test_overfit_tiny_batch(self):
        torch.manual_seed(0)
        m = M.build_model("resnet18", pretrained=False, init="scratch")
        x, y = torch.randn(16, 3, 64, 64), torch.randint(0, 9, (16,))
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0)
        m.eval()
        for _ in range(60):
            opt.zero_grad()
            loss = F.cross_entropy(m(x), y)
            loss.backward()
            opt.step()
        self.assertLess(loss.item(), 0.1)


class TestAutoSelection(unittest.TestCase):
    """Chọn tự động chỉ dựa trên val: backbone, công thức, suy luận."""

    @staticmethod
    def _runs(rows):
        return pd.DataFrame([{"exp_id": e, "seed": s, "backbone": b, "val_macro_f1": f} for e, s, b, f in rows])

    def test_select_backbone_with_latency_limit(self):
        runs = self._runs([("B01", 0, "resnet50", 0.90), ("B06", 0, "efficientnet_b0", 0.88),
                           ("B03", 0, "convnext_tiny", 0.93)])
        lat = {"resnet50": {"p50": 12.0}, "efficientnet_b0": {"p50": 8.0}, "convnext_tiny": {"p50": 20.0}}
        self.assertEqual(EX.select_backbone(runs, lat), "convnext_tiny")
        self.assertEqual(EX.select_backbone(runs, lat, max_lat_ms=15), "resnet50")
        with self.assertRaises(ValueError):
            EX.select_backbone(runs, lat, max_lat_ms=1)

    def test_recipe_selection_uses_noise_threshold(self):
        base = EX.baseline_config("resnet18", seed=0)
        abl = EX.training_ablations("resnet18", seed=0)
        rows = [("T00", 0, "resnet18", 0.900), ("T00", 1, "resnet18", 0.902), ("T00", 2, "resnet18", 0.898),
                ("T05", 0, "resnet18", 0.930),   # cutmix: thắng rõ
                ("T06", 0, "resnet18", 0.915),   # mixup thắng nhưng kém cutmix -> cùng nhóm "mix", không chọn
                ("T07", 0, "resnet18", 0.9005),  # label smoothing: trong nhiễu -> bỏ
                ("T12", 0, "resnet18", 0.912),   # EMA: thắng vượt nhiễu (2·std = 0.004)
                ("T13", 0, "resnet18", 0.99)]    # độ phân giải 256 không bao giờ đưa vào kết hợp
        runs = self._runs(rows)
        self.assertAlmostEqual(EX.noise_threshold(runs), 0.004, places=6)
        combo, notes = EX.select_recipe(runs, abl, base)
        self.assertEqual(combo, {"mix": "cutmix", "ema_decay": 0.999})
        self.assertEqual(len(notes), len(EX.RECIPE_GROUPS))
        # T14 chưa chạy: chọn yếu tố đơn tốt nhất (T05); T13 bị loại dù F1 cao nhất
        self.assertEqual(EX.choose_final(runs), "T05")
        runs2 = pd.concat([runs, self._runs([("T14", 0, "resnet18", 0.94)])])
        self.assertEqual(EX.choose_final(runs2), "T14")
        runs3 = pd.concat([runs, self._runs([("T14", 0, "resnet18", 0.901)])])
        self.assertEqual(EX.choose_final(runs3), "T05")
        self.assertEqual(EX.choose_final(self._runs(rows[:3])), "T00")      # không có gì vượt nhiễu

    def test_combo_config_is_buildable_and_final_kwargs(self):
        cfg = TR.Config(exp_id="T14", seed=0, backbone="resnet18", mix="cutmix", ema_decay=0.999,
                        images_dir="x", out_dir="o")
        kw = EX.final_kwargs(cfg)
        for k in ("exp_id", "seed", "images_dir", "out_dir", "save_test_predictions"):
            self.assertNotIn(k, kw)
        f = TR.Config(exp_id="F01", seed=2, images_dir="x", **kw)
        self.assertEqual((f.mix, f.ema_decay, f.backbone), ("cutmix", 0.999, "resnet18"))

    def test_select_inference_prefers_cheap_unless_clear_gain(self):
        def df(i00, i01, i03, i02a):
            return pd.DataFrame({"exp_id": ["I00", "I01", "I03", "I02a", "I07", "I04_288"],
                                 "val_macro_f1": [i00, i01, i03, i02a, 0.99, 0.99]})
        self.assertEqual(EX.select_inference(df(0.90, 0.9005, 0.901, 0.9015)), ("none", "prob", "I00"))
        self.assertEqual(EX.select_inference(df(0.90, 0.91, 0.915, 0.905)), ("hflip", "logit", "I03"))
        self.assertEqual(EX.select_inference(df(0.90, 0.91, 0.905, 0.92)), ("crop5", "prob", "I02a"))


if __name__ == "__main__":
    unittest.main()
