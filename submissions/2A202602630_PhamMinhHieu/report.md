# Báo cáo Lab Day 2 — Backbone, công thức huấn luyện và suy luận trên DeepWeeds

**Sinh viên:** Phạm Minh Hiếu · **MSSV:** 2A202602630 · **Nền tảng:** Kaggle, GPU Tesla T4 · **Dữ liệu:** DeepWeeds, fold 0 chia sẵn

Mọi con số dưới đây lấy từ log chạy thật (`log.txt`), `results.xlsx` và các file trong `predictions/`; chỉ số test tính lại bằng `eval.py` gốc (không sửa). Số trích từ bài báo gốc được ghi rõ là *trích dẫn*.

---

## 1. Tóm tắt

- **Bài toán:** phân loại 9 lớp (8 loài cỏ dại + `Negatives`) trên 17.509 ảnh, dữ liệu mất cân bằng (lớp `Negatives` chiếm 52%).
- **Đã làm:** 7 backbone (B01–B07), 14 thí nghiệm công thức huấn luyện (T01–T14, 7 trục), 13 phương pháp suy luận trên val (I00–I08), chung kết F01 và mốc T00 với 3 seed.
- **Backbone tốt nhất:** `convnext_tiny` (macro-F1 val 0,9636), vượt xa ResNet-50 (0,8016), ResNeXt-50 (0,7521), EfficientNet-B0 (0,8184) và MobileNetV3 (0,7275); chênh lệch này lớn hơn nhiễu rất nhiều, nhưng *không tách được* ảnh hưởng của kiến trúc và của bộ trọng số tiền huấn luyện (xem mục 3).
- **Cấu hình chung kết F01:** ConvNeXt-T + công thức T14 (TrivialAugment + CutMix + label smoothing 0,1 + sampler cân bằng + LR head = LR backbone + EMA 0,999) + TTA 10 view (5-crop + lật) + temperature scaling.
- **Kết quả test (3 seed, mean ± std):** top-1 **0,9739 ± 0,0009**, macro-F1 **0,9699 ± 0,0014**, balanced accuracy 0,9797 ± 0,0018, ECE 0,0057 ± 0,0014. Mốc T00 + 1 view: top-1 0,9737 ± 0,0030, macro-F1 0,9669 ± 0,0024, ECE 0,0187 ± 0,0018.
- **Mức cải thiện so với mốc nhỏ:** macro-F1 +0,0030 (Welch p ≈ 0,15, chưa phân biệt được với nhiễu khi chỉ có 3 seed); top-1 không đổi (p ≈ 0,92); balanced accuracy +0,0097 (p ≈ 0,03, chưa hiệu chỉnh đa so sánh). Cải thiện rõ nhất là hiệu chuẩn: ECE test 0,0889 → 0,0057 nhờ temperature scaling.
- **Yếu tố đóng góp nhiều nhất:** backbone/trọng số (khoảng 0,16–0,24 macro-F1 giữa tốt nhất và kém nhất) ≫ công thức huấn luyện (+0,006 đến +0,010 trên val) ≫ suy luận (+0,002 đến +0,003 với TTA).
- **Thời gian thực:** ConvNeXt-T một view trên T4 cho p95 ≈ 5,9 ms ở batch 1 (FP32); cấu hình chung kết với 10 view có p95 ≈ 58 ms, vẫn dưới 100 ms.

---

## 2. Dữ liệu và thiết lập

### 2.1 Dữ liệu và kiểm tra chia

Dùng `train_subset0.csv`, `val_subset0.csv`, `test_subset0.csv` nguyên bản của tác giả (không sửa, không lọc, không chia lại). Kết quả kiểm tra bắt buộc (log, `split_check.json`):

| Tập | Số ảnh | Tỉ lệ | Chinee Apple | Lantana | Parkinsonia | Parthenium | Prickly Acacia | Rubber Vine | Siam Weed | Snake Weed | Negatives |
|---|---|---|---|---|---|---|---|---|---|---|---|
| train | 10.501 | 60,0% | 675 | 637 | 618 | 613 | 637 | 605 | 644 | 609 | 5.463 |
| val | 3.501 | 20,0% | 225 | 213 | 206 | 204 | 212 | 202 | 215 | 203 | 1.821 |
| test | 3.507 | 20,0% | 226 | 213 | 207 | 205 | 213 | 202 | 215 | 204 | 1.822 |

- Giao train∩val, train∩test, val∩test đều **bằng 0**; hợp ba tập **17.509** ảnh; **0** file thiếu. Tỉ lệ lớp lớn nhất / nhỏ nhất = **9,02**.
- Đối chiếu Table 1 của bài báo: tổng theo lớp khớp ở 7/9 lớp; Chinee Apple có 1.126 ảnh (bài báo 1.125) và Lantana có 1.063 (bài báo 1.064), lệch 1 ảnh mỗi lớp, tổng vẫn 17.509. Số đếm thật được dùng trong báo cáo này.
- Số ảnh train/val/test (10.501 / 3.501 / 3.507) hơi khác số suy ra từ tỉ lệ 60/20/20 (≈10.505 / 3.502 / 3.502); lệch dưới 0,1 điểm phần trăm.
- Biểu đồ phân bố lớp, ảnh mẫu và ảnh sau augmentation được lưu trong output của notebook (`eda_class_distribution.png`, `eda_samples.png`, `aug_check.png`).

### 2.2 Kiểm tra pipeline trước khi chạy thật

- Loss ban đầu của head mới: **2,185** (kỳ vọng ln 9 = 2,197).
- Overfit một batch 16 ảnh (ResNet-50): loss 2,200 → 0,566 (bước 10) → 0,0005 (bước 20) → 0,0000 (bước 30).
- Ảnh sau augmentation vẽ cùng nhãn để kiểm tra khớp (`aug_check.png`).

### 2.3 Công thức nền T00 (mọi backbone dùng chung)

Khởi tạo ImageNet (tag `timm` ghi ở mục 3), thay head 9 lớp, tinh chỉnh toàn bộ · train: `RandomResizedCrop(224, scale 0,25–1)` + lật ngang; eval: resize 256 rồi `CenterCrop(224)` · AdamW, LR backbone 1e-4, LR head 1e-3, weight decay 0,05 (không áp dụng cho norm/bias) · warmup 1 epoch rồi cosine, cập nhật theo bước · CE · batch 64 · 12 epoch · AMP · clip gradient 1,0 · chọn checkpoint theo macro-F1 val (hòa lấy epoch sớm hơn).

### 2.4 Chỉ số, phần cứng, seed

- Macro-F1 (chỉ số chính, 9 lớp), top-1, balanced accuracy, ECE 15 bin, mean ± std với `ddof=1`, đều theo định nghĩa của `eval.py`.
- Phần cứng và thư viện: Tesla T4 (dùng 1 trong 2 GPU của Kaggle), Python 3.13.15, torch 2.11.0+cu128, timm 1.0.29.
- Seed: B01–B07 và T01–T14 chạy **1 seed (seed 0)**; T00 và F01 chạy **3 seed (0, 1, 2)**.
- Toàn bộ pipeline (Bước 0–5) chạy một lần trên Kaggle, khoảng 17.860 s ≈ 5 giờ.

---

## 3. So sánh backbone (B01–B07, công thức T00, seed 0)

| exp_id | Backbone | Tag trọng số | Tham số (M) | GMAC | macro-F1 val | top-1 val | Thời gian train / epoch (s) | Độ trễ batch 1 p50 (ms) |
|---|---|---|---|---|---|---|---|---|
| B01 | resnet50 | a1_in1k | 23,5 | 4,09 | 0,8016 | 0,8563 | 40,0 | 5,83 |
| B02 | resnext50_32x4d | a1h_in1k | 23,0 | 4,23 | 0,7521 | 0,8021 | 51,1 | 7,74 |
| **B03** | **convnext_tiny** | in12k_ft_in1k | 27,8 | 4,45 | **0,9636** | **0,9712** | 47,1 | **5,50** |
| B04 | deit_small_patch16_224 | fb_in1k | 21,7 | 4,60 | 0,9515 | 0,9660 | 30,6 | 4,97 |
| B05 | swin_tiny_patch4_window7_224 | ms_in1k | 27,5 | 4,49 | 0,9575 | 0,9686 | 59,0 | 10,27 |
| B06 | efficientnet_b0 | ra_in1k | 4,0 | 0,38 | 0,8184 | 0,8623 | 27,0 | 7,72 |
| B07 | mobilenetv3_large_100 | ra_in1k | 4,2 | 0,22 | 0,7275 | 0,7963 | 18,3 | 6,19 |

Độ trễ: forward FP32, batch 1, 224×224, T4, warmup + `synchronize`, 100 lần đo, không gồm tiền xử lý (p95/p99 trong sheet `Latency` và log). Đường cong: `curves/B0x_*.png`; hình đánh đổi: `figures/backbone_tradeoff.png`.

**Nhận xét**

- **ConvNeXt-T tốt nhất** (0,9636), kế đó Swin-T (0,9575) và DeiT-S (0,9515). ResNet-50, ResNeXt-50 và hai mạng nhẹ thấp hơn nhiều (0,73–0,82). Khoảng cách giữa nhóm đầu và nhóm cuối (≥ 0,14) vượt xa nhiễu seed đo được ở T00 (std val 0,0006), nên đây là khác biệt thật *trong điều kiện thí nghiệm này*.
- **Không kết luận được “kiến trúc nào tốt hơn”.** Mỗi backbone dùng một bộ trọng số khác nhau: ConvNeXt-T dùng `in12k_ft_in1k` (tiền huấn luyện trên ImageNet-12k thêm dữ liệu), còn ResNet-50 và ResNeXt-50 dùng tag `a1_in1k`/`a1h_in1k`. Thứ hạng có thể phản ánh dữ liệu tiền huấn luyện và công thức của trọng số hơn là kiến trúc. Giả thuyết (chưa kiểm chứng): trọng số `a1*` ít hợp với tinh chỉnh bằng AdamW, LR 1e-4, CE, 12 epoch. Việc ResNet-50 và ResNeXt-50 dao động mạnh ở các epoch đầu (val loss của B02 nhảy lên 2,48 ở epoch 4) phù hợp với giả thuyết này nhưng không chứng minh nó.
- **FLOPs không dự đoán độ trễ:** EfficientNet-B0 (0,38 GMAC) chậm hơn ConvNeXt-T (4,45 GMAC) ở batch 1 (7,72 so với 5,50 ms); Swin-T chậm nhất (10,27 ms) dù GMAC tương đương.
- **Quá khớp nhẹ ở ConvNeXt-T:** train loss 0,016 so với val loss 0,148 ở epoch 12; val loss thấp nhất ở epoch 8 (0,130) nhưng macro-F1 val còn tăng nhẹ đến epoch 12.
- **Chọn backbone đi tiếp:** `convnext_tiny` theo quy tắc “macro-F1 val cao nhất” (không đặt ràng buộc độ trễ). Lựa chọn này cũng hợp lý theo đánh đổi: p50 5,50 ms là nhanh thứ hai (sau DeiT-S 4,97 ms, thấp hơn 0,012 macro-F1).

---

## 4. Công thức huấn luyện (T00–T14, backbone ConvNeXt-T)

**Thiết kế:** mỗi thí nghiệm T01–T13 chỉ khác T00 đúng một yếu tố (kiểm soát từng yếu tố một, *không* dùng cách tham lam: mọi Δ tính so với T00). T14 là kết hợp. Nhiễu: T00 chạy 3 seed cho macro-F1 val 0,9636 / 0,9638 / 0,9627 (mean 0,9634, std 0,0006). Std này từ 3 seed nên có thể bị đánh giá thấp: macro-F1 *test* của T00 qua 3 seed có std 0,0024. Vì vậy dưới đây chỉ gọi là “khác biệt đáng tin hơn” khi Δ ≳ 0,005 (≈ 2 × 0,0024), còn Δ nhỏ hơn được coi là *không phân biệt được* dù quy tắc chọn tự động dùng ngưỡng thấp hơn (0,002, xem mục 8).

| exp_id | Trục | Khác T00 ở điểm nào | macro-F1 val | top-1 val | Δ so với T00 | ECE val |
|---|---|---|---|---|---|---|
| T00 | — | nền (seed 0) | 0,9636 | 0,9712 | — | 0,0198 |
| T01 | A. khởi tạo | đóng băng backbone, chỉ train head | 0,8617 | 0,8895 | −0,1019 | 0,0254 |
| T02 | A. khởi tạo | train từ đầu (không ImageNet) | 0,2885 | 0,5216 | −0,6751 | 0,0553 |
| T03 | B. augmentation | thêm ColorJitter | 0,9665 | 0,9723 | +0,0029 | 0,0209 |
| T04 | B. augmentation | TrivialAugmentWide | 0,9705 | 0,9777 | **+0,0069** | 0,0142 |
| T05 | B. augmentation | CutMix (α=1) | **0,9738** | 0,9791 | **+0,0102** | 0,0062 |
| T06 | B. augmentation | Mixup (α=1) | 0,9650 | 0,9729 | +0,0014 | 0,0180 |
| T07 | C. loss | label smoothing 0,1 | 0,9663 | 0,9740 | +0,0027 | 0,0795 |
| T08 | C. loss | focal loss γ=2 | 0,9662 | 0,9737 | +0,0026 | 0,0161 |
| T09 | C. loss | CE trọng số lớp (class-balanced, β=0,999) | 0,9630 | 0,9700 | −0,0006 | 0,0165 |
| T10 | D. sampler | sampler cân bằng lớp | 0,9661 | 0,9734 | +0,0025 | 0,0186 |
| T11 | E. LR | LR head = LR backbone (1e-4) | 0,9673 | 0,9743 | +0,0037 | 0,0169 |
| T12 | F. chính quy hoá | EMA trọng số (0,999) | 0,9697 | 0,9769 | **+0,0060** | 0,0130 |
| T13 | G. độ phân giải | train và test ở 256 | 0,9678 | 0,9746 | +0,0042 | 0,0181 |
| T14 | kết hợp | T04 + T05 + T07 + T10 + T11 + T12 | 0,9694 | 0,9749 | +0,0058 | 0,0843 |

Hình: `figures/ablation_delta.png`; đường cong: `curves/T0x_*.png`, `curves/T14_*.png`.

**Nhận xét**

1. **Khởi tạo là yếu tố lớn nhất.** Train từ đầu gần như không học trong 12 epoch (macro-F1 val 0,2885, không hội tụ); đóng băng backbone chỉ đạt 0,8617. Tinh chỉnh toàn bộ từ trọng số tiền huấn luyện cho 0,9636. Với ~10 nghìn ảnh và 12 epoch, trọng số tiền huấn luyện là điều kiện cần.
2. **Augmentation là trục có tác dụng đáng tin nhất:** CutMix +0,0102, TrivialAugment +0,0069. Mixup (+0,0014) và ColorJitter (+0,0029) không phân biệt được với nhiễu. Giả thuyết (chưa kiểm chứng): CutMix giúp hơn Mixup vì ảnh cỏ dại có đặc trưng cục bộ (hoa, lá) và cắt dán giữ nguyên kết cấu cục bộ.
3. **EMA** (+0,0060) là yếu tố “miễn phí” ở suy luận, hiệu quả ngang TrivialAugment.
4. **Loss không phân biệt được:** CE 0,9636, label smoothing 0,9663, focal 0,9662, CE có trọng số lớp 0,9630; tất cả trong khoảng 0,003. Tương tự sampler cân bằng (+0,0025) và LR head (+0,0037). Với `Negatives` chiếm 52%, các cách xử lý mất cân bằng không cải thiện macro-F1 val rõ rệt ở điều kiện này; ảnh hưởng lên recall từng lớp thấy rõ hơn ở bộ test (mục 6).
5. **Label smoothing làm mô hình “thiếu tự tin”:** ECE val tăng từ 0,0198 lên 0,0795 (T07) và 0,0843 (T14), trong khi accuracy không đổi nhiều. Temperature scaling sửa được (mục 5). CutMix lại cho ECE val thấp nhất (0,0062).
6. **Kết hợp triệt tiêu, không cộng dồn.** Tổng Δ của sáu yếu tố đơn trong T14 là +0,0320, nhưng T14 chỉ đạt Δ = +0,0058; T14 (0,9694) thấp hơn CutMix đơn (0,9738), TrivialAugment đơn (0,9705) và EMA đơn (0,9697). Giả thuyết (chưa kiểm chứng): chồng nhiều chính quy hoá mạnh (TrivialAugment + CutMix + label smoothing) trong lịch chỉ 12 epoch làm mô hình khớp chậm hơn (train loss của T14 còn 0,955 ở epoch 12, val loss 0,1734 cao hơn T00 0,1475). Mỗi con số ở đây từ 1 seed; chênh T14 so với T05 (0,0044) nhỏ hơn mức nhiễu test của T00 (0,0024 × 2), nên không khẳng định T05 chắc chắn tốt hơn T14.
7. **Độ phân giải 256** (+0,0042, chậm hơn 1,3×: 59,6 so với 46,0 s/epoch; GMAC 5,82 so với 4,45) không phân biệt được với nhiễu.

---

## 5. Suy luận (I00–I08, trên val, mô hình T14 seed 0)

Mọi số trong bảng này là **val**, từ một mô hình (T14 seed 0). Độ trễ: batch 1, FP32, T4, forward model (không gồm tiền xử lý), 100 lần đo, warmup, `synchronize`. Với TTA K view, độ trễ đo K lần forward liên tiếp ở batch 1.

| exp_id | Phương pháp | K | macro-F1 val | top-1 val | ECE val | p50 / p95 / p99 (ms) | Chi phí so với I00 |
|---|---|---|---|---|---|---|---|
| I00 | 1 view (center crop 224) | 1 | 0,9689 | 0,9746 | 0,0840 | 5,48 / 6,72 / 6,94 | 1,00× |
| I01 | TTA lật, gộp xác suất | 2 | 0,9710 | 0,9763 | 0,0865 | 11,09 / 11,81 / 13,00 | 2,02× |
| I03 | TTA lật, gộp logit | 2 | 0,9712 | 0,9763 | 0,0859 | 11,02 / 11,66 / 12,08 | 2,01× |
| I02a | 5-crop 224 | 5 | 0,9704 | 0,9757 | 0,0872 | 28,94 / 31,05 / 32,20 | 5,28× |
| I02b | 5-crop + lật (10 view) | 10 | 0,9715 | 0,9771 | 0,0888 | 54,76 / 58,17 / 70,56 | 9,99× |
| I04_256 | độ phân giải test 256 | 1 | 0,9712 | 0,9763 | 0,1246 | 6,08 / 7,44 / 7,46 | 1,11× |
| I04_288 | độ phân giải test 288 (phóng từ 256) | 1 | **0,9795** | **0,9837** | 0,1547 | 7,44 / 7,76 / 12,48 | 1,36× |
| I04_320 | độ phân giải test 320 (phóng từ 256) | 1 | 0,9784 | 0,9829 | 0,1857 | 9,33 / 13,22 / 13,44 | 1,70× |
| I05 | ensemble 3 mô hình (T14 s0 + T00 s1, s2) | 3 | 0,9721 | 0,9780 | 0,0310 | 16,44 / 17,38 / 17,81 (ước lượng = 3 × một mô hình) | 3,00× |
| I07 | temperature scaling, T = 0,654 | 1 | 0,9689 | 0,9746 | **0,0059** | 5,56 / 5,88 / 6,33 | 1,01× |
| I08a | gộp BN vào Conv | 1 | 0,9689 | 0,9746 | 0,0840 | 5,60 / 9,03 / 10,06 | 1,02× |
| I08b | AMP (autocast fp16) | 1 | 0,9694 | 0,9749 | 0,0843 | 7,96 / 9,38 / 10,16 | 1,45× |
| I08c | FP16 (`model.half()`) | 1 | 0,9689* | 0,9746* | 0,0840* | 5,56 / 8,99 / 10,17 | 1,01× |

\* I08c: độ chính xác lấy theo FP32 (chưa chạy lại bằng `half()`); chỉ độ trễ đo bằng `half()`.

Độ trễ theo batch và dtype (sheet `Latency`): FP32 batch 1 p50 5,61 ms, batch 32 123,4 ms (259 ảnh/s); AMP batch 1 7,62 ms, batch 32 47,9 ms (668 ảnh/s); FP16 batch 1 5,60 ms, batch 32 38,3 ms (835 ảnh/s). Hình đánh đổi: `figures/inference_tradeoff.png`.

**Nhận xét**

1. **TTA cho lợi ích nhỏ và đắt:** lật ngang +0,0021 đến +0,0023 macro-F1 val với 2× độ trễ; 10 view +0,0026 với ~10× độ trễ. Khác biệt giữa I01, I03, I02a, I02b (≤ 0,0011) *không phân biệt được* với nhiễu; gộp xác suất (0,9710) và gộp logit (0,9712) cũng như nhau.
2. **Độ phân giải test là cách rẻ nhất để tăng:** test ở 288 cho macro-F1 val 0,9795 (+0,0106 so với I00) với 1,36× độ trễ, cao nhất trong mọi phương pháp. Giả thuyết phù hợp với FixRes (chưa kiểm chứng): `RandomResizedCrop` khi train làm vật thể trông lớn hơn ở test dùng center crop, nên tăng độ phân giải test bù lại. Đổi lại ECE tăng mạnh (0,155) nếu không hiệu chuẩn lại. **Phương pháp này không được dùng ở chung kết** (xem mục 8, hạn chế 3).
3. **Ensemble 3 mô hình** (+0,0032) đồng thời giảm ECE xuống 0,0310 (trước khi temperature scaling), với 3× chi phí.
4. **Hiệu chuẩn:** T = 0,654 (< 1: mô hình thiếu tự tin, do label smoothing) giảm ECE val từ 0,0840 xuống 0,0059 mà không đổi accuracy. ECE val sau hiệu chuẩn này là **trong mẫu** (T khớp trên chính val); kiểm chứng ngoài mẫu trên test ở mục 6 (ECE 0,0889 → 0,0057).
5. **Gộp BN không áp dụng được cho ConvNeXt** (dùng LayerNorm): code báo “không có cặp Conv-BN”, nên dòng I08a là bản sao của FP32; chênh lệch độ trễ (5,60 so với 5,48 ms) chỉ là nhiễu đo, không phải tăng tốc.
6. **AMP chậm hơn FP32 ở batch 1** (7,96 so với 5,48 ms; Latency: 7,62 so với 5,61 ms) nhưng nhanh hơn 2,6× ở batch 32; FP16 nhanh hơn 3,2× ở batch 32. Phù hợp cảnh báo của slide (đo thật, không giả định).
7. **Phù hợp ngoại tuyến:** ensemble, 10-view, độ phân giải cao. **Phù hợp thời gian thực:** 1 view (+ temperature scaling, miễn phí về độ trễ), có thể thêm độ phân giải 288 (1,36×) sau khi kiểm chứng riêng.

---

## 6. Cấu hình tốt nhất và kết quả chung kết (test, 3 seed)

### 6.1 Cấu hình

F01 = `convnext_tiny` (`in12k_ft_in1k`), công thức T14: TrivialAugmentWide + CutMix (α=1) + label smoothing 0,1 + sampler cân bằng + LR head 1e-4 (= LR backbone) + EMA 0,999 (còn lại như T00) · suy luận: TTA 10 view (5-crop 224 + lật), gộp xác suất, temperature scaling với T khớp trên val của chính từng seed (T = 0,641 / 0,625 / 0,631). Seed 0, 1, 2. Cấu hình được **chọn tự động chỉ dựa trên val** bằng các quy tắc ở mục 8 (không dùng test). Test chạy đúng **một lần cho mỗi seed** (log: “test chạy 1 lần cho seed k”; code từ chối ghi đè file test đã có).

### 6.2 Kết quả

| exp_id | seed | macro-F1 val | macro-F1 test | top-1 test | balanced acc test | ECE test |
|---|---|---|---|---|---|---|
| F01 | 0 | 0,9720 | 0,9710 | 0,9741 | 0,9802 | 0,0061 |
| F01 | 1 | 0,9748 | 0,9704 | 0,9746 | 0,9811 | 0,0041 |
| F01 | 2 | 0,9741 | 0,9684 | 0,9729 | 0,9777 | 0,0068 |
| **F01 mean ± std** | | 0,9737 | **0,9699 ± 0,0014** | **0,9739 ± 0,0009** | 0,9797 ± 0,0018 | **0,0057 ± 0,0014** |
| T00 | 0 | 0,9636 | 0,9672 | 0,9738 | 0,9672 | 0,0180 |
| T00 | 1 | 0,9638 | 0,9643 | 0,9706 | 0,9743 | 0,0207 |
| T00 | 2 | 0,9627 | 0,9691 | 0,9766 | 0,9684 | 0,0172 |
| **T00 mean ± std (mốc)** | | 0,9634 | 0,9669 ± 0,0024 | 0,9737 ± 0,0030 | 0,9700 ± 0,0038 | 0,0187 ± 0,0018 |

Chưa temperature scaling (`F01_uncal`): cùng accuracy và macro-F1, ECE test **0,0889 ± 0,0031**, NLL 0,1715 → sau hiệu chuẩn 0,0972.

**So với mốc (T00 + 1 view):**

| Chỉ số | Δ (F01 − T00) | Welch t-test (n=3 mỗi nhóm) | Kết luận |
|---|---|---|---|
| macro-F1 | +0,0030 | p ≈ 0,147 | lớn hơn std của mốc (0,0024) nhưng chưa phân biệt được ở mức 0,05 |
| top-1 | +0,0002 | p ≈ 0,92 | không đổi |
| balanced accuracy | +0,0097 | p ≈ 0,031 (chưa hiệu chỉnh đa so sánh) | có khả năng thật nhưng chỉ 3 seed |
| ECE | −0,0130 | — | F01 hiệu chuẩn tốt hơn nhiều (chủ yếu nhờ temperature scaling) |

**Độ chênh val–test:** macro-F1 val 0,9737, test 0,9699, chênh 0,0037 (≤ 0,02).

**Theo lớp (test, mean 3 seed):**

| Lớp | Số ảnh test | F01 precision | F01 recall | F01 F1 | T00 recall | T00 F1 |
|---|---|---|---|---|---|---|
| Chinee Apple | 226 | 0,9672 | **0,9528** | 0,9599 | 0,9410 | 0,9580 |
| Lantana | 213 | 0,9501 | 0,9828 | 0,9662 | 0,9812 | 0,9632 |
| Parkinsonia | 207 | 0,9686 | 0,9936 | 0,9809 | 0,9807 | 0,9783 |
| Parthenium | 205 | 0,9793 | 0,9870 | 0,9831 | 0,9821 | 0,9781 |
| Prickly Acacia | 213 | 0,9129 | 0,9844 | 0,9473 | 0,9624 | 0,9476 |
| Rubber Vine | 202 | 0,9738 | 0,9785 | 0,9761 | 0,9670 | 0,9726 |
| Siam Weed | 215 | 0,9598 | 0,9953 | 0,9772 | 0,9876 | 0,9682 |
| Snake Weed | 204 | 0,9447 | **0,9755** | 0,9598 | 0,9493 | 0,9540 |
| Negatives | 1.822 | 0,9908 | 0,9673 | 0,9789 | 0,9782 | 0,9819 |

Hai lớp khó của bài báo (recall, *trích dẫn* ResNet-50, trung bình 5 fold, ~100 epoch: Chinee Apple 88,5%, Snake Weed 88,8%): F01 đạt **95,3%** và **97,5%**. So sánh chỉ mang tính tham khảo vì khác điều kiện huấn luyện, khác cách đo (bài báo ghi “weighted average accuracy”) và chỉ dùng fold 0. Tương tự, top-1 0,9739 của F01 so với 95,7% của bài báo.

Tự chấm phần I bằng `eval.py grade` (đề xuất, giảng viên xác nhận): I1 7/7, I2 4/5, I3 4/4, I4 2/2, I5 2/2, tổng **19/20**. Ghi chú về I5 ở mục 8 (hạn chế 6).

### 6.3 Phân tích lỗi

Ma trận nhầm lẫn tổng 3 seed: `figures/confusion_F01_vs_T00.png`. Tổng số lỗi gần như bằng nhau (F01 275 / 10.521, T00 277 / 10.521) nhưng **phân bố lỗi khác**:

- **Phần lớn lỗi liên quan lớp `Negatives`** (nền không phải loài mục tiêu, rất đa dạng): `Negatives` bị đoán thành loài cỏ (dương tính giả) 179 lần ở F01 và 119 lần ở T00, nhiều nhất thành Prickly Acacia (58 so với 32), Lantana (33 so với 26), Siam Weed (26 so với 29). Ngược lại cỏ bị đoán là `Negatives` giảm từ 78 xuống 49.
- Nghĩa là F01 **đánh đổi recall của `Negatives` (0,978 → 0,967) lấy recall của các loài cỏ**. Giả thuyết (chưa kiểm chứng): do sampler cân bằng lớp làm mô hình ít thiên về lớp áp đảo hơn; phù hợp với việc balanced accuracy tăng (+0,0097) còn top-1 không đổi. Đây cũng là lý do Prickly Acacia có precision thấp nhất (0,913).
- **Cặp khó Chinee Apple ↔ Snake Weed:** Chinee→Snake 15/678 (2,2%) ở F01, 14/678 (2,1%) ở T00; Snake→Chinee 3/612 (0,5%) ở F01, 5/612 (0,8%) ở T00. Theo bài báo (*trích dẫn*, ResNet-50) tỉ lệ này là 3,4% và 4,1%. Ở đây Chinee Apple bị đoán thành `Negatives` (16 lần) cũng nhiều ngang bị đoán thành Snake Weed. Các lỗi giữa các loài khác không đáng kể (Parthenium→Parkinsonia 7, Prickly Acacia→Parkinsonia 6).
- **Lỗi nhất quán qua seed:** F01 có 139 ảnh sai ít nhất ở một seed, trong đó **55 ảnh sai ở cả 3 seed** (T00: 161 và 35). Lỗi lặp lại ở nhiều seed gợi ý ảnh khó hoặc nhãn mơ hồ chứ không phải nhiễu huấn luyện. Ở seed 0, 35/91 ảnh sai có độ tin cậy > 0,9 (độ tin cậy trung bình của ảnh đúng 0,978, của ảnh sai 0,778).

> **[CẦN BỔ SUNG]** Báo cáo này chưa xem trực tiếp ảnh bị đoán sai. Hãy mở `errors_chinee_snake.png` (output notebook Kaggle) và viết 2–3 câu nhận xét thật về hình ảnh (ví dụ ảnh bị đoán sai có bị che khuất, mờ, nền phức tạp, hay nhãn có vẻ sai) để hoàn tất phân tích lỗi theo GUIDE mục 5.5.

---

## 7. Kết luận và khuyến nghị

1. **Cấu hình tốt nhất** (trên val, rồi test 3 seed): ConvNeXt-T + công thức T14 + TTA 10 view + temperature scaling: top-1 0,9739 ± 0,0009, macro-F1 0,9699 ± 0,0014. So với mốc (công thức nền + 1 view) macro-F1 tăng 0,0030, **chưa vượt nhiễu một cách chắc chắn** (p ≈ 0,15, 3 seed); top-1 không đổi; balanced accuracy và ECE cải thiện rõ hơn.
2. **Yếu tố đóng góp nhiều nhất:** (a) *backbone/trọng số*: macro-F1 val từ 0,73 đến 0,96 giữa các backbone cùng công thức, và từ 0,29 (từ đầu) lên 0,96 (tinh chỉnh); (b) *công thức huấn luyện*: tốt nhất +0,010 (CutMix), kết hợp +0,006; (c) *suy luận*: +0,002 đến +0,003 với TTA, với chi phí 2–10×; riêng *temperature scaling* không đổi accuracy nhưng giảm ECE test từ 0,089 xuống 0,006.
3. **Triển khai trên robot (ngân sách 30–100 ms/khung, T4):** chọn **ConvNeXt-T một view + temperature scaling**: p50 5,6 ms, p95 5,9 ms ở batch 1 (FP32), macro-F1 val 0,9689, độ tin cậy được hiệu chuẩn. TTA 10 view chỉ thêm khoảng +0,003 macro-F1 (không phân biệt được với nhiễu) nhưng tốn 10× độ trễ (p95 58 ms), nên không đáng cho thời gian thực. Độ phân giải test 288 là ứng viên đáng thử tiếp (+0,011 trên val, 1,36× độ trễ) nhưng cần kiểm chứng và hiệu chuẩn lại trước khi dùng. Cấu hình ngoại tuyến tốt nhất theo bằng chứng hiện có: F01 (hoặc ensemble nhiều seed).
4. **Việc làm tiếp nếu có thêm thời gian:** (i) chạy nhiều seed (≥ 3) cho các ablation để loại nhiễu; (ii) thêm độ phân giải test và ensemble vào quy trình chọn rồi hiệu chuẩn lại; (iii) tách ảnh hưởng của trọng số và kiến trúc (cùng tag trọng số hoặc LR theo từng họ); (iv) xem kỹ ảnh sai lặp lại ở cả 3 seed; (v) nhiều fold và thử trên ảnh khác điều kiện (ánh sáng, mùa, địa điểm).

---

## 8. Hạn chế và trung thực

1. **Một fold, chia ngẫu nhiên.** Chỉ dùng fold 0; DeepWeeds chia ngẫu nhiên, không theo địa điểm, nên điểm test (0,97) có thể **lạc quan** so với khi gặp địa điểm, mùa hoặc ánh sáng mới. Chưa đo lệch phân phối.
2. **Số seed thấp.** Backbone và ablation chỉ 1 seed. Std val của T00 (0,0006, 3 seed) có thể thấp hơn nhiễu thật (std test 0,0024); vì vậy các Δ dưới ~0,005 không được diễn giải. Welch t-test với n = 3 có công suất thấp.
3. **Quy tắc chọn tự động có điểm yếu** (tất cả chỉ dùng val, nhưng chưa tối ưu):
   - Chung kết dùng T14 theo quy tắc “kết hợp nếu vượt T00 quá ngưỡng”; T05 (CutMix đơn) có macro-F1 val cao hơn T14 (0,9738 so với 0,9694), dù chênh lệch nằm trong nhiễu.
   - Ngưỡng nhiễu của quy tắc là max(2 × std của T00, 0,002) = 0,002 vì std val của T00 quá nhỏ; một số yếu tố được đưa vào T14 (label smoothing +0,0027, sampler +0,0025, LR head +0,0037) thật ra không phân biệt được với nhiễu.
   - Các phương pháp suy luận được quy tắc xem xét chỉ gồm I00, I01, I02a/b, I03. **Độ phân giải test (I04) và ensemble (I05) không được cân nhắc**, dù I04_288 tốt nhất trên val (0,9795). Phương pháp được chọn (10 view) hơn lật ngang (2 view) chỉ 0,0003, trong nhiễu, nhưng đắt gấp 5 lần.
   - Không chạy lại test để so sánh các lựa chọn khác (sẽ vi phạm quy tắc test một lần).
4. **So sánh backbone bị nhiễu bởi trọng số** (mục 3) và công thức nền có thể không hợp với ResNet/ResNeXt/EfficientNet/MobileNet (LR, số epoch cố định cho tất cả); kết quả thấp của các mạng này không nói rằng chúng kém ở mọi công thức.
5. **12 epoch** (bài báo ~100 epoch, augmentation mạnh, 5 fold, 13 giờ/model); so sánh số tuyệt đối với bài báo chỉ mang tính tham khảo.
6. **Đo độ trễ và I5.** Ô đo độ trễ chung kết trong notebook đo **một view** (p95 6,5 ms) vì mã chọn K theo `VIEWS == "hflip"`, trong khi F01 dùng 10 view; con số 6,5 ms đã được đưa vào `eval.py grade` cho tiêu chí I5. Độ trễ thật của suy luận 10 view là p50 54,8 / p95 58,2 / p99 70,6 ms (I02b, đo trên T14 seed 0 ở bước suy luận), vẫn ≤ 100 ms; kết luận I5 không đổi nhưng con số trong `grade` không phải của cấu hình chung kết. Mốc test 1 view có sẵn là T00; **chưa có điểm test cho cấu hình T14 một view** (không chạy thêm test).
7. **Độ trễ chỉ là forward của model**, trên T4, không gồm đọc ảnh, tiền xử lý hay truyền dữ liệu; chưa đo trên phần cứng của robot thật. ECE val sau temperature scaling ở mục 5 là trong mẫu; số ngoài mẫu là ECE test ở mục 6.
8. **Chưa xem ảnh sai trực tiếp** trong báo cáo này (xem hộp “CẦN BỔ SUNG”). Các giả thuyết ở mục 6.3 chưa được kiểm chứng bằng hình ảnh.
9. Gộp BatchNorm không áp dụng cho ConvNeXt; độ chính xác FP16 (I08c) chưa chạy lại bằng `half()`. Log có cảnh báo PyTorch “`lr_scheduler.step()` trước `optimizer.step()`” ở mọi run; khả năng do `GradScaler` (AMP) bỏ qua bước tối ưu đầu tiên khi phát hiện gradient inf, ảnh hưởng chỉ là bỏ qua một giá trị LR đầu (chưa kiểm chứng).
10. Không làm phần điểm thưởng.

---

## 9. Phụ lục

### 9.1 Danh sách thí nghiệm

| Nhóm | exp_id | Nội dung | Seed | Ảnh đường cong |
|---|---|---|---|---|
| Backbone | B01–B07 | T00 với 7 backbone (mục 3) | 0 | `curves/B0x_*.png` |
| Nền | T00 | ConvNeXt-T, công thức nền | 0, 1, 2 | `curves/T00_convnext_tiny*.png` |
| Công thức | T01–T13 | mỗi thí nghiệm khác T00 một yếu tố (mục 4) | 0 | `curves/T0x_*.png`, `T1x_*.png` |
| Kết hợp | T14 | T04 + T05 + T07 + T10 + T11 + T12 | 0 | `curves/T14_convnext_tiny.png` |
| Suy luận | I00–I08 | mục 5, trên val, mô hình T14 seed 0 | — | — |
| Chung kết | F01 | T14 + TTA 10 view + temperature scaling | 0, 1, 2 | `curves/F01_convnext_tiny*.png` |

### 9.2 Cấu hình đầy đủ của F01

`backbone=convnext_tiny` (tag `in12k_ft_in1k`), `init=finetune`, `img_size=224`, `aug=trivial` (RandomResizedCrop(224, scale 0,25–1) + lật ngang + TrivialAugmentWide), `mix=cutmix`, `mix_alpha=1,0`, `loss=ls`, `label_smoothing=0,1`, `sampler=balanced`, `epochs=12`, `batch_size=64`, `lr_backbone=1e-4`, `lr_head=1e-4`, `weight_decay=0,05` (norm/bias: 0), `warmup_epochs=1`, lịch cosine theo bước, `ema_decay=0,999`, `amp=True`, `grad_clip=1,0`, AdamW, checkpoint chọn theo macro-F1 val; suy luận: 5-crop 224 từ ảnh 256 + lật (10 view), gộp xác suất, T khớp trên val của từng seed.

### 9.3 Tái lập

- Notebook Kaggle: xem `README.md`. Mã nguồn: thư mục `code/` (dùng `eval.py` gốc).
- Tính lại chỉ số từ file dự đoán: `python code/eval.py score --pred "predictions/F01_seed*_test.csv" --tag F01` (thêm `--test-csv` và `--labels` nếu có CSV fold 0).
- Số trong báo cáo khớp `results.xlsx` (các sheet `Backbones`, `Training`, `Inference`, `Final`, `PerClass`, `Latency`, `Summary`) và `eval.py`.
