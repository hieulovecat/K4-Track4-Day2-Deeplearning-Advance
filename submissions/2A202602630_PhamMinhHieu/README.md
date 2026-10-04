# Lab Day 2 — DeepWeeds: backbone, công thức huấn luyện, suy luận

**Sinh viên:** Phạm Minh Hiếu · **MSSV:** 2A202602630

Bài nộp cho Lab Day 2 (Track 4): so sánh 7 backbone, 14 thí nghiệm công thức huấn luyện và 13 phương pháp suy luận trên DeepWeeds (fold 0), rồi chạy chung kết 3 seed.
Kết quả chính (test, 3 seed, mean ± std): **top-1 0,9739 ± 0,0009, macro-F1 0,9699 ± 0,0014** (mốc công thức nền + 1 view: top-1 0,9737 ± 0,0030, macro-F1 0,9669 ± 0,0024). Chi tiết, phân tích và hạn chế trong [`report.md`](report.md).

## Link chạy lại

- **Notebook Kaggle:** <https://www.kaggle.com/code/mycatis/notebook80259dbf2a> *(kiểm tra lại link của version đã chạy xong và đặt notebook ở chế độ chia sẻ để người chấm xem được)*
- **Mã nguồn (GitHub):** <https://github.com/hieulovecat/K4-Track4-Day2-Deeplearning-Advance> (thư mục `code/`)

## Cấu trúc bài nộp

```
├── README.md              # file này
├── report.md              # báo cáo kết luận
├── results.xlsx           # Backbones, Training, Inference, Final, PerClass, Latency, Summary
├── curves/                # 27 ảnh đường cong training, mỗi exp_id (và seed) một ảnh
├── predictions/           # F01 và T00: test 3 seed; F01_uncal: test chưa temperature scaling; *_val.csv của mọi run
├── figures/               # ma trận nhầm lẫn, đánh đổi backbone/suy luận, Δ ablation (tạo từ results.xlsx và predictions/)
└── code/                  # train.py, model.py, dataset.py, losses.py, inference.py, benchmark.py, checks.py,
                           # experiments.py, lab_day2.ipynb, tests/test_code.py, eval.py (bản gốc, không sửa)
```

Tên file trong `curves/`: `B0x_<backbone>.png`, `T00_..._seed1.png`, `T01…T14_convnext_tiny.png`, `F01_convnext_tiny[_seed1|_seed2].png`.

## Môi trường đã dùng

| Mục | Giá trị |
|---|---|
| Nền tảng | Kaggle Notebook, GPU Tesla T4 (2 GPU được gắn, code dùng 1) |
| Python / torch / timm | 3.13.15 / 2.11.0+cu128 / 1.0.29 |
| Thư viện khác | `pandas`, `numpy`, `matplotlib` (có sẵn trên Kaggle), `openpyxl` và `timm` (cài trong notebook); `eval.py` chỉ cần `numpy` và `pandas`. Chưa ghi phiên bản của các thư viện này |
| Seed | backbone và ablation: 0; T00 và F01: 0, 1, 2 |
| Thời gian | cả pipeline khoảng 17.860 s ≈ 5 giờ trên T4 |

Tag trọng số `timm` đã dùng: resnet50 `a1_in1k`, resnext50_32x4d `a1h_in1k`, convnext_tiny `in12k_ft_in1k`, deit_small_patch16_224 `fb_in1k`, swin_tiny_patch4_window7_224 `ms_in1k`, efficientnet_b0 `ra_in1k`, mobilenetv3_large_100 `ra_in1k`.

## Cách chạy lại

1. Trên Kaggle: tạo notebook từ `code/lab_day2.ipynb` (File → Import Notebook).
2. Settings: Accelerator = GPU T4, **Internet = On** (để `git clone`, cài `timm`, tải ảnh).
3. Ô 0: `PLATFORM = "kaggle"`, `USE_GIT = True`, `IMAGES_SOURCE = "download"`, `RUN_UP_TO = 5`, `EPOCHS = 12`.
4. Save Version → **Save & Run All (Commit)**. Notebook chạy nền từ Bước 0 đến Bước 5:
   - Bước 0: tải ảnh từ Zenodo (kiểm MD5 `b7b30f96d466fba86016aa5a26606e0f`) và CSV fold 0 của tác giả, kiểm tra chia dữ liệu, EDA, kiểm tra pipeline.
   - Bước 1: 7 backbone (B01–B07), chọn backbone có macro-F1 val cao nhất.
   - Bước 2: T00 (3 seed) và T01–T13, rồi T14 kết hợp; chọn cấu hình đi vào chung kết.
   - Bước 3: nghiên cứu suy luận I00–I08 trên val và đo độ trễ.
   - Bước 4: F01 (3 seed) và mốc T00, **test chạy đúng một lần mỗi seed**, rồi `eval.py score` và `eval.py grade`.
   - Bước 5: `results.xlsx` và gói bài nộp.
5. Mọi lựa chọn (backbone, công thức, suy luận) được tự động chọn **chỉ dựa trên val**; quy tắc nằm trong `code/experiments.py` (`select_backbone`, `select_recipe`, `choose_final`, `select_inference`) và được mô tả trong `report.md` (mục 8).

Chạy trên Colab cũng được (`PLATFORM = "colab"`); xem `code/README.md`. Kết quả chạy lại có thể lệch nhẹ do GPU và thư viện (cudnn deterministic được bật nhưng không đảm bảo bit-exact giữa các phiên bản).

## Kiểm tra số liệu

Tính lại chỉ số từ `predictions/` bằng `eval.py` gốc (có CSV fold 0 thì thêm `--test-csv` và `--labels`):

```bash
python code/eval.py score --pred "predictions/F01_seed*_test.csv" --tag F01
python code/eval.py score --pred "predictions/T00_seed*_test.csv" --tag T00
python code/eval.py grade --final "predictions/F01_seed*_test.csv" --baseline "predictions/T00_seed*_test.csv" \
    --uncal "predictions/F01_uncal_seed*_test.csv" --final-val "predictions/F01_seed*_val.csv"
```

Test của `code/`: `PYTHONUTF8=1 python -m unittest discover -s code/tests -v` (chạy trên CPU, không cần dữ liệu thật).

## Lưu ý quan trọng

- **Test chạy một lần mỗi seed.** `train.run` và `experiments.predict_final` không ghi đè file test đã tồn tại; log ghi “test chạy 1 lần cho seed k”.
- **Một số hạn chế đã biết** (chi tiết ở `report.md`, mục 8): một fold chia ngẫu nhiên; ablation 1 seed; độ phân giải test 288 tốt nhất trên val nhưng không nằm trong quy tắc chọn tự động; ô đo độ trễ chung kết đo 1 view thay vì 10 view (độ trễ thật 10 view: p95 58,2 ms); chưa xem ảnh sai trực tiếp.
- Không commit dataset hay checkpoint `.pt`; `results.xlsx`, `curves/`, `predictions/` và `figures/` đều nhỏ.
