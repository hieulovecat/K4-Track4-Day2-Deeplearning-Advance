# code/ — bản hoàn thiện của `starter/`

| File | Nội dung |
|---|---|
| `dataset.py` | đọc fold 0, `check_split` (S1–S4), transform + augmentation (`basic/flipv/color/trivial/randaug`), `DataLoader` (sampler cân bằng) |
| `model.py` | backbone qua `timm`, đóng băng, 3 nhóm tham số (norm/bias không weight decay), đếm params/GMAC |
| `losses.py` | CE, label smoothing, focal, CE có trọng số lớp, Mixup/CutMix |
| `train.py` | `run(Config)` dùng chung: AMP, warmup + cosine, EMA, chọn checkpoint theo macro-F1 val, vẽ đường cong, chạy lại được khi Colab ngắt |
| `inference.py` | TTA (lật, 5-crop, đa tỉ lệ), gộp xác suất/logit, ensemble, temperature scaling, gộp Conv-BN |
| `benchmark.py` | độ trễ p50/p95/p99 (warmup, `synchronize`, ≥ 50 lần) |
| `checks.py` | loss ban đầu ≈ ln 9, overfit 1 batch, xem ảnh sau augmentation |
| `experiments.py` | danh sách B/T/I/F, nghiên cứu suy luận, ghi predictions chung kết, tạo `results.xlsx` |
| `lab_day2.ipynb` | **notebook chạy trên Colab hoặc Kaggle** (đổi `PLATFORM` ở ô 0) |
| `tests/test_code.py` | 26 test chạy trên CPU (không cần mạng/dữ liệu) |

`eval.py` nằm ở thư mục gốc repo, **không sửa**.

## Chạy trên Google Colab (đặt `PLATFORM = "colab"`)

1. Đưa code lên GitHub (từ máy bạn): `git add code && git commit -m "Hoàn thiện code" && git push`.
2. Mở `code/lab_day2.ipynb` trên Colab (File → Open notebook → GitHub, hoặc upload file). Chọn **Runtime → Change runtime type → T4 GPU**.
3. Chạy lần lượt từ trên xuống. Ô "Lấy code" cần `REPO_URL`; ô dữ liệu tự tải `images.zip` (kiểm MD5) và CSV fold 0.
   Nếu đã có `images.zip` trên Drive: `IMAGES_SOURCE = "drive"`.
4. Các điểm **bạn phải tự quyết trên val** (trong notebook có ghi `<- ĐỔI`): `BEST_BACKBONE` (sau Bước 1), cấu hình kết hợp `COMBO`, `INF_SRC`, `FINAL_KW`, `VIEWS/AGG/TS` (sau Bước 2–3).
5. Kết quả lưu ở `MyDrive/lab_day2/` (`runs/`, `predictions/`, `curves/`, `results.xlsx`). Nếu Colab ngắt, mở lại và chạy lại các ô: run nào xong sẽ được bỏ qua.

## Chạy trên Kaggle (đặt `PLATFORM = "kaggle"`)

1. Tải `code/lab_day2.ipynb` về máy (từ GitHub, nút Download raw file). Trên kaggle.com: **Create → New Notebook → File → Import Notebook** → chọn file.
2. Panel phải **Settings**: Accelerator = **GPU T4 x2** (code chỉ dùng 1 GPU; tránh P100 vì không có tensor core cho AMP),
   **Internet = On** (cần xác minh số điện thoại cho tài khoản; cần để `git clone`, `pip`, tải ảnh).
3. Ô 0: `PLATFORM = "kaggle"`; để `IMAGES_SOURCE = "download"` (Zenodo) hoặc dùng dataset riêng (`"input"`, xem notebook).
4. Chạy lần lượt như trên Colab. Kết quả ở `/kaggle/working/lab_day2/`.
5. **Giữ kết quả:** `/kaggle/working` chỉ được lưu khi bạn **Save Version** (Save & Run All / Quick Save). Phiên tối đa 12 giờ,
   hạn mức GPU khoảng 30 giờ/tuần (xem trong tài khoản). Làm từng bước rồi Save Version sau mỗi bước.
6. **Chạy tiếp ở phiên sau:** notebook → Add Input → Your Work/Notebook Output → chọn notebook cũ; đặt
   `PREV_OUTPUT_DIR = "/kaggle/input/<tên-notebook-cũ>"`. Ô 0.2 sẽ chép `runs/`, `predictions/`, `curves/` về, run nào xong sẽ được bỏ qua.

Không có git? Nén cả thư mục repo thành `lab.zip`, tải lên `MyDrive/lab_day2/lab.zip`, đặt `USE_GIT = False`.

## Chạy từ dòng lệnh

```bash
cd code
python train.py --set exp_id=B01 backbone=resnet50 seed=0 images_dir=../data/images labels_dir=../data/labels
python -m unittest discover -s tests -v      # test (đặt PYTHONUTF8=1 trên Windows)
```

## Quy tắc đã được cài sẵn trong code

- `run()` không tính hay in chỉ số test; test chỉ được ghi khi `save_test_predictions=True` và **không ghi đè** file test đã có (S4).
- Val loss luôn là CE thường, nên so sánh được giữa các loss khác nhau.
- Chung kết `run_final` huấn luyện rồi suy luận một lần; T khớp trên val của chính seed đó, rồi áp sang test.
- Chỉ số tính bằng `eval.compute_metrics`, cùng định nghĩa với lúc chấm.
