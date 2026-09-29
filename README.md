# Drowsiness Detection — Phát hiện buồn ngủ bằng đặc trưng khuôn mặt

Hệ thống phát hiện buồn ngủ của người lái xe/người dùng từ video khuôn mặt, dùng
[MediaPipe Face Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker)
để trích xuất đặc trưng (EAR, MAR, blendshapes, head pose), sau đó phân loại bằng
Random Forest. Dữ liệu: [UTA-RLDD](https://sites.google.com/view/utarldd/home)
(Real-Life Drowsiness Dataset, Ghoddoosian et al., CVPRW 2019).

## Kết quả

Đánh giá **cross-subject** (LOSO — Leave-One-Subject-Out, và fold-split train/test
theo nhóm subject riêng biệt), trên 18/60 subject của UTA-RLDD:

| Bài toán | Phương pháp | LOSO | Fold-split (train 1-12, test 13-18) |
|---|---|---|---|
| 2 lớp (Alert / Drowsy) | Random Forest, EAR/MAR + Blendshape | **65.6%** | 65.6% |
| 3 lớp (Alert / Low Vigilant / Drowsy) | Random Forest, EAR/MAR + Blendshape | ~40-45% | ~38-45% |

Baseline gốc do nhóm tạo ra UTA-RLDD công bố (5-fold cross-subject, 3 lớp) là
**65.2%** — kết quả của project này tương đồng với baseline đó, cho thấy pipeline
hoạt động đúng như kỳ vọng đối với dataset vốn nổi tiếng là khó (nhãn tự báo cáo,
biểu hiện buồn ngủ kín đáo, đánh giá cross-subject).

Các hướng đã thử để cải thiện — kết quả cụ thể ghi trong code/log — không vượt qua
được mốc ~65.6% trên bộ 18 subject: tinh chỉnh hyperparameter, thêm đặc trưng xu
hướng (trend/slope), mô hình chuỗi thời gian CNN+LSTM. Nút thắt hiện tại nằm ở
**số lượng subject**, không phải ở model hay bộ đặc trưng.

## Cấu trúc project

```
drowsiness-detection/
├── src/
│   ├── extract_features.py     # Video -> features_v2.csv (chạy đầu tiên)
│   ├── evaluate.py             # features_v2.csv -> đánh giá LOSO + fold-split
│   └── train_final_model.py    # Train model cuối trên toàn bộ dữ liệu, lưu .pkl
├── experiments/
│   ├── tune_random_forest.py   # Thử tinh chỉnh hyperparameter + feature mới
│   └── train_lstm_cnn.py       # Thử mô hình chuỗi thời gian CNN+LSTM
├── utils/
│   └── mark_existing_done.py   # Tiện ích resume khi thêm fold mới vào CSV đã có
├── requirements.txt
└── README.md
```

## Cài đặt

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Tải model MediaPipe Face Landmarker (`face_landmarker.task`) từ
[trang MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker/index#models),
đặt cùng thư mục với `src/extract_features.py`.

## Dữ liệu

Tải [UTA-RLDD](https://sites.google.com/view/utarldd/home) (Real-Life Drowsiness
Dataset) tại link Google Drive sau:

**[https://drive.google.com/drive/folders/1d_QwgpMXnLY_FmLYXDn7TLcw0D-svXEl](https://drive.google.com/drive/folders/1d_QwgpMXnLY_FmLYXDn7TLcw0D-svXEl)**

Dataset chia thành nhiều fold, mỗi fold có các thư mục con là số thứ tự subject
(`1`, `07`, `13`...), bên trong là video đặt tên theo nhãn (`0.mp4` = Alert,
`5.mp4` = Low Vigilant, `10.mp4` = Drowsy):

```
Fold1_part1/
├── 1/
│   ├── 0.mp4
│   ├── 5.mp4
│   └── 10.mp4
├── 2/
│   └── ...
Fold1_part2/
├── 07/
│   └── ...
```

Đặt các thư mục fold cùng cấp với `src/`, rồi cập nhật danh sách
`DATASET_FOLDERS` ở đầu `src/extract_features.py` cho khớp với các fold bạn đã
tải (mặc định đang để `Fold1_part1`, `Fold1_part2`, `Fold2_part1`).

## Cách chạy (theo đúng thứ tự)

### 1. Trích xuất đặc trưng từ video

```bash
python src/extract_features.py
```

Đọc toàn bộ video trong `DATASET_FOLDERS`, chạy MediaPipe Face Landmarker để lấy
EAR, MAR, head pose (từ facial transformation matrix, không dùng solvePnP để tránh
lỗi lật góc ±180°) và 12 blendshapes (blink, squint, jawOpen...) cho từng frame có
phát hiện được mặt. Ghi ra `features_v2.csv`.

**Chạy tiếp được (resume):** mỗi video xử lý xong được ghi vào
`features_v2_done.txt`; nếu bị ngắt giữa chừng, chạy lại sẽ tự bỏ qua video đã
xong. Nếu tải thêm fold mới sau này, chỉ cần thêm tên fold vào
`DATASET_FOLDERS` rồi chạy lại lệnh trên.

> Nếu bạn đã từng chạy 1 phiên bản script cũ hơn (không có cơ chế resume) và đã có
> sẵn `features_v2.csv`, chạy `python utils/mark_existing_done.py` một lần trước,
> để tránh xử lý trùng lại dữ liệu cũ.

### 2. Đánh giá mô hình

```bash
python src/evaluate.py
```

Đọc `features_v2.csv`, chuẩn hoá đặc trưng theo từng subject (median/std riêng của
từng người — quan trọng vì EAR/MAR khác nhau tự nhiên giữa mỗi người), cắt thành
các window 300 frame (~10s, overlap 50%), tính đặc trưng thống kê (PERCLOS, số lần
chớp mắt, độ dốc...), rồi train/đánh giá Random Forest theo 2 cách:

- **LOSO:** lần lượt để 1 subject ra làm test, train bằng số còn lại, lặp lại cho
  hết — cho biết khả năng tổng quát tới người hoàn toàn mới.
- **Fold-split:** train cố định trên 1 nhóm subject, test trên nhóm còn lại —
  mô phỏng kịch bản "train xong rồi đưa vào dùng thực tế".

In ra classification report, confusion matrix (lưu ảnh `.png`), và feature
importance cho cả 2 chế độ 2 lớp và 3 lớp.

### 3. Train model cuối để dùng thực tế

```bash
python src/train_final_model.py
```

Train Random Forest trên **toàn bộ** dữ liệu hiện có (không chia test — việc đo độ
chính xác đã làm ở bước 2), lưu ra:

- `drowsiness_model_final.pkl` — model, nạp lại bằng `joblib.load()`.
- `drowsiness_model_config.json` — danh sách đặc trưng theo đúng thứ tự cần dùng
  khi dự đoán, cùng các ngưỡng đã dùng lúc train.

**Lưu ý khi triển khai cho người dùng mới:** model cần chuẩn hoá EAR/MAR/jawOpen
theo median của chính người đang được quay. Với người hoàn toàn mới, cần ước
lượng baseline này từ vài chục giây đầu video (lúc họ còn tỉnh táo) trước khi đưa
vào model, nếu không độ chính xác thực tế sẽ thấp hơn con số báo cáo ở bước 2.

## Các thử nghiệm khác (experiments/)

- **`tune_random_forest.py`** — tìm hyperparameter tốt nhất (RandomizedSearchCV +
  GroupKFold) cho cả Random Forest và Gradient Boosting, cộng thêm đặc trưng xu
  hướng (slope, so sánh nửa đầu/nửa sau window). Kết quả: không vượt qua được bản
  gốc đơn giản — dấu hiệu overfit vào quá trình tìm hyperparameter khi chỉ có 12
  subject để train nội bộ.
- **`train_lstm_cnn.py`** — mô hình chuỗi thời gian (Conv1D + BiLSTM) học trực
  tiếp trên chuỗi frame thay vì đặc trưng thống kê window. Cần cài thêm
  `tensorflow`. Kết quả: tốt hơn Random Forest ở 1 lần fold-split (~68-73%) nhưng
  overfit rất nhanh (val loss tăng ngay từ epoch 2), không ổn định — nhiều khả
  năng cần nhiều subject hơn mới phát huy được lợi thế của deep learning.

## Hạn chế đã biết

- Chỉ 18/60 subject của UTA-RLDD được dùng — số lượng subject là nút thắt chính
  của độ chính xác hiện tại, không phải model hay đặc trưng.
- Nhãn 3 mức (Alert/Low Vigilant/Drowsy) do người tham gia **tự báo cáo** (KSS),
  mang tính chủ quan; ranh giới Alert–Low Vigilant đặc biệt khó phân biệt (accuracy
  3 lớp chỉ ~40%, gần mức ngẫu nhiên 33%).
- Head pose được thử nghiệm nhưng không đóng góp giá trị dự đoán trong mọi bài
  toán — không dùng trong model cuối.
- Đánh giá cross-subject (LOSO/fold-split) cho kết quả thấp hơn nhiều so với các
  con số 90-99% thường thấy ở một số paper khác dùng UTA-RLDD hoặc dataset khác
  (NTHU-DDD, YawDD) — phần lớn chênh lệch đến từ cách đánh giá (within-subject
  hoặc random split theo frame dễ bị rò rỉ dữ liệu) và từ đặc điểm dataset (hành
  vi buồn ngủ trong UTA-RLDD kín đáo hơn nhiều so với dataset yêu cầu diễn).

## Hướng phát triển

- Tải thêm fold của UTA-RLDD (hiện dùng 18/60 subject).
- Thử CNN học trực tiếp trên ảnh khuôn mặt thay vì đặc trưng thủ công.
- Hiệu chỉnh ngưỡng quyết định (`predict_proba` thay vì `predict`) để ưu tiên
  recall của lớp Drowsy — quan trọng hơn accuracy tổng trong ứng dụng cảnh báo an
  toàn.

## Trích dẫn dataset

```
Ghoddoosian, R., Galib, M., & Athitsos, V. (2019). A Realistic Dataset and
Baseline Temporal Model for Early Drowsiness Detection. In Proceedings of the
IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops
(CVPRW).
```