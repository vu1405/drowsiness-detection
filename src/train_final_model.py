"""
BẢN CUỐI: train Random Forest trên TOÀN BỘ 18 subject (không chia train/test
nữa, vì việc đánh giá độ chính xác đã làm xong ở check_dataset_v3.py /
tune_rf.py — kết quả tham khảo: ~65.6% accuracy, LOSO, 2 lớp Alert/Drowsy).

Mục đích file này: tạo ra 1 model THẬT SỰ DÙNG ĐƯỢC cho demo/ứng dụng,
bằng cách train trên hết dữ liệu đang có (nhiều dữ liệu train hơn luôn tốt
hơn cho model cuối, dù không dùng để đo accuracy nữa).

Output:
  - drowsiness_model_final.pkl   (model đã train, dùng joblib.load để nạp lại)
  - drowsiness_model_config.json (danh sách feature + cấu hình cần thiết
    để tiền xử lý dữ liệu MỚI giống hệt lúc train)
"""
import json
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier

# ============================================================
# CẤU HÌNH (giống hệt bản cho kết quả tốt nhất: check_dataset_v3.py)
# ============================================================
CSV_PATH = "features_v2.csv"
WINDOW_SIZE = 300
STEP = 150
MIN_FILL = 0.8
BLINK_CLOSED = 0.5
EAR_CLOSED_RATIO = 0.75
JAW_Z = 2.0
LABELS, NAMES = [0, 10], ["Alert", "Drowsy"]   # 2 lớp: kết quả tốt & ổn định nhất

MODEL_PATH = "drowsiness_model_final.pkl"
CONFIG_PATH = "drowsiness_model_config.json"


# ============================================================
# 1. ĐỌC + CLEAN + CHUẨN HOÁ THEO SUBJECT
# ============================================================
df = pd.read_csv(CSV_PATH)
df = df[(df["EAR"] <= 1) & (df["MAR"] <= 1)].dropna().copy()
df = df.sort_values(["Subject", "Label", "FRAME"]).reset_index(drop=True)
df = df[df["Label"].isin([0, 10])].copy()
print("Sau clean (2 lớp):", df.shape)

g = df.groupby("Subject")
df["EAR_n"] = df["EAR"] / g["EAR"].transform("median")
df["MAR_z"] = (df["MAR"] - g["MAR"].transform("median")) / g["MAR"].transform("std")
df["blink"] = (df["eyeBlinkLeft"] + df["eyeBlinkRight"]) / 2
df["squint"] = (df["eyeSquintLeft"] + df["eyeSquintRight"]) / 2
df["wide"] = (df["eyeWideLeft"] + df["eyeWideRight"]) / 2
df["lookdown"] = (df["eyeLookDownLeft"] + df["eyeLookDownRight"]) / 2
df["brow"] = (df["browDownLeft"] + df["browDownRight"]) / 2
df["jaw_z"] = (df["jawOpen"] - g["jawOpen"].transform("median")) / g["jawOpen"].transform("std")


# ============================================================
# 2. FEATURE (bộ EAR/MAR + Blendshape, KHÔNG dùng head pose
#    vì mọi thử nghiệm trước đều cho thấy pose không có giá trị)
# ============================================================
def run_lengths(mask):
    runs, n = [], 0
    for v in mask:
        if v:
            n += 1
        elif n:
            runs.append(n)
            n = 0
    if n:
        runs.append(n)
    return runs


def closure_feats(closed, prefix):
    runs = run_lengths(closed)
    return {
        f"{prefix}_PERCLOS": closed.mean(),
        f"{prefix}_blinks": len([r for r in runs if 2 <= r <= 15]),
        f"{prefix}_long": len([r for r in runs if r > 15]),
        f"{prefix}_max_run": max(runs) if runs else 0,
        f"{prefix}_mean_run": np.mean(runs) if runs else 0,
    }


def window_features(w):
    ear, mar = w["EAR_n"].to_numpy(), w["MAR_z"].to_numpy()
    blink, jaw = w["blink"].to_numpy(), w["jaw_z"].to_numpy()
    f = {
        "EAR_mean": ear.mean(), "EAR_std": ear.std(), "EAR_min": ear.min(),
        "EAR_max": ear.max(), "EAR_vel": np.abs(np.diff(ear)).mean(),
        "MAR_mean": mar.mean(), "MAR_std": mar.std(), "MAR_max": mar.max(),
        **closure_feats(ear < EAR_CLOSED_RATIO, "EAR"),
        "BS_blink_mean": blink.mean(), "BS_blink_std": blink.std(),
        "BS_blink_max": blink.max(), "BS_blink_p90": np.percentile(blink, 90),
        "BS_blink_vel": np.abs(np.diff(blink)).mean(),
        **closure_feats(blink > BLINK_CLOSED, "BS"),
        "BS_squint_mean": w["squint"].mean(), "BS_wide_mean": w["wide"].mean(),
        "BS_lookdown_mean": w["lookdown"].mean(), "BS_brow_mean": w["brow"].mean(),
        "BS_jaw_mean": jaw.mean(), "BS_jaw_std": jaw.std(), "BS_jaw_max": jaw.max(),
        "BS_yawn_ratio": (jaw > JAW_Z).mean(),
    }
    return f


rows = []
for (subject, label), grp in df.groupby(["Subject", "Label"]):
    fmin, fmax = int(grp["FRAME"].min()), int(grp["FRAME"].max())
    for start in range(fmin, fmax - WINDOW_SIZE + 2, STEP):
        w = grp[(grp["FRAME"] >= start) & (grp["FRAME"] < start + WINDOW_SIZE)]
        if len(w) < MIN_FILL * WINDOW_SIZE:
            continue
        f = window_features(w)
        f["Subject"], f["Label"] = subject, label
        rows.append(f)

wdf = pd.DataFrame(rows).fillna(0)
FEATURES = [c for c in wdf.columns if c not in ("Subject", "Label")]
print(f"Window dataset: {wdf.shape} | {len(FEATURES)} feature")
print("Số lớp:", wdf["Label"].value_counts().to_dict())


# ============================================================
# 3. TRAIN TRÊN TOÀN BỘ 18 SUBJECT
# ============================================================
X, y = wdf[FEATURES], wdf["Label"]

model = RandomForestClassifier(
    n_estimators=300, min_samples_leaf=3,
    random_state=42, n_jobs=-1,
)
model.fit(X, y)
print("\nĐã train xong trên toàn bộ", len(X), "window.")

imp = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
print("\nTop feature importance:\n", imp.head(10).round(3))


# ============================================================
# 4. LƯU MODEL + CONFIG (để dùng lại khi predict trên video mới)
# ============================================================
joblib.dump(model, MODEL_PATH)

config = {
    "features_in_order": FEATURES,   # THỨ TỰ cột bắt buộc phải giữ đúng khi predict
    "window_size": WINDOW_SIZE,
    "ear_closed_ratio": EAR_CLOSED_RATIO,
    "blink_closed": BLINK_CLOSED,
    "jaw_z": JAW_Z,
    "labels": LABELS,
    "label_names": NAMES,
    "note": (
        "Model 2 lớp (Alert=0, Drowsy=10), train trên 18 subject, "
        "accuracy tham khảo ~65.6% (LOSO, xem check_dataset_v3.py). "
        "QUAN TRỌNG: EAR_n/MAR_z/jaw_z được chuẩn hoá theo MEDIAN/STD của "
        "chính subject đó (xem code). Khi dùng cho người MỚI, cần ước lượng "
        "median EAR/MAR/jawOpen của người đó từ vài giây đầu video (lúc họ "
        "tỉnh táo) để chuẩn hoá tương tự, nếu không model sẽ kém chính xác "
        "hơn con số 65.6% ở trên."
    ),
}
with open(CONFIG_PATH, "w", encoding="utf-8") as f:
    json.dump(config, f, ensure_ascii=False, indent=2)

print(f"\nĐã lưu model: {MODEL_PATH}")
print(f"Đã lưu config: {CONFIG_PATH}")
print("\nCách nạp lại để dùng:")
print("  import joblib, json")
print(f"  model = joblib.load('{MODEL_PATH}')")
print(f"  config = json.load(open('{CONFIG_PATH}'))")
print("  # X_new: DataFrame có đúng các cột trong config['features_in_order']")
print("  pred = model.predict(X_new[config['features_in_order']])")
print("  proba = model.predict_proba(X_new[config['features_in_order']])  # xác suất, dùng để chỉnh ngưỡng cảnh báo")
