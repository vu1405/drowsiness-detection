import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import LeaveOneGroupOut, RandomizedSearchCV, GroupKFold
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)

# ============================================================
# CẤU HÌNH
# ============================================================
CSV_PATH = sys.argv[1] if len(sys.argv) > 1 else "features_v2.csv"
WINDOW_SIZE = 300
STEP = 150
MIN_FILL = 0.8
BLINK_CLOSED = 0.5
EAR_CLOSED_RATIO = 0.75
JAW_Z = 2.0
LABELS, NAMES = [0, 10], ["Alert", "Drowsy"]   # chỉ 2 lớp

TRAIN_SUBJECTS = list(range(1, 13))
TEST_SUBJECTS = list(range(13, 19))

SEED = 42


# ============================================================
# 1. ĐỌC + CLEAN + CHUẨN HOÁ (giống v3)
# ============================================================
df = pd.read_csv(CSV_PATH)
df = df[(df["EAR"] <= 1) & (df["MAR"] <= 1)].dropna().copy()
df = df.sort_values(["Subject", "Label", "FRAME"]).reset_index(drop=True)
df = df[df["Label"].isin([0, 10])].copy()   # chỉ 2 lớp ngay từ đầu
print("Sau clean (2 lớp):", df.shape)

g = df.groupby("Subject")


def wrap180(a):
    return (a + 180) % 360 - 180


df["EAR_n"] = df["EAR"] / g["EAR"].transform("median")
df["MAR_z"] = (df["MAR"] - g["MAR"].transform("median")) / g["MAR"].transform("std")
df["blink"] = (df["eyeBlinkLeft"] + df["eyeBlinkRight"]) / 2
df["squint"] = (df["eyeSquintLeft"] + df["eyeSquintRight"]) / 2
df["wide"] = (df["eyeWideLeft"] + df["eyeWideRight"]) / 2
df["lookdown"] = (df["eyeLookDownLeft"] + df["eyeLookDownRight"]) / 2
df["brow"] = (df["browDownLeft"] + df["browDownRight"]) / 2
df["jaw_z"] = (df["jawOpen"] - g["jawOpen"].transform("median")) / g["jawOpen"].transform("std")
for c in ["Pitch", "Yaw", "Roll"]:
    df[c + "_rel"] = wrap180(df[c] - g[c].transform("median"))


# ============================================================
# 2. FEATURE (bộ cũ + THÊM feature xu hướng trong window)
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


def slope(y):
    """Độ dốc tuyến tính của y theo thời gian trong window."""
    x = np.arange(len(y))
    if len(y) < 2 or np.std(y) == 0:
        return 0.0
    return np.polyfit(x, y, 1)[0]


def half_delta(y):
    """Nửa sau trừ nửa đầu -> dương nghĩa là tăng dần trong window."""
    mid = len(y) // 2
    if mid == 0:
        return 0.0
    return y[mid:].mean() - y[:mid].mean()


def window_features(w):
    ear, mar = w["EAR_n"].to_numpy(), w["MAR_z"].to_numpy()
    blink, jaw = w["blink"].to_numpy(), w["jaw_z"].to_numpy()
    squint, wide = w["squint"].to_numpy(), w["wide"].to_numpy()
    brow = w["brow"].to_numpy()
    ear_closed = ear < EAR_CLOSED_RATIO
    blink_closed = blink > BLINK_CLOSED

    f = {
        # --- bộ cũ ---
        "EAR_mean": ear.mean(), "EAR_std": ear.std(), "EAR_min": ear.min(),
        "EAR_max": ear.max(), "EAR_vel": np.abs(np.diff(ear)).mean(),
        "MAR_mean": mar.mean(), "MAR_std": mar.std(), "MAR_max": mar.max(),
        **closure_feats(ear_closed, "EAR"),
        "BS_blink_mean": blink.mean(), "BS_blink_std": blink.std(),
        "BS_blink_max": blink.max(), "BS_blink_p90": np.percentile(blink, 90),
        "BS_blink_vel": np.abs(np.diff(blink)).mean(),
        **closure_feats(blink_closed, "BS"),
        "BS_squint_mean": squint.mean(), "BS_wide_mean": wide.mean(),
        "BS_lookdown_mean": w["lookdown"].mean(), "BS_brow_mean": brow.mean(),
        "BS_jaw_mean": jaw.mean(), "BS_jaw_std": jaw.std(), "BS_jaw_max": jaw.max(),
        "BS_yawn_ratio": (jaw > JAW_Z).mean(),
        "P_pitch_mean": w["Pitch_rel"].mean(), "P_pitch_std": w["Pitch_rel"].std(),
        "P_yaw_std": w["Yaw_rel"].std(), "P_roll_std": w["Roll_rel"].std(),
        "P_pitch_vel": np.abs(np.diff(w["Pitch_rel"].to_numpy())).mean(),

        # --- MỚI: feature xu hướng (trend) trong window ---
        "EAR_slope": slope(ear),
        "EAR_half_delta": half_delta(ear),
        "BS_blink_slope": slope(blink),
        "BS_blink_half_delta": half_delta(blink),
        "BS_squint_slope": slope(squint),
        "BS_brow_slope": slope(brow),
        "EAR_PERCLOS_half_delta": half_delta(ear_closed.astype(float)),
        "EAR_p25": np.percentile(ear, 25),
        "EAR_p75": np.percentile(ear, 75),
        "EAR_iqr": np.percentile(ear, 75) - np.percentile(ear, 25),
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
print(f"\nWindow dataset: {wdf.shape} | {len(FEATURES)} feature")


# ============================================================
# 3. TINH CHỈNH HYPERPARAMETER (chỉ trên tập TRAIN, dùng GroupKFold
#    theo Subject để không rò rỉ dữ liệu giữa các "fold" nội bộ)
# ============================================================
train = wdf[wdf["Subject"].isin(TRAIN_SUBJECTS)]
test = wdf[wdf["Subject"].isin(TEST_SUBJECTS)]
X_train, y_train = train[FEATURES], train["Label"]
X_test, y_test = test[FEATURES], test["Label"]
groups_train = train["Subject"]

print(f"Train: {X_train.shape} | Test: {X_test.shape}")

cv = GroupKFold(n_splits=4)

print("\n--- Tìm hyperparameter tốt nhất cho Random Forest ---")
rf_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=SEED, n_jobs=-1, class_weight="balanced"),
    param_distributions={
        "n_estimators": [200, 400, 600],
        "max_depth": [None, 8, 12, 16, 20],
        "min_samples_leaf": [1, 2, 3, 5, 8],
        "max_features": ["sqrt", "log2", 0.5, 0.7],
    },
    n_iter=25, cv=cv, scoring="accuracy", random_state=SEED, n_jobs=-1,
)
rf_search.fit(X_train, y_train, groups=groups_train)
print("Best RF params:", rf_search.best_params_)
print("Best RF CV accuracy (nội bộ trên train):", round(rf_search.best_score_, 4))

print("\n--- Tìm hyperparameter tốt nhất cho Gradient Boosting ---")
gb_search = RandomizedSearchCV(
    HistGradientBoostingClassifier(random_state=SEED),
    param_distributions={
        "max_iter": [150, 250, 400],
        "max_depth": [3, 5, 7, None],
        "learning_rate": [0.03, 0.06, 0.1],
        "l2_regularization": [0.0, 0.1, 0.5, 1.0],
        "min_samples_leaf": [10, 20, 30],
    },
    n_iter=25, cv=cv, scoring="accuracy", random_state=SEED, n_jobs=-1,
)
gb_search.fit(X_train, y_train, groups=groups_train)
print("Best GB params:", gb_search.best_params_)
print("Best GB CV accuracy (nội bộ trên train):", round(gb_search.best_score_, 4))


# ============================================================
# 4. ĐÁNH GIÁ TRÊN TEST THẬT (Subject 13-18, chưa từng thấy)
# ============================================================
def evaluate(model, tag, png):
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    print(f"\n===== {tag} =====")
    print(f">>> Test accuracy (fold-split): {acc:.4f}")
    print(classification_report(y_test, pred, labels=LABELS, target_names=NAMES))
    cm = confusion_matrix(y_test, pred, labels=LABELS)
    print(cm)
    ConfusionMatrixDisplay(cm, display_labels=NAMES).plot()
    plt.title(tag)
    plt.savefig(png, dpi=150, bbox_inches="tight")
    plt.close()
    if hasattr(model, "feature_importances_"):
        imp = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
        print("\nTop feature importance:\n", imp.head(15).round(3))
    return acc, model


acc_rf, best_rf = evaluate(rf_search.best_estimator_, "Random Forest (đã tinh chỉnh + feature mới)",
                           "confusion_matrix_rf_tuned.png")
acc_gb, best_gb = evaluate(gb_search.best_estimator_, "Gradient Boosting (đã tinh chỉnh + feature mới)",
                           "confusion_matrix_gb_tuned.png")


# ============================================================
# 5. LOSO CHO MODEL TỐT NHẤT (để kiểm tra độ tin cậy, không chỉ 1 lần)
# ============================================================
best_name, best_params, best_is_rf = (
    ("Random Forest", rf_search.best_params_, True) if acc_rf >= acc_gb
    else ("Gradient Boosting", gb_search.best_params_, False)
)
print(f"\n\n=== Model tốt hơn trên test: {best_name} (acc={max(acc_rf, acc_gb):.4f}) ===")
print("Giờ chạy LOSO (18 subject) với model này để kiểm tra độ ổn định...")

X_all, y_all, groups_all = wdf[FEATURES], wdf["Label"], wdf["Subject"]
y_true, y_pred = [], []
for tr, te in LeaveOneGroupOut().split(X_all, y_all, groups_all):
    if best_is_rf:
        m = RandomForestClassifier(random_state=SEED, n_jobs=-1, class_weight="balanced", **best_params)
    else:
        m = HistGradientBoostingClassifier(random_state=SEED, **best_params)
    m.fit(X_all.iloc[tr], y_all.iloc[tr])
    p = m.predict(X_all.iloc[te])
    subj = groups_all.iloc[te].iloc[0]
    print(f"  Test subject {subj}: acc = {accuracy_score(y_all.iloc[te], p):.3f}")
    y_true += list(y_all.iloc[te])
    y_pred += list(p)

loso_acc = accuracy_score(y_true, y_pred)
print(f"\n>>> LOSO accuracy ({best_name}, đã tinh chỉnh + feature mới): {loso_acc:.4f}")
print(classification_report(y_true, y_pred, labels=LABELS, target_names=NAMES))

print("\n========== TỔNG KẾT ==========")
print(f"  Fold-split - Random Forest (tuned):       {acc_rf:.4f}")
print(f"  Fold-split - Gradient Boosting (tuned):    {acc_gb:.4f}")
print(f"  LOSO       - {best_name} (tuned):    {loso_acc:.4f}")
print(f"  (so sánh)  - RF gốc, feature cũ, LOSO:      0.6558 (kết quả trước đó)")
