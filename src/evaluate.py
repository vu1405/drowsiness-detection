import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut
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
BLINK_CLOSED = 0.5     # eyeBlink blendshape > 0.5 => mắt nhắm
EAR_CLOSED_RATIO = 0.75
JAW_Z = 2.0
LABELS = [0, 5, 10]
NAMES = ["Alert", "Low Vigilant", "Drowsy"]


# ============================================================
# 1. ĐỌC + CLEAN
# ============================================================
df = pd.read_csv(CSV_PATH)
df = df[(df["EAR"] <= 1) & (df["MAR"] <= 1)].dropna().copy()
df = df.sort_values(["Subject", "Label", "FRAME"]).reset_index(drop=True)
print("Sau clean:", df.shape)


# ============================================================
# 2. CHUẨN HOÁ THEO SUBJECT
# ============================================================
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
# 3. FEATURE
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
        # nhóm EAR/MAR (tự tính)
        "EAR_mean": ear.mean(), "EAR_std": ear.std(), "EAR_min": ear.min(),
        "EAR_max": ear.max(), "EAR_vel": np.abs(np.diff(ear)).mean(),
        "MAR_mean": mar.mean(), "MAR_std": mar.std(), "MAR_max": mar.max(),
        **closure_feats(ear < EAR_CLOSED_RATIO, "EAR"),
        # nhóm blendshape
        "BS_blink_mean": blink.mean(), "BS_blink_std": blink.std(),
        "BS_blink_max": blink.max(), "BS_blink_p90": np.percentile(blink, 90),
        "BS_blink_vel": np.abs(np.diff(blink)).mean(),
        **closure_feats(blink > BLINK_CLOSED, "BS"),
        "BS_squint_mean": w["squint"].mean(), "BS_wide_mean": w["wide"].mean(),
        "BS_lookdown_mean": w["lookdown"].mean(), "BS_brow_mean": w["brow"].mean(),
        "BS_jaw_mean": jaw.mean(), "BS_jaw_std": jaw.std(), "BS_jaw_max": jaw.max(),
        "BS_yawn_ratio": (jaw > JAW_Z).mean(),
        # head pose tương đối
        "P_pitch_mean": w["Pitch_rel"].mean(), "P_pitch_std": w["Pitch_rel"].std(),
        "P_yaw_std": w["Yaw_rel"].std(), "P_roll_std": w["Roll_rel"].std(),
        "P_pitch_vel": np.abs(np.diff(w["Pitch_rel"].to_numpy())).mean(),
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
print("\nWindow dataset:", wdf.shape)
print(wdf.groupby(["Subject", "Label"]).size().unstack(fill_value=0))


# ============================================================
# 4. LOSO
# ============================================================
EAR_FEATS = [c for c in wdf.columns if c.startswith(("EAR_", "MAR_"))]
BS_FEATS = [c for c in wdf.columns if c.startswith("BS_")]
POSE_FEATS = [c for c in wdf.columns if c.startswith("P_")]


def make_model():
    return RandomForestClassifier(n_estimators=300, min_samples_leaf=3,
                                  random_state=42, n_jobs=-1)


def loso(features, tag, plot=False, data=None, labels=LABELS, names=NAMES, png="confusion_matrix_v3.png"):
    data = wdf if data is None else data
    X, y, groups = data[features], data["Label"], data["Subject"]
    y_true, y_pred = [], []
    print(f"\n===== {tag} ({len(features)} features) =====")
    for tr, te in LeaveOneGroupOut().split(X, y, groups):
        p = make_model().fit(X.iloc[tr], y.iloc[tr]).predict(X.iloc[te])
        print(f"  Test subject {groups.iloc[te].iloc[0]}: acc = {accuracy_score(y.iloc[te], p):.3f}")
        y_true += list(y.iloc[te])
        y_pred += list(p)
    acc = accuracy_score(y_true, y_pred)
    print(f"  >>> LOSO accuracy: {acc:.4f}")
    if plot:
        print(classification_report(y_true, y_pred, labels=labels, target_names=names))
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        print(cm)
        ConfusionMatrixDisplay(cm, display_labels=names).plot()
        plt.title(f"LOSO - {tag}")
        plt.savefig(png, dpi=150, bbox_inches="tight")
        plt.close()
        imp = pd.Series(make_model().fit(X, y).feature_importances_,
                        index=features).sort_values(ascending=False)
        print("\nTop feature importance:\n", imp.head(12).round(3))
    return acc


results = {
    "EAR/MAR": loso(EAR_FEATS, "EAR/MAR"),
    "Blendshape": loso(BS_FEATS, "Blendshape"),
    "Blendshape + EAR/MAR": loso(BS_FEATS + EAR_FEATS, "Blendshape + EAR/MAR"),
}
best_tag = max(results, key=results.get)
results["Tất cả + pose"] = loso(BS_FEATS + EAR_FEATS + POSE_FEATS, "Tất cả + head pose")

print("\n========== TỔNG KẾT 3 LỚP (Alert / Low Vigilant / Drowsy) ==========")
for k, v in results.items():
    print(f"  {k:<24}{v:.4f}")


# ============================================================
# 5. CHẾ ĐỘ 2 LỚP: Alert (0) vs Drowsy (10), bỏ lớp 5
# ============================================================
wdf2 = wdf[wdf["Label"].isin([0, 10])].copy()
L2, N2 = [0, 10], ["Alert", "Drowsy"]
print(f"\n\n########## CHẾ ĐỘ 2 LỚP ({len(wdf2)} window, đoán mò = 50%) ##########")

results2 = {
    "EAR/MAR": loso(EAR_FEATS, "2 lớp - EAR/MAR", data=wdf2, labels=L2, names=N2),
    "Blendshape": loso(BS_FEATS, "2 lớp - Blendshape", data=wdf2, labels=L2, names=N2),
    "Blendshape + EAR/MAR": loso(BS_FEATS + EAR_FEATS, "2 lớp - Blendshape + EAR/MAR",
                                 data=wdf2, labels=L2, names=N2),
}
results2["Tất cả + pose"] = loso(BS_FEATS + EAR_FEATS + POSE_FEATS, "2 lớp - Tất cả + pose",
                                 plot=True, data=wdf2, labels=L2, names=N2,
                                 png="confusion_matrix_2class.png")

print("\n========== TỔNG KẾT 2 LỚP (Alert / Drowsy) ==========")
for k, v in results2.items():
    print(f"  {k:<24}{v:.4f}")


# ============================================================
# 6. ĐÁNH GIÁ THEO FOLD: train Fold1 (subject 1-12), test Fold2 (subject 13-18)
# ============================================================
# Sửa lại 2 danh sách này nếu bạn tải thêm fold hoặc đặt Subject ID khác.
TRAIN_SUBJECTS = list(range(1, 13))   # Fold1_part1 (1-6) + Fold1_part2 (7-12)
TEST_SUBJECTS = list(range(13, 19))   # Fold2_part1 (13-18)


def fold_split_eval(features, tag, data, labels, names, png):
    train = data[data["Subject"].isin(TRAIN_SUBJECTS)]
    test = data[data["Subject"].isin(TEST_SUBJECTS)]
    print(f"\n===== {tag} ({len(features)} features) =====")
    print(f"  Train: {len(train)} window (subject {sorted(train['Subject'].unique())})")
    print(f"  Test:  {len(test)} window (subject {sorted(test['Subject'].unique())})")
    if len(train) == 0 or len(test) == 0:
        print("  Bỏ qua: thiếu dữ liệu train hoặc test.")
        return None

    model = make_model().fit(train[features], train["Label"])
    pred = model.predict(test[features])
    acc = accuracy_score(test["Label"], pred)
    print(f"  >>> Fold-split accuracy: {acc:.4f}")
    print(classification_report(test["Label"], pred, labels=labels, target_names=names))
    cm = confusion_matrix(test["Label"], pred, labels=labels)
    print(cm)
    ConfusionMatrixDisplay(cm, display_labels=names).plot()
    plt.title(f"Fold-split (train 1-12, test 13-18) - {tag}")
    plt.savefig(png, dpi=150, bbox_inches="tight")
    plt.close()
    imp = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    print("\nTop feature importance:\n", imp.head(12).round(3))
    return acc


print("\n\n========================================================")
print("ĐÁNH GIÁ THEO FOLD: train Fold1 (subject 1-12), test Fold2 (subject 13-18)")
print("========================================================")

print("\n########## 3 LỚP (Alert / Low Vigilant / Drowsy) ##########")
fold_results = {
    "EAR/MAR": fold_split_eval(EAR_FEATS, "3 lớp - EAR/MAR", wdf, LABELS, NAMES,
                               "confusion_matrix_fold_3class_earmar.png"),
    "Blendshape": fold_split_eval(BS_FEATS, "3 lớp - Blendshape", wdf, LABELS, NAMES,
                                  "confusion_matrix_fold_3class_bs.png"),
    "Blendshape + EAR/MAR": fold_split_eval(BS_FEATS + EAR_FEATS, "3 lớp - Blendshape + EAR/MAR",
                                            wdf, LABELS, NAMES,
                                            "confusion_matrix_fold_3class_bsear.png"),
    "Tất cả + pose": fold_split_eval(BS_FEATS + EAR_FEATS + POSE_FEATS, "3 lớp - Tất cả + pose",
                                     wdf, LABELS, NAMES,
                                     "confusion_matrix_fold_3class_all.png"),
}

print("\n########## 2 LỚP (Alert / Drowsy) ##########")
fold_results2 = {
    "EAR/MAR": fold_split_eval(EAR_FEATS, "2 lớp - EAR/MAR", wdf2, L2, N2,
                               "confusion_matrix_fold_2class_earmar.png"),
    "Blendshape": fold_split_eval(BS_FEATS, "2 lớp - Blendshape", wdf2, L2, N2,
                                  "confusion_matrix_fold_2class_bs.png"),
    "Blendshape + EAR/MAR": fold_split_eval(BS_FEATS + EAR_FEATS, "2 lớp - Blendshape + EAR/MAR",
                                            wdf2, L2, N2,
                                            "confusion_matrix_fold_2class_bsear.png"),
    "Tất cả + pose": fold_split_eval(BS_FEATS + EAR_FEATS + POSE_FEATS, "2 lớp - Tất cả + pose",
                                     wdf2, L2, N2,
                                     "confusion_matrix_fold_2class_all.png"),
}

print("\n========== TỔNG KẾT FOLD-SPLIT 3 LỚP (train 1-12, test 13-18) ==========")
for k, v in fold_results.items():
    print(f"  {k:<24}{v:.4f}" if v is not None else f"  {k:<24}(bỏ qua)")

print("\n========== TỔNG KẾT FOLD-SPLIT 2 LỚP (train 1-12, test 13-18) ==========")
for k, v in fold_results2.items():
    print(f"  {k:<24}{v:.4f}" if v is not None else f"  {k:<24}(bỏ qua)")
