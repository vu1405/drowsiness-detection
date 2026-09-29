import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ============================================================
# CẤU HÌNH
# ============================================================
CSV_PATH = sys.argv[1] if len(sys.argv) > 1 else "features_v2.csv"
WINDOW_SIZE = 300      # số frame mỗi sequence (~10s @ 30fps)
STEP = 150             # overlap 50%
MIN_FILL = 0.8         # window phải có >= 80% frame thật (không bị mất mặt quá nhiều)

TRAIN_SUBJECTS = list(range(1, 13))   # Fold1_part1 + Fold1_part2
TEST_SUBJECTS = list(range(13, 19))   # Fold2_part1

TWO_CLASS_ONLY = True   # True: chỉ Alert(0) vs Drowsy(10). False: cả 3 lớp.

SEQ_FEATURES = [
    "EAR_n", "MAR_z", "blink", "squint", "wide", "lookdown", "brow", "jaw_z",
]

BATCH_SIZE = 32
EPOCHS = 80
SEED = 42
L2_REG = 1e-3   # regularization để chống overfit

np.random.seed(SEED)
tf.random.set_seed(SEED)


# ============================================================
# 1. ĐỌC + CLEAN + CHUẨN HOÁ THEO SUBJECT (giống check_dataset_v3.py)
# ============================================================
df = pd.read_csv(CSV_PATH)
df = df[(df["EAR"] <= 1) & (df["MAR"] <= 1)].dropna().copy()
df = df.sort_values(["Subject", "Label", "FRAME"]).reset_index(drop=True)
print("Sau clean:", df.shape)

g = df.groupby("Subject")
df["EAR_n"] = df["EAR"] / g["EAR"].transform("median")
df["MAR_z"] = (df["MAR"] - g["MAR"].transform("median")) / g["MAR"].transform("std")
df["blink"] = (df["eyeBlinkLeft"] + df["eyeBlinkRight"]) / 2
df["squint"] = (df["eyeSquintLeft"] + df["eyeSquintRight"]) / 2
df["wide"] = (df["eyeWideLeft"] + df["eyeWideRight"]) / 2
df["lookdown"] = (df["eyeLookDownLeft"] + df["eyeLookDownRight"]) / 2
df["brow"] = (df["browDownLeft"] + df["browDownRight"]) / 2
df["jaw_z"] = (df["jawOpen"] - g["jawOpen"].transform("median")) / g["jawOpen"].transform("std")
df = df.fillna(0)

if TWO_CLASS_ONLY:
    df = df[df["Label"].isin([0, 10])].copy()
    LABELS, NAMES = [0, 10], ["Alert", "Drowsy"]
else:
    LABELS, NAMES = [0, 5, 10], ["Alert", "Low Vigilant", "Drowsy"]


# ============================================================
# 2. TẠO SEQUENCE (thay vì thống kê window, giữ nguyên chuỗi frame)
# ============================================================
def build_sequence(w):
    """Trả về mảng (WINDOW_SIZE, n_features), pad/lặp nếu thiếu frame
    do mất mặt vài chỗ (frame bị mất được lấp bằng nội suy tuyến tính)."""
    arr = w[SEQ_FEATURES].to_numpy(dtype=np.float32)
    if len(arr) < WINDOW_SIZE:
        # nội suy để đủ WINDOW_SIZE điểm (co giãn nhẹ theo thời gian)
        old_idx = np.linspace(0, 1, len(arr))
        new_idx = np.linspace(0, 1, WINDOW_SIZE)
        arr = np.stack([
            np.interp(new_idx, old_idx, arr[:, j]) for j in range(arr.shape[1])
        ], axis=1).astype(np.float32)
    elif len(arr) > WINDOW_SIZE:
        arr = arr[:WINDOW_SIZE]
    return arr


X_list, y_list, subj_list = [], [], []
for (subject, label), grp in df.groupby(["Subject", "Label"]):
    fmin, fmax = int(grp["FRAME"].min()), int(grp["FRAME"].max())
    for start in range(fmin, fmax - WINDOW_SIZE + 2, STEP):
        w = grp[(grp["FRAME"] >= start) & (grp["FRAME"] < start + WINDOW_SIZE)]
        if len(w) < MIN_FILL * WINDOW_SIZE:
            continue
        X_list.append(build_sequence(w))
        y_list.append(label)
        subj_list.append(subject)

X = np.stack(X_list)                       # (N, WINDOW_SIZE, n_features)
y_raw = np.array(y_list)
subjects = np.array(subj_list)
label_to_idx = {lab: i for i, lab in enumerate(LABELS)}
y = np.array([label_to_idx[v] for v in y_raw])

print(f"\nSequence dataset: X={X.shape}, y={y.shape}")
print("Số lớp:", {n: int((y_raw == l).sum()) for l, n in zip(LABELS, NAMES)})


# ============================================================
# 3. CHIA TRAIN / TEST THEO FOLD
# ============================================================
train_mask = np.isin(subjects, TRAIN_SUBJECTS)
test_mask = np.isin(subjects, TEST_SUBJECTS)

X_train, y_train = X[train_mask], y[train_mask]
X_test, y_test = X[test_mask], y[test_mask]
print(f"Train: {X_train.shape} | Test: {X_test.shape}")

# Chuẩn hoá feature (z-score) dựa trên TRAIN, áp dụng cho cả 2 tập
mean = X_train.reshape(-1, X_train.shape[-1]).mean(axis=0)
std = X_train.reshape(-1, X_train.shape[-1]).std(axis=0) + 1e-6
X_train = (X_train - mean) / std
X_test = (X_test - mean) / std


# ============================================================
# 4. MODEL: 1D-CNN + LSTM
# ============================================================
n_classes = len(LABELS)
n_features = X.shape[-1]

model = keras.Sequential([
    layers.Input(shape=(WINDOW_SIZE, n_features)),
    layers.Conv1D(16, kernel_size=7, padding="same", activation="relu",
                  kernel_regularizer=keras.regularizers.l2(L2_REG)),
    layers.BatchNormalization(),
    layers.MaxPooling1D(2),
    layers.Dropout(0.3),

    layers.Conv1D(32, kernel_size=5, padding="same", activation="relu",
                  kernel_regularizer=keras.regularizers.l2(L2_REG)),
    layers.BatchNormalization(),
    layers.MaxPooling1D(2),
    layers.Dropout(0.3),

    layers.Bidirectional(layers.LSTM(
        24, return_sequences=False,
        kernel_regularizer=keras.regularizers.l2(L2_REG),
        recurrent_dropout=0.0,   # recurrent_dropout làm chậm CPU rất nhiều, dùng Dropout layer thay thế
    )),
    layers.Dropout(0.5),
    layers.Dense(16, activation="relu",
                kernel_regularizer=keras.regularizers.l2(L2_REG)),
    layers.Dropout(0.4),
    layers.Dense(1 if n_classes == 2 else n_classes,
                 activation="sigmoid" if n_classes == 2 else "softmax"),
])

loss = "binary_crossentropy" if n_classes == 2 else "sparse_categorical_crossentropy"
model.compile(optimizer=keras.optimizers.Adam(1e-3), loss=loss, metrics=["accuracy"])
model.summary()

y_train_fit = y_train.astype(np.float32) if n_classes == 2 else y_train
y_test_fit = y_test.astype(np.float32) if n_classes == 2 else y_test

callbacks = [
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=15,
                                  restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=6),
]

history = model.fit(
    X_train, y_train_fit,
    validation_data=(X_test, y_test_fit),
    epochs=EPOCHS, batch_size=BATCH_SIZE,
    callbacks=callbacks, verbose=2,
)


# ============================================================
# 5. ĐÁNH GIÁ
# ============================================================
if n_classes == 2:
    proba = model.predict(X_test).ravel()
    pred = (proba > 0.5).astype(int)
else:
    proba = model.predict(X_test)
    pred = proba.argmax(axis=1)

acc = accuracy_score(y_test, pred)
print(f"\n>>> Test accuracy (fold-split, LSTM/CNN): {acc:.4f}")
print(classification_report(y_test, pred, target_names=NAMES))

cm = confusion_matrix(y_test, pred)
print(cm)
ConfusionMatrixDisplay(cm, display_labels=NAMES).plot()
plt.title(f"Sequence model (CNN+LSTM) - {'2 lớp' if n_classes == 2 else '3 lớp'}")
plt.savefig("confusion_matrix_lstm.png", dpi=150, bbox_inches="tight")
plt.close()

# Curve huấn luyện
plt.figure()
plt.plot(history.history["accuracy"], label="train acc")
plt.plot(history.history["val_accuracy"], label="test acc")
plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend()
plt.title("Training curve")
plt.savefig("training_curve_lstm.png", dpi=150, bbox_inches="tight")
plt.close()

model.save("drowsiness_lstm_model.keras")
print("\nĐã lưu model: drowsiness_lstm_model.keras")
print("Đã lưu: confusion_matrix_lstm.png, training_curve_lstm.png")
