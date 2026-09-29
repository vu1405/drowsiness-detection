"""
Chạy 1 LẦN DUY NHẤT trước khi chạy main_v2.py lần đầu với nhiều fold mới.

Việc này đánh dấu các video trong DATASET_FOLDERS mà Subject ID của nó
đã CÓ SẴN trong features_v2.csv là "đã xử lý xong", để main_v2.py không
xử lý lại và ghi trùng dữ liệu.

Sau khi chạy xong file này, chạy main_v2.py như bình thường.
"""
import os
import pandas as pd

OUTPUT_CSV = "features_v2.csv"
DONE_LOG = "features_v2_done.txt"

# Phải giống hệt danh sách folder trong main_v2.py
DATASET_FOLDERS = ["Fold1_part1", "Fold1_part2", "Fold2_part1"]

if not os.path.exists(OUTPUT_CSV):
    raise SystemExit(f"Không thấy {OUTPUT_CSV}, không cần chạy file này.")

existing_subjects = set(pd.read_csv(OUTPUT_CSV, usecols=["Subject"])["Subject"].unique().tolist())
print("Subject đã có sẵn trong CSV:", sorted(existing_subjects))

already_done = set()
if os.path.exists(DONE_LOG):
    with open(DONE_LOG) as f:
        already_done = set(line.strip() for line in f if line.strip())

marked = 0
with open(DONE_LOG, "a") as done_f:
    for dataset_folder in DATASET_FOLDERS:
        if not os.path.isdir(dataset_folder):
            continue
        for subject_name in os.listdir(dataset_folder):
            subject_path = os.path.join(dataset_folder, subject_name)
            if not os.path.isdir(subject_path):
                continue
            try:
                subject_id = int(subject_name)
            except ValueError:
                continue
            if subject_id not in existing_subjects:
                continue  # subject này CHƯA có trong CSV -> để main_v2.py xử lý bình thường

            for filename in os.listdir(subject_path):
                if filename.lower().endswith((".mov", ".mp4", ".avi")):
                    video_path = os.path.join(subject_path, filename)
                    if video_path not in already_done:
                        done_f.write(video_path + "\n")
                        marked += 1

print(f"Đã đánh dấu {marked} video (thuộc {len(existing_subjects)} subject có sẵn) là ĐÃ XONG.")
print(f"Giờ chạy: python main_v2.py")
