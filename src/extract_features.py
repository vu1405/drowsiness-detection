import os
import math
import csv
import cv2
import numpy as np
import mediapipe as mp

# ============================================================
# CẤU HÌNH
# ============================================================
# Liệt kê TẤT CẢ các folder fold đã tải, chạy 1 lần là đủ cho tất cả.
# Có thể chạy lại nhiều lần (ví dụ mỗi lần tải thêm 1 fold): video nào
# đã xử lý xong rồi sẽ tự bị bỏ qua.
DATASET_FOLDERS = ["Fold1_part1", "Fold1_part2", "Fold2_part1"]
MODEL_PATH = "face_landmarker.task"
OUTPUT_CSV = "features_v2.csv"
DONE_LOG = "features_v2_done.txt"  # danh sách video đã xử lý xong, để resume

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_faces=1,
    output_face_blendshapes=True,                # <-- MỚI
    output_facial_transformation_matrixes=True,  # <-- MỚI
)

# Blendshape cần lưu (tên chuẩn của MediaPipe)
BLENDSHAPES = [
    "eyeBlinkLeft", "eyeBlinkRight",
    "eyeSquintLeft", "eyeSquintRight",
    "eyeWideLeft", "eyeWideRight",
    "eyeLookDownLeft", "eyeLookDownRight",
    "browDownLeft", "browDownRight",
    "jawOpen", "mouthClose",
]

LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
MOUTH = [61, 82, 13, 312, 291, 317, 14, 87]


# ============================================================
# TÌM VIDEO (quét nhiều fold, chuẩn hoá Subject ID để không trùng)
# ============================================================
video_files = []
seen_subjects = {}  # subject_id (int) -> (fold, folder_name) đầu tiên gặp, để cảnh báo trùng

for dataset_folder in DATASET_FOLDERS:
    if not os.path.isdir(dataset_folder):
        print(f"CẢNH BÁO: không tìm thấy folder '{dataset_folder}', bỏ qua.")
        continue

    for subject_name in os.listdir(dataset_folder):
        subject_path = os.path.join(dataset_folder, subject_name)
        if not os.path.isdir(subject_path):
            continue

        try:
            subject_id = int(subject_name)  # "07" -> 7, tự bỏ số 0 ở đầu
        except ValueError:
            print(f"  Bỏ qua folder không phải số subject: {subject_path}")
            continue

        if subject_id in seen_subjects and seen_subjects[subject_id] != (dataset_folder, subject_name):
            other = seen_subjects[subject_id]
            print(f"CẢNH BÁO: Subject ID {subject_id} bị trùng giữa "
                  f"{other[0]}/{other[1]} và {dataset_folder}/{subject_name}. "
                  f"Dữ liệu 2 người sẽ bị gộp nhầm! Hãy đổi tên 1 trong 2 folder.")
        seen_subjects.setdefault(subject_id, (dataset_folder, subject_name))

        for filename in os.listdir(subject_path):
            if filename.lower().endswith((".mov", ".mp4", ".avi")):
                label = int(os.path.splitext(filename)[0])
                video_files.append((subject_id, label, os.path.join(subject_path, filename)))

video_files.sort(key=lambda x: (x[0], x[1]))
print("Tổng số video tìm thấy:", len(video_files))

# ============================================================
# RESUME: bỏ qua video đã xử lý xong ở lần chạy trước
# ============================================================
done_keys = set()
if os.path.exists(DONE_LOG):
    with open(DONE_LOG) as f:
        done_keys = set(line.strip() for line in f if line.strip())

remaining = [v for v in video_files if v[2] not in done_keys]
print(f"Đã xử lý trước đó: {len(video_files) - len(remaining)} | Còn lại: {len(remaining)}")
video_files = remaining

file_exists = os.path.exists(OUTPUT_CSV)


# ============================================================
# HÀM TÍNH TOÁN
# ============================================================
def pts(face_landmarks, indices, w, h):
    # Giữ float (không int()) để không mất độ chính xác
    return [(face_landmarks[i].x * w, face_landmarks[i].y * h) for i in indices]


def dist(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def calc_ear(e):
    h = dist(e[0], e[3])
    if h == 0:
        return 0.0
    return (dist(e[1], e[5]) + dist(e[2], e[4])) / (2 * h)


def calc_mar(m):
    h = dist(m[0], m[4])
    if h == 0:
        return 0.0
    return (dist(m[1], m[7]) + dist(m[2], m[6]) + dist(m[3], m[5])) / (3 * h)


def head_pose_from_matrix(M):
    """Euler angles (độ) từ facial_transformation_matrix 4x4 của MediaPipe.
    Ma trận này đã ở toạ độ camera nên mặt nhìn thẳng ~ (0,0,0),
    không bị lật ±180 như solvePnP + RQDecomp3x3 cũ."""
    R = np.array(M)[:3, :3].astype(np.float64)
    # bỏ scale nếu có
    R = R / np.linalg.norm(R, axis=0, keepdims=True)
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy > 1e-6:
        pitch = math.atan2(R[2, 1], R[2, 2])
        yaw = math.atan2(-R[2, 0], sy)
        roll = math.atan2(R[1, 0], R[0, 0])
    else:
        pitch = math.atan2(-R[1, 2], R[1, 1])
        yaw = math.atan2(-R[2, 0], sy)
        roll = 0.0
    return math.degrees(pitch), math.degrees(yaw), math.degrees(roll)


# ============================================================
# XỬ LÝ
# ============================================================
header = ["Subject", "Label", "FRAME", "EAR", "MAR", "Pitch", "Yaw", "Roll"] + BLENDSHAPES

if not video_files:
    print("Không có video mới nào cần xử lý. Xong.")
    raise SystemExit

with open(OUTPUT_CSV, "a", newline="") as f, \
        open(DONE_LOG, "a") as done_f, \
        FaceLandmarker.create_from_options(options) as landmarker:

    writer = csv.writer(f)
    if not file_exists:
        writer.writerow(header)
    global_ts = 0  # timestamp phải tăng dần qua các video xử lý TRONG LẦN CHẠY NÀY

    for subject, label, video_path in video_files:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30
        step_ms = max(1, round(1000 / fps))
        frame_no = 0
        detected = 0
        print(f"\nSubject {subject} | Label {label} | {video_path} | fps={fps:.1f}")

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_no += 1
            global_ts += step_ms

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(mp_image, global_ts)

            if frame_no % 500 == 0:
                print(f"  {frame_no} frames, detect {detected}/{frame_no}")

            if not result.face_landmarks:
                continue  # FRAME vẫn tăng -> khoảng trống được nhận ra ở bước window

            h, w = frame.shape[:2]
            lm = result.face_landmarks[0]

            ear = (calc_ear(pts(lm, LEFT_EYE, w, h)) + calc_ear(pts(lm, RIGHT_EYE, w, h))) / 2
            mar = calc_mar(pts(lm, MOUTH, w, h))

            if result.facial_transformation_matrixes:
                pitch, yaw, roll = head_pose_from_matrix(result.facial_transformation_matrixes[0])
            else:
                pitch = yaw = roll = float("nan")

            bs = {b.category_name: b.score for b in result.face_blendshapes[0]} \
                if result.face_blendshapes else {}
            bs_values = [bs.get(name, float("nan")) for name in BLENDSHAPES]

            writer.writerow([subject, label, frame_no, ear, mar, pitch, yaw, roll] + bs_values)
            detected += 1

        cap.release()
        f.flush()
        done_f.write(video_path + "\n")
        done_f.flush()
        print(f"  Xong: {detected}/{frame_no} frame có mặt")

print("\nĐã lưu:", OUTPUT_CSV)
