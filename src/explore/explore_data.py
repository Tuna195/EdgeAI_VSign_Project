from pathlib import Path
import numpy as np

RAW = Path("data/raw")

# 1. Đếm video và từ
files = sorted(RAW.glob("*/*/*.npy"))
print("Số lượng video:", len(files))
words =  sorted({f.parent.name for f in files})
print("Số lượng từ:", len(words))

# 2. Mở 1 video
f = files[0]
a = np.load(f)
print("Shape của video:", a.shape, "dtype:", a.dtype)

# 3. Tách 3 phần
left, right, pose = a[:, :63], a[:, 63:126], a[:, 126:]
print("Shape của phần trái:", left.shape)
print("Shape của phần phải:", right.shape)
print("Shape của phần pose:", pose.shape)

# 4. Dạng điểm (T, 75, 3)
x = a.reshape(-1, 75, 3)
print("Shape của dạng điểm:", x.shape)
print("Mũi (Pose 0) ở frame 0:", x[0, 42 + 0])
print("Vai trái (Pose 11) ở frame 0:", x[0, 42 + 11])

# 5. Mỗi frame thấy gì
seen_left = x[:, :21].any(axis=(1, 2))
seen_right = x[:, 21:42].any(axis=(1, 2))
seen_pose = x[:, 42:].any(axis=(1, 2))
print(f"Trong {len(x)} frame, có {seen_left.sum()} frame thấy tay trái, {seen_right.sum()} frame thấy tay phải, {seen_pose.sum()} frame thấy pose.")

#BT1
print("Số lượng video train:", len(list(RAW.glob("train/*/*.npy"))))
print("Số lượng video test:", len(list(RAW.glob("test/*/*.npy"))))

#BT2
right_ngon_tro = x[len(x)//2, 29, 0 : 2]
print("Tọa độ x, y của ngón trỏ tay phải ở frame giữa:", right_ngon_tro)

#BT3
left = x[:, :21]
print("Check frame không có tay trái thì tất cả các điểm tay trái đều bằng 0:", left[~seen_left].sum() == 0)

#BT4
print("% frame thấy tay:", len(x[seen_left | seen_right]) / len(x) * 100)
print(len(x))
print(len([1 for a, b in zip(seen_left, seen_right) if (a != b) or (a and b)]))

#BT4
max_frame = -1
min_frame = float("inf")
max_frame_video = ""
min_frame_video = ""
for f in files:
    frame_count = np.load(f).shape[0]

    if frame_count > max_frame:
        max_frame = frame_count
        max_frame_video = str(f)

    if frame_count < min_frame:
        min_frame = frame_count
        min_frame_video = str(f)

print("Video nhiều frame nhất:", max_frame_video, max_frame)
print("Video ít frame nhất:", min_frame_video, min_frame)
