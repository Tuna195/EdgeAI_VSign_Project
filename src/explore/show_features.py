"""Đi từng bước của make_features trên 1 video. 
Chạy: python -m src.explore.show_features"""
from src.core.features import make_features, points, sample_positions, shoulder_frame
from src.dataset.build_sequences import load_sequences

KEY, ROW = "test/Bố/003266", 15                      # video và dòng (trong 30 dòng) muốn xem
cfg = {"norm": "shoulder", "use_z": True, "n_frames": 30}

seq = load_sequences()[KEY]
center, width = shoulder_frame(points(seq))
idx = sample_positions(len(seq), cfg["n_frames"])
feat = make_features(seq, idx, cfg)

print(KEY, "| chuỗi", seq.shape, "-> đặc trưng", feat.shape)
print("Tâm vai:", center.round(3), "| độ rộng vai:", round(width, 3))
print("Frame được chọn:", idx)

frame = idx[ROW]                                     # dòng ROW của đặc trưng ứng với frame này của chuỗi
wrist = points(seq)[frame, 21, :2]                   # cổ tay phải (điểm 21): x, y gốc
by_hand = (wrist - center) / width                   # tự tính theo công thức
print(f"Dòng {ROW} = frame {frame}: cổ tay phải gốc {wrist.round(3)}")
print(f"  tự tính: {by_hand.round(3)} | make_features: {feat[ROW, 63:65].round(3)} | cờ trái, phải: {feat[ROW, 159:161]}")