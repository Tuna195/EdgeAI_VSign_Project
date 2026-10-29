"""Tiền xử lý bản 2: DÙNG CHUNG cho train và nhận diện trực tiếp (cùng hàm -> cùng kết quả).

Chỉ cần numpy (không cần mediapipe/cv2) -> chạy được trên Colab.

Hai giai đoạn:
  prepare_sequence(raw, w, h)   chạy 1 lần mỗi video (build_sequences.py lưu kết quả vào sequences.npz):
      [1] sửa tỉ lệ khung hình: x *= W/H   [2] fill_gaps   [3] bỏ frame trống ở 2 đầu
      -> chuỗi (T, 225) độ dài thay đổi, CHƯA chuẩn hoá
  make_features(seq, idx, cfg)  chạy mỗi lần lấy mẫu (sau augmentation và chọn N frame):
      chuẩn hoá theo cfg["norm"], z bật/tắt, giữ 11 điểm Pose, 2 cờ -> (N, D)

cfg: {"norm": "none" | "shoulder" | "shoulder_wrist", "use_z": bool, "n_frames": int}
  none            toạ độ MediaPipe (đã sửa tỉ lệ), model tự học "tay so với người"
  shoulder        gốc = giữa 2 vai, đơn vị = độ rộng vai (trung vị cả chuỗi): "tay ở đâu so với người"
  shoulder_wrist  như shoulder + thêm bản tay lấy cổ tay làm gốc, đơn vị = cỡ lòng bàn tay: "bàn tay hình gì"
"""
import numpy as np

from src.core.preprocess import fill_gaps

NUM_POINTS = 75                 # 21 tay trái + 21 tay phải + 33 thân
LEFT, RIGHT, BODY = slice(0, 21), slice(21, 42), slice(42, 75)
L_SH, R_SH = 11, 12             # vai trái, vai phải (chỉ số trong 33 điểm Pose)
# 11 điểm Pose giữ lại: mũi, 2 mắt, 2 khoé miệng, 2 vai, 2 khuỷu, 2 cổ tay.
# Bỏ thân dưới (23-32: kiosk quay đến ngang eo) và ngón tay của Pose (17-22: trùng Hand).
POSE_KEEP = [0, 2, 5, 9, 10, 11, 12, 13, 14, 15, 16]
WRIST, MIDDLE_MCP = 0, 9        # Hand: cổ tay, gốc ngón giữa -> cỡ lòng bàn tay
NORMS = ("none", "shoulder", "shoulder_wrist")

MIN_FRAMES = 10                 # chuỗi ngắn hơn -> bỏ video
MIN_SHOULDER = 0.05             # độ rộng vai (đơn vị = chiều cao ảnh) nhỏ hơn -> MediaPipe đặt sai vai, bỏ video
MIN_PALM = 0.02                 # cỡ lòng bàn tay nhỏ hơn thì kẹp lại (tránh chia cho số quá nhỏ)


def fix_aspect(raw, width, height):
    """MediaPipe chia x cho chiều rộng ảnh, y cho chiều cao -> ảnh không vuông thì hình bị méo.
    Nhân x (và z, cùng thang với x) với W/H -> mọi toạ độ cùng đơn vị "chiều cao ảnh". Điểm = 0 vẫn = 0."""
    pts = raw.reshape(len(raw), NUM_POINTS, 3).copy()
    pts[:, :, [0, 2]] *= width / height
    return pts.reshape(len(raw), -1)


def shoulder_frame(pts, body_ok):
    """(tâm (2,), độ rộng vai) = trung vị trên các frame thấy thân, như normalize() cũ.
    Dùng 1 giá trị cho cả chuỗi: 1 frame MediaPipe đặt sai vai không làm hỏng frame đó."""
    sh = pts[body_ok][:, [42 + L_SH, 42 + R_SH], :2]
    center = np.median(sh.mean(axis=1), axis=0)
    width = float(np.median(np.linalg.norm(sh[:, 0] - sh[:, 1], axis=1)))
    return center, width


def prepare_sequence(raw, width, height):
    """raw (T, 225) từ MediaPipe -> (chuỗi (T', 225) float32, None) hoặc (None, lý do bỏ video)."""
    if len(raw) == 0:
        return None, "video rỗng"
    if not np.isfinite(raw).all():
        return None, "có NaN/vô cực"
    seq = fill_gaps(fix_aspect(raw, width, height))
    # Bỏ frame trống (cả 225 số = 0: không thấy tay, không thấy thân) liền mạch ở 2 đầu; frame trống ở giữa giữ nguyên
    used = np.flatnonzero(seq.any(axis=1))
    if len(used) == 0:
        return None, "mọi frame đều trống"
    seq = seq[used[0]:used[-1] + 1]
    pts = seq.reshape(len(seq), NUM_POINTS, 3)
    body_ok = pts[:, BODY].any(axis=(1, 2))
    if not body_ok.any():
        return None, "không thấy thân ở frame nào"
    if len(seq) < MIN_FRAMES:
        return None, f"ngắn hơn {MIN_FRAMES} frame"
    if shoulder_frame(pts, body_ok)[1] < MIN_SHOULDER:
        return None, "độ rộng vai quá nhỏ"
    return seq.astype(np.float32), None


def sample_positions(length, n, rng=None, speed=(0.8, 1.25)):
    """Chỉ số n frame trong chuỗi dài length. Luôn giữ frame đầu và frame cuối.

    rng = None: cách đều (val, test, nhận diện trực tiếp).
    Có rng (train): đổi tốc độ ký. Chọn 1 điểm gãy m trong đoạn 30-70%, điểm đó dời đi một chút ->
    nửa đầu ký nhanh hơn, nửa sau chậm hơn (hoặc ngược lại), tốc độ mỗi nửa thay đổi trong khoảng speed.
    Không xoá frame: chỉ là chọn N vị trí giãn/dồn rồi lấy frame gần nhất.
    """
    u = np.linspace(0, 1, n)
    if rng is not None:
        lo, hi = speed
        m = rng.uniform(0.3, 0.7)
        # m' phải làm cho cả 2 đoạn [0, m] -> [0, m'] và [m, 1] -> [m', 1] có hệ số trong [lo, hi]
        m2 = rng.uniform(max(lo * m, 1 - hi * (1 - m)), min(hi * m, 1 - lo * (1 - m)))
        u = np.interp(u, [0, m2, 1], [0, m, 1])
    return (u * (length - 1)).round().astype(int)


def feature_dim(cfg):
    c = 3 if cfg["use_z"] else 2
    hands = 2 * 21 * c * (2 if cfg["norm"] == "shoulder_wrist" else 1)
    return hands + len(POSE_KEEP) * c + 2


def make_features(seq, idx, cfg):
    """seq (T, 225) (đầu ra prepare_sequence, có thể đã augmentation), idx: n chỉ số frame -> (n, D) float32.

    Tâm vai, độ rộng vai, cỡ lòng bàn tay tính trên CẢ chuỗi rồi mới lấy n frame (giống nhau dù n bao nhiêu).
    Thứ tự mỗi frame: [2 tay (theo khung ảnh hoặc theo vai) | 2 tay theo cổ tay (chỉ shoulder_wrist) |
    11 điểm Pose | cờ trái, cờ phải]. Điểm vắng mặt = 0, cờ = 0.
    """
    norm, c = cfg["norm"], 3 if cfg["use_z"] else 2
    if norm not in NORMS:
        raise ValueError(f"norm phải là một trong {NORMS}")
    full = seq.reshape(len(seq), NUM_POINTS, 3)
    pts = full[idx].copy()
    left_ok = pts[:, LEFT].any(axis=(1, 2))
    right_ok = pts[:, RIGHT].any(axis=(1, 2))
    body_ok = pts[:, BODY].any(axis=(1, 2))

    parts = []
    if norm == "none":
        rel = pts
    else:
        center, width = shoulder_frame(full, full[:, BODY].any(axis=(1, 2)))
        rel = pts.copy()
        rel[:, :, :2] = (rel[:, :, :2] - center) / width
        rel[:, :, 2] /= width   # z cùng đơn vị với x, y
    rel[~left_ok, LEFT] = 0
    rel[~right_ok, RIGHT] = 0
    rel[~body_ok, BODY] = 0
    parts.append(rel[:, :42, :c].reshape(len(idx), -1))

    if norm == "shoulder_wrist":
        for hand, ok in ((LEFT, left_ok), (RIGHT, right_ok)):
            h_full = full[:, hand]
            seen = h_full.any(axis=(1, 2))
            palm = np.median(np.linalg.norm(h_full[seen, MIDDLE_MCP, :2] - h_full[seen, WRIST, :2], axis=1)) \
                if seen.any() else 1.0
            h = pts[:, hand]
            local = (h - h[:, WRIST:WRIST + 1]) / max(palm, MIN_PALM)
            local[~ok] = 0
            parts.append(local[:, :, :c].reshape(len(idx), -1))

    parts.append(rel[:, 42:][:, POSE_KEEP, :c].reshape(len(idx), -1))
    parts.append(np.stack([left_ok, right_ok], axis=1))
    return np.concatenate(parts, axis=1).astype(np.float32)
