"""Tiền xử lý DÙNG CHUNG cho train và nhận diện trực tiếp (cùng hàm -> cùng kết quả).

Giai đoạn 1, một lần cho mỗi video:  prepare_sequence(raw, width, height) -> chuỗi (T, 225), CHƯA chuẩn hoá
Giai đoạn 2, mỗi lần lấy mẫu:        sample_positions(T, n) -> make_features(seq, idx, cfg) -> (n, D)
cfg = {"norm": "none" | "shoulder" | "shoulder_wrist", "use_z": True | False, "n_frames": 30}
"""

import numpy as np

LEFT, RIGHT, BODY = slice(0, 21), slice(21, 42), slice(42, 75)   # vị trí tay trái, tay phải, thân trong 75 điểm
L_SHOULDER, R_SHOULDER = 42 + 11, 42 + 12                        # vai trái, vai phải (Pose 11, 12)
MAX_GAP = 4          # tay mất <= 4 frame thì nội suy; lâu hơn coi như tay vắng mặt thật
MIN_FRAMES = 10      # chuỗi ngắn hơn thì bỏ video
MIN_SHOULDER = 0.05  # độ rộng vai nhỏ hơn mức này (đơn vị: chiều cao ảnh) = MediaPipe đặt sai vai
POSE_KEEP = [0, 2, 5, 9, 10, 11, 12, 13, 14, 15, 16]   # mũi, 2 mắt, 2 khoé miệng, 2 vai, 2 khuỷu, 2 cổ tay
NORMS = ("none", "shoulder", "shoulder_wrist")
WRIST, MIDDLE_BASE = 0, 9   # Hand: cổ tay, gốc ngón giữa -> khoảng cách giữa chúng là cỡ lòng bàn tay
MIN_PALM = 0.02             # cỡ lòng bàn tay nhỏ hơn thì kẹp lại (tránh chia cho số quá nhỏ)

def points(seq):
    """(T, 225) -> (T, 75, 3): 75 điểm, mỗi điểm (x, y, z)."""
    return seq.reshape(len(seq), 75, 3)

def fix_aspect(raw, width, height):
    """[1] Nhân x và z với W/H để x, y, z cùng đơn vị 'chiều cao ảnh'. Điểm bằng 0 vẫn bằng 0."""
    pts = points(raw).copy()
    pts[:, :, [0, 2]] *= width / height
    return pts.reshape(len(raw), -1)

def fill_gaps(raw):
    """[2] Nội suy tuyến tính chỗ tay bị mất <= MAX_GAP frame, nếu trước và sau đều thấy tay."""
    out = raw.copy()
    for start in (0, 63):                                    # tay trái: cột 0..62, tay phải: cột 63..125
        hand = out[:, start:start + 63]                      # "cửa sổ" nhìn vào out: sửa hand là sửa out
        seen = np.flatnonzero(hand.any(axis=1))              # các frame thấy tay này
        for a, b in zip(seen[:-1], seen[1:]):                # từng cặp frame thấy tay liền kề nhau
            if 1 < b - a <= MAX_GAP + 1:                     # giữa a và b mất 1..MAX_GAP frame
                w = np.linspace(0, 1, b - a + 1)[1:-1, None]     # trọng số của các frame ở giữa
                hand[a + 1:b] = (1 - w) * hand[a] + w * hand[b]
    return out

def shoulder_frame(pts):
    """pts (T, 75, 3) -> (tâm vai (2,), độ rộng vai): TRUNG VỊ trên các frame thấy thân."""
    body_ok = pts[:, BODY].any(axis=(1, 2))
    shoulders = pts[body_ok][:, [L_SHOULDER, R_SHOULDER], :2]                # (số frame thấy thân, 2 vai, xy)
    center = np.median(shoulders.mean(axis=1), axis=0)
    width = float(np.median(np.linalg.norm(shoulders[:, 0] - shoulders[:, 1], axis=1)))
    return center, width

def prepare_sequence(raw, width, height):
    """raw (T, 225) từ MediaPipe -> (chuỗi sạch (T', 225) float32, None) hoặc (None, lý do bỏ video)."""
    if len(raw) == 0:
        return None, "video rỗng"
    if not np.isfinite(raw).all():
        return None, "có NaN hoặc vô cực"
    seq = fill_gaps(fix_aspect(raw, width, height))
    used = np.flatnonzero(seq.any(axis=1))                   # [3] các frame KHÔNG trống
    if len(used) == 0:
        return None, "mọi frame đều trống"
    seq = seq[used[0]:used[-1] + 1]                          # giữ từ frame không trống đầu tiên đến cuối cùng
    pts = points(seq)
    if not pts[:, BODY].any():
        return None, "không thấy thân ở frame nào"
    if len(seq) < MIN_FRAMES:
        return None, f"ngắn hơn {MIN_FRAMES} frame"
    if shoulder_frame(pts)[1] < MIN_SHOULDER:
        return None, "độ rộng vai quá nhỏ"
    return seq.astype(np.float32), None



def sample_positions(length, n):
    """n vị trí frame cách đều trong chuỗi dài length. Luôn có frame đầu và frame cuối."""
    return (np.linspace(0, 1, n) * (length - 1)).round().astype(int)


def feature_dim(cfg):
    """Độ dài D của vector đặc trưng mỗi frame (= số đầu vào của model) theo cấu hình cfg.

    Vector mỗi frame là các phần sau nối tiếp nhau (c = 3 nếu dùng z: x, y, z; c = 2 nếu không: x, y):
      1. 2 tay x 21 điểm x c                     = 42*c  (toạ độ theo khung ảnh, hoặc theo vai)
      2. 2 tay x 21 điểm x c, tính theo cổ tay   = 42*c  (CHỈ có khi norm = "shoulder_wrist")
      3. 11 điểm thân (POSE_KEEP) x c            = 11*c
      4. 2 cờ: có thấy tay trái / tay phải       = 2

    Ví dụ: use_z=True,  norm="shoulder"       -> 126 + 33 + 2       = 161
           use_z=True,  norm="shoulder_wrist" -> 126 + 126 + 33 + 2 = 287
           use_z=False, norm="none"           -> 84 + 22 + 2        = 108
    """
    c = 3 if cfg["use_z"] else 2
    hand_copies = 2 if cfg["norm"] == "shoulder_wrist" else 1
    return 42 * c * hand_copies + len(POSE_KEEP) * c + 2

def make_features(seq, idx, cfg):
    """Biến n frame được chọn của 1 video thành ma trận đặc trưng (n, D) float32 để đưa vào model.

    Vào:
      seq: chuỗi (T, 225) đã qua prepare_sequence (đã sửa tỉ lệ, lấp chỗ mất, cắt 2 đầu; CHƯA chuẩn hoá).
      idx: n vị trí frame cần lấy, thường từ sample_positions(T, n).
      cfg: {"norm": cách chuẩn hoá toạ độ, "use_z": có giữ z không}.
    Ra: mỗi hàng là 1 frame, gồm các phần nối tiếp (D = feature_dim(cfg)):
      [tay trái 21 điểm | tay phải 21 điểm | (2 tay theo cổ tay, chỉ shoulder_wrist) | 11 điểm thân | cờ trái, cờ phải]
    Điểm không thấy = 0, cờ = 0. Cờ cho model phân biệt "không thấy tay" với "tay nằm ở toạ độ 0".

    norm:
      "none":     giữ toạ độ theo khung ảnh -> phụ thuộc người đứng ở đâu, xa hay gần camera.
      "shoulder": gốc toạ độ = tâm 2 vai, đơn vị = độ rộng vai -> không phụ thuộc vị trí, khoảng cách.
    """
    norm = cfg["norm"]
    c = 3 if cfg["use_z"] else 2                             # số toạ độ giữ lại mỗi điểm: (x, y, z) hoặc (x, y)
    if norm not in NORMS:                                    # gõ sai tên -> báo lỗi ngay, không âm thầm chạy sai
        raise ValueError(f"norm phải là một trong {NORMS}")

    full = points(seq)                                       # cả chuỗi (T, 75, 3)
    pts = full[idx].copy()                                   # chỉ n frame được chọn (n, 75, 3)

    # Ghi nhớ frame nào THẤY từng bộ phận, TRƯỚC khi chuẩn hoá (sau chuẩn hoá điểm 0 không còn bằng 0).
    # any(axis=(1, 2)): có số nào khác 0 trong (21 điểm × 3 toạ độ) của frame đó không -> mảng (n,) True/False.
    left_ok = pts[:, LEFT].any(axis=(1, 2))
    right_ok = pts[:, RIGHT].any(axis=(1, 2))
    body_ok = pts[:, BODY].any(axis=(1, 2))

    rel = pts.copy()                                         # toạ độ sẽ đưa vào model
    if norm != "none":
        # Tâm vai và độ rộng vai lấy TRUNG VỊ trên CẢ chuỗi (không chỉ n frame):
        # ổn định hơn, và cùng 1 video thì cùng 1 khung toạ độ dù lấy bao nhiêu frame.
        # An toàn chia: prepare_sequence đã bảo đảm có thân và độ rộng vai >= MIN_SHOULDER.
        center, width = shoulder_frame(full)                 # center (2,): (x, y); width: 1 số
        rel[:, :, :2] = (rel[:, :, :2] - center) / width     # x, y: dời gốc về tâm vai, rồi đổi đơn vị
        rel[:, :, 2] /= width                                # z: MediaPipe đã cho tương đối sẵn -> chỉ đổi đơn vị

    # Điểm vắng mặt (= 0) sau khi trừ tâm vai thành (-center/width) = số rác -> đặt lại 0.
    # rel[~left_ok, LEFT]: các frame KHÔNG thấy tay trái (~ là đảo True/False), 21 điểm của tay trái.
    rel[~left_ok, LEFT] = 0
    rel[~right_ok, RIGHT] = 0
    rel[~body_ok, BODY] = 0

    # Ghép đặc trưng. Mỗi phần đều được làm phẳng thành (n, số cột) để nối ngang được với nhau.
    n = len(idx)
    parts = [rel[:, :42, :c].reshape(n, -1)]                 # phần 1: 2 tay (42 điểm × c)
    if norm == "shoulder_wrist":                             # thêm bản tay lấy cổ tay làm gốc #Phần 2
        for hand, ok in ((LEFT, left_ok), (RIGHT, right_ok)):
            whole = full[:, hand]                            # bàn tay này trên cả chuỗi (T, 21, 3)
            seen = whole.any(axis=(1, 2))
            palm = np.median(np.linalg.norm(whole[seen, MIDDLE_BASE, :2] - whole[seen, WRIST, :2], axis=1)) \
                if seen.any() else 1.0                       # cỡ lòng bàn tay: trung vị cả chuỗi
            h = pts[:, hand]                                 # toạ độ GỐC của n frame được chọn
            local = (h - h[:, WRIST:WRIST + 1]) / max(palm, MIN_PALM)
            local[~ok] = 0
            parts.append(local[:, :, :c].reshape(len(idx), -1))

    # phần 3: rel[:, BODY] lấy 33 điểm Pose, rồi [:, POSE_KEEP] chọn 11 điểm trong số đó (11 điểm × c)
    parts.append(rel[:, BODY][:, POSE_KEEP, :c].reshape(n, -1))
    parts.append(np.stack([left_ok, right_ok], axis=1))      # phần 4: 2 cờ (n, 2) True/False
    return np.concatenate(parts, axis=1).astype(np.float32)  # nối ngang -> (n, D); True/False -> 1.0/0.0

