"""Các model. Cùng giao diện: vào (batch, N frame, D đặc trưng) -> ra (batch, số từ) = logits (chưa softmax).
Phần 4: MLP. Phần 5 sẽ thêm GRU, BiGRU, Conv1D."""
import torch
from torch import nn


class MLP(nn.Module):
    """Mốc so sánh: trải phẳng cả chuỗi thành 1 vector -> không biết frame nào trước, frame nào sau."""

    def __init__(self, n_frames, dim, n_classes, dropout=0.3, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),                          # (B, N, D) -> (B, N*D)
            nn.Linear(n_frames * dim, hidden),     # lớp ẩn
            nn.ReLU(),
            nn.Dropout(dropout),                   # chỉ có tác dụng lúc train (Buổi 3)
            nn.Linear(hidden, n_classes),          # lớp ra: mỗi từ 1 số
        )

    def forward(self, x):
        return self.net(x)


def count_params(model):
    """Tổng số tham số (trọng số + bias) của model."""
    return sum(p.numel() for p in model.parameters())


class GRU(nn.Module):
    """Đọc lần lượt từng frame, giữ một bộ nhớ 128 số; bộ nhớ sau frame cuối -> phân loại.
    bidirectional=True (BiGRU, Buổi 2): thêm một GRU đọc ngược từ cuối về đầu, mỗi chiều 64 số, ghép lại thành 128."""

    def __init__(self, n_frames, dim, n_classes, dropout=0.3, bidirectional=False):
        super().__init__()
        hidden = 64 if bidirectional else 128
        self.gru = nn.GRU(dim, hidden, batch_first=True, bidirectional=bidirectional)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(128, n_classes))

    def forward(self, x):
        _, h = self.gru(x)                            # h: (số chiều, batch, hidden): bộ nhớ sau frame cuối
        return self.head(torch.cat(list(h), dim=1))   # 1 chiều: (batch, 128); 2 chiều: ghép xuôi + ngược


class Conv1D(nn.Module):
    """3 lớp tích chập theo thời gian (mỗi lớp nhìn 5 frame liền nhau) -> trung bình theo thời gian -> phân loại."""

    def __init__(self, n_frames, dim, n_classes, dropout=0.3, channels=128):
        super().__init__()
        layers, c_in = [], dim
        for _ in range(3):
            layers += [nn.Conv1d(c_in, channels, kernel_size=5, padding=2), nn.BatchNorm1d(channels), nn.ReLU()]
            c_in = channels
        self.conv = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(channels, n_classes))

    def forward(self, x):
        x = x.transpose(1, 2)                         # (B, N, D) -> (B, D, N): Conv1d cần kênh đứng trước thời gian
        return self.head(self.conv(x).mean(dim=2))    # trung bình theo thời gian -> (B, 128)


MODELS = ("mlp", "gru", "bigru", "conv1d")


def build_model(name, n_frames, dim, n_classes, dropout=0.3):
    """Tạo model theo tên. Phần 6, 7 chỉ cần đổi tên là đổi model."""
    if name == "mlp":
        return MLP(n_frames, dim, n_classes, dropout)
    if name in ("gru", "bigru"):
        return GRU(n_frames, dim, n_classes, dropout, bidirectional=(name == "bigru"))
    if name == "conv1d":
        return Conv1D(n_frames, dim, n_classes, dropout)
    raise ValueError(f"model phải là một trong {MODELS}")
