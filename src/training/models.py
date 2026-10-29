"""4 model, cùng giao diện: vào (batch, N frame, D đặc trưng) -> ra (batch, số từ) (logits, chưa softmax)."""
import torch
from torch import nn

MODELS = ("mlp", "gru", "bigru", "conv1d")


class MLP(nn.Module):
    """Mốc so sánh: trải phẳng cả chuỗi thành 1 vector -> không biết frame nào trước, frame nào sau."""

    def __init__(self, n_frames, dim, n_classes, dropout):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(n_frames * dim, 256), nn.ReLU(),
                                 nn.Dropout(dropout), nn.Linear(256, n_classes))

    def forward(self, x):
        return self.net(x)


class GRU(nn.Module):
    """Đọc lần lượt từng frame, giữ 1 "trí nhớ" 128 số; trí nhớ sau frame cuối -> phân loại.
    bidirectional: thêm 1 GRU đọc ngược từ cuối về đầu (64 + 64 = 128). Hợp lệ vì phân loại trọn đoạn ký."""

    def __init__(self, n_frames, dim, n_classes, dropout, bidirectional=False):
        super().__init__()
        hidden = 64 if bidirectional else 128
        self.gru = nn.GRU(dim, hidden, batch_first=True, bidirectional=bidirectional)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(128, n_classes))

    def forward(self, x):
        _, h = self.gru(x)                          # h: (số chiều, batch, hidden)
        return self.head(torch.cat(list(h), dim=1))  # 1 chiều: (batch, 128); 2 chiều: ghép xuôi + ngược


class Conv1D(nn.Module):
    """3 lớp tích chập theo thời gian (mỗi lớp nhìn 5 frame liền nhau) -> trung bình theo thời gian -> phân loại."""

    def __init__(self, n_frames, dim, n_classes, dropout, channels=128):
        super().__init__()
        layers, c_in = [], dim
        for _ in range(3):
            layers += [nn.Conv1d(c_in, channels, kernel_size=5, padding=2), nn.BatchNorm1d(channels), nn.ReLU()]
            c_in = channels
        self.conv = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(channels, n_classes))

    def forward(self, x):
        return self.head(self.conv(x.transpose(1, 2)).mean(dim=2))   # Conv1d cần (batch, D, N)


def build_model(name, n_frames, dim, n_classes, dropout=0.3):
    if name == "mlp":
        return MLP(n_frames, dim, n_classes, dropout)
    if name in ("gru", "bigru"):
        return GRU(n_frames, dim, n_classes, dropout, bidirectional=name == "bigru")
    if name == "conv1d":
        return Conv1D(n_frames, dim, n_classes, dropout)
    raise ValueError(f"model phải là một trong {MODELS}")


def count_params(model):
    return sum(p.numel() for p in model.parameters())
