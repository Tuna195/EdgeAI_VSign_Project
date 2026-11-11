"""Các model. Cùng giao diện: vào(batch, N frame, D đặc trưng) -> ra (batch, số từ)"""

from torch import nn

class MLP(nn.Module):
    """Mốc so sánh: trải phẳng cả chuỗi thành 1 vector -> Không biết frame nào trước, frame nào sau """

    def __init__(self, n_frames, dim, n_classes, dropout=0.3, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),                   # (B, N, D) -> (B, N*D)
            nn.Linear(n_frames*dim, hidden), #lớp ẩn
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes)    #Lớp ra: mỗi từ nhận 1 xác suất từ 0 -> 1
        )
    
    def forward(self, x):
        return self.net(x)
    
def count_params(model):
    """Tổng số tham số (trọng số + bias) của model"""
    return sum(p.numel() for p in model.parameters())
