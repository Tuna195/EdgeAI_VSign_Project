import torch
from torch import nn

from src.training.models import MLP, count_params

torch.manual_seed(0)

# 1. Một nơ-ron, tính tay
x = torch.tensor([0.5, -1.0, 2.0])
w = torch.tensor([0.2, 0.4, 0.1])
b = 0.1
z = (w * x).sum() + b
print("1. z =", round(z.item(), 4), "| ReLU(z) =", torch.relu(z).item())

# 2. Một lớp Linear(3, 2): so tính tay với PyTorch
layer = nn.Linear(3, 2)
print("2. weight", tuple(layer.weight.shape), "| bias", tuple(layer.bias.shape))
by_hand = layer.weight @ x + layer.bias
print("   tính tay:", by_hand.detach().numpy().round(4), "| PyTorch:", layer(x).detach().numpy().round(4))
print("   số tham số:", count_params(layer))

# 3. MLP của dự án trên 1 batch giả
model = MLP(n_frames=30, dim=161, n_classes=28)
print("3.", model)
batch = torch.randn(32, 30, 161)                   # 32 "video" ngẫu nhiên
logits = model(batch)
print("   vào", tuple(batch.shape), "-> ra", tuple(logits.shape))
for name, p in model.named_parameters():
    print(f"   {name:14s} {str(tuple(p.shape)):12s} {p.numel():>9,d}")
print(f"   TỔNG: {count_params(model):,d} tham số")