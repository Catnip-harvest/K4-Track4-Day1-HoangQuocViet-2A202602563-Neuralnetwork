"""model.py — MLP cho bài toán 7 lớp, shape cố định (README mục 3, GUIDE "Quy định kiến trúc"):

    x (B, 54) -> Linear(54, h1) -> ReLU -> [Dropout] -> Linear(h1, h2) -> ReLU -> [Dropout]
              -> ... -> Linear(h_last, 7) -> logits (B, 7)

Quy tắc:
  - Lớp cuối ra logit thô, KHÔNG softmax trong model (softmax nằm trong hàm mất mát).
  - Dropout chỉ đặt sau ReLU của lớp ẩn; không đặt trên đầu vào hay logit.
  - Mọi nn.Linear đều có bias. Không BatchNorm, không residual.
  - Số tham số phải khớp EXPECTED_PARAMS bên dưới.

Đếm tham số M-base: (54·256 + 256) + (256·128 + 128) + (128·7 + 7) = 14 080 + 32 896 + 903 = 47 879.
"""
from __future__ import annotations

import torch
import torch.nn as nn

# Số tham số bắt buộc ứng với từng kiến trúc (in_features=54, num_classes=7)
EXPECTED_PARAMS = {
    (256, 128): 47_879,        # M-base  (baseline)
    (512, 256): 161_287,       # M-wide  (tuỳ chọn)
    (256, 128, 64): 55_687,    # M-deep  (tuỳ chọn)
}

INITS = ("zeros", "normal", "xavier", "he", "default")


class MLP(nn.Module):
    """MLP theo quy định ở đầu file.

    Args:
        hidden:   tuple số nơ-ron các lớp ẩn, ví dụ (256, 128)
        dropout:  xác suất TẮT nơ-ron q (nn.Dropout dùng p chính là xác suất tắt); 0.0 = không dùng
        init:     "zeros" | "normal" | "xavier" | "he" | "default"
    """

    def __init__(self, hidden=(256, 128), dropout: float = 0.0, init: str = "he",
                 in_features: int = 54, num_classes: int = 7):
        super().__init__()
        layers: list[nn.Module] = []
        n_in = in_features
        for h in hidden:
            layers += [nn.Linear(n_in, h), nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))   # sau ReLU của lớp ẩn
            n_in = h
        layers.append(nn.Linear(n_in, num_classes))     # lớp ra: logit thô
        self.net = nn.Sequential(*layers)
        init_weights(self, init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 54) float32  ->  logits: (B, 7) float32."""
        return self.net(x)


def init_weights(model: nn.Module, init: str) -> None:
    """Khởi tạo tham số của MỌI nn.Linear; với mọi cách trừ "default", bias = 0.

    init:
        "zeros"   : W = 0
        "normal"  : W ~ N(0, 0.01^2)
        "xavier"  : nn.init.xavier_normal_  -> Var[W] = 2/(n_in + n_out)  (KHÔNG phải 1/n_in của slide)
        "he"      : nn.init.kaiming_normal_(w, nonlinearity="relu") -> Var[W] = 2/n_in  (fan_in)
        "default" : giữ khởi tạo mặc định của nn.Linear: kaiming_uniform_(a=√5) cho W, tức
                    W ~ U(-1/√n_in, 1/√n_in), Var[W] = 1/(3·n_in); bias cũng U(±1/√n_in). KHÔNG phải He.
    """
    if init not in INITS:
        raise ValueError(f"init phải thuộc {INITS}, nhận {init!r}")
    if init == "default":
        return
    for m in model.modules():
        if not isinstance(m, nn.Linear):
            continue
        if init == "zeros":
            nn.init.zeros_(m.weight)
        elif init == "normal":
            nn.init.normal_(m.weight, mean=0.0, std=0.01)
        elif init == "xavier":
            nn.init.xavier_normal_(m.weight)
        elif init == "he":
            nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
        nn.init.zeros_(m.bias)


def count_params(model: nn.Module) -> int:
    """Tổng số tham số huấn luyện được. Dùng để assert với EXPECTED_PARAMS ngay sau khi tạo model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def activation_stats(model: nn.Module, x: torch.Tensor) -> list[float]:
    """Độ lệch chuẩn của kích hoạt ở bước 0 trên một lô (dùng cho thí nghiệm khởi tạo).

    Đo SAU MỖI nn.Linear (tiền kích hoạt, trước ReLU). Phần tử cuối cùng là std của logits,
    liên hệ trực tiếp với loss bước 0: logits càng xa 0 thì softmax càng "tự tin" và loss bước 0
    càng lệch khỏi ln 7.
    """
    model.eval()
    stds = []
    h = x
    for layer in model.net:
        h = layer(h)
        if isinstance(layer, nn.Linear):
            stds.append(h.float().std().item())
    return stds
