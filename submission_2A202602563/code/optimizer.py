"""optimizer.py — chọn bộ tối ưu và cắt gradient, để `train.py` gọn và mọi thí nghiệm công bằng.

Công thức (slide Chương 4):
    SGD            : w <- w - lr * g
    SGD + momentum : v <- mu * v + g ;  w <- w - lr * v          (dạng PyTorch)
    Adam           : m <- b1 m + (1-b1) g ; v <- b2 v + (1-b2) g^2 ; w <- w - lr * m_hat / (sqrt(v_hat) + eps)
    AdamW          : như Adam nhưng suy giảm trọng số tách riêng: w <- w - lr * wd * w - lr * m_hat / (sqrt(v_hat) + eps)
"""
from __future__ import annotations

import math

import torch

OPTIMIZERS = ("sgd", "sgd_momentum", "adam", "adamw")


def build_optimizer(name: str, params, lr: float, weight_decay: float = 0.0,
                    momentum: float = 0.9, betas=(0.9, 0.999), eps: float = 1e-8):
    """Trả về một torch.optim.Optimizer.

    weight_decay của Adam là L2 trộn vào gradient (g <- g + wd·w, rồi bị chia cho √v̂ như mọi
    gradient khác), còn của AdamW là suy giảm tách riêng (w <- w - lr·wd·w, không đi qua √v̂).
    Với weight_decay = 0 hai bộ này cho cùng phép cập nhật.
    """
    if name not in OPTIMIZERS:
        raise ValueError(f"optimizer phải thuộc {OPTIMIZERS}, nhận {name!r}")
    if lr is None:
        raise ValueError("lr chưa được chọn (DEFAULT_CFG để lr=None; chọn bằng val)")
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd_momentum":
        return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
    return torch.optim.AdamW(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)


def build_scheduler(optimizer, name: str | None, total_steps: int, **kwargs):
    """(Tuỳ chọn) Bộ lập lịch tốc độ học. Trả về None nếu name là None.

    "cosine": CosineAnnealingLR, giảm lr theo nửa chu kỳ cos về eta_min sau total_steps bước
    (gọi scheduler.step() sau MỖI bước cập nhật).
    """
    if name is None:
        return None
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps, eta_min=kwargs.get("eta_min", 0.0))
    raise ValueError(f"scheduler không hỗ trợ: {name!r}")


def clip_gradients(params, max_norm: float | None) -> torch.Tensor:
    """Cắt gradient theo chuẩn L2 toàn cục, và TRẢ VỀ chuẩn gradient TRƯỚC KHI cắt.

    ‖g‖ = sqrt(Σ_tham số Σ_phần tử g²) gộp mọi tham số; nếu ‖g‖ > c thì g <- g · c/‖g‖.
    max_norm = None: chỉ đo, không cắt (clip_grad_norm_ với max_norm = inf không bao giờ cắt).
    Trả về tensor 0-chiều (không .item()) để vòng huấn luyện không phải đồng bộ GPU mỗi bước;
    gọi float(...) khi cần một số Python.
    Với FP16 + GradScaler: phải scaler.unscale_(optimizer) TRƯỚC khi gọi hàm này.
    """
    limit = math.inf if max_norm is None else float(max_norm)
    return torch.nn.utils.clip_grad_norm_(params, limit)
