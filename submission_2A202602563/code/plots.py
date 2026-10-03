"""plots.py — mỗi thí nghiệm một ảnh figures/<exp_id>.png, và ảnh chồng figures/compare_<nhóm>.png."""
from __future__ import annotations

import matplotlib.pyplot as plt

OPT_LABEL = {"sgd": "SGD", "sgd_momentum": "SGD+mom", "adam": "Adam", "adamw": "AdamW"}


def describe_cfg(cfg: dict) -> str:
    """Một dòng tóm tắt cấu hình cho tiêu đề ảnh."""
    hidden = "-".join(str(h) for h in cfg["hidden"])
    clip = "none" if cfg.get("clip_norm") is None else cfg["clip_norm"]
    parts = [f"{cfg['loss'].upper()}", f"{OPT_LABEL.get(cfg['optimizer'], cfg['optimizer'])} lr={cfg['lr']:g}",
             f"wd={cfg['weight_decay']:g}", f"batch={cfg['batch']}", f"{hidden}", f"drop={cfg['dropout']:g}",
             f"clip={clip}", cfg["precision"], f"init={cfg['init']}", f"seed={cfg['seed']}"]
    if cfg.get("scheduler"):
        parts.append(f"sched={cfg['scheduler']}")
    return " | ".join(parts)


def plot_run(result: dict, path: str) -> None:
    """Một thí nghiệm -> PNG 3 ô: (1) train/val loss, (2) val acc + macro-F1, (3) grad_norm trước clip.

    Đường thẳng đứng nét đứt = best_epoch (val loss thấp nhất). Ô 3 vẽ cả trung bình và max trong
    epoch, và ngưỡng c nếu có clip, để thấy clipping có thực sự kích hoạt không.
    """
    cfg, h, s = result["cfg"], result["history"], result["summary"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
    ep = h["epoch"]

    ax = axes[0]
    ax.plot(ep, h["train_loss"], "o-", ms=3, label="train loss (eval mode, 50k cố định)")
    ax.plot(ep, h["val_loss"], "s-", ms=3, label="val loss")
    ax.plot([], [], " ", label=f"loss bước 0 (val) = {s['step0_loss']:.3f}")   # chỉ ghi ở chú thích để không ép thang trục
    ax.set_title(f"Loss ({cfg['loss'].upper()})")
    ax.set_xlabel("epoch"); ax.set_ylabel("loss")

    ax = axes[1]
    ax.plot(ep, h["val_acc"], "o-", ms=3, label="val accuracy")
    ax.plot(ep, h["val_macro_f1"], "s-", ms=3, label="val macro-F1")
    ax.axhline(0.4876, color="gray", lw=0.8, ls=":", label="đoán đa số (acc 0.4876)")
    ax.set_title("Val accuracy / macro-F1")
    ax.set_xlabel("epoch"); ax.set_ylabel("điểm")

    ax = axes[2]
    ax.plot(ep, h["grad_norm"], "o-", ms=3, label="‖g‖ trung bình / epoch")
    ax.plot(ep, h["grad_norm_max"], "^--", ms=3, alpha=0.7, label="‖g‖ max / epoch")
    if cfg.get("clip_norm") is not None:
        ax.axhline(cfg["clip_norm"], color="red", lw=1, ls="--", label=f"ngưỡng clip c = {cfg['clip_norm']:g}")
    ax.set_yscale("log")
    ax.set_title("grad_norm (L2 toàn cục, TRƯỚC clip)")
    ax.set_xlabel("epoch"); ax.set_ylabel("‖g‖ (log)")

    for ax in axes:
        if s["best_epoch"] is not None:
            ax.axvline(s["best_epoch"], color="green", lw=0.8, ls="--")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    status = "DIVERGED" if s["diverged"] else (
        f"best ep {s['best_epoch']}: val_loss {s['best_val_loss']:.4f}, acc {s['val_acc']:.4f}, "
        f"macro-F1 {s['val_macro_f1']:.4f}" if s["best_epoch"] is not None else "không có epoch hoàn chỉnh")
    fig.suptitle(f"{cfg['exp_id']} — {status}\n{describe_cfg(cfg)}", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_compare(results: list[dict], metric, path: str, title: str = "", logy: bool = False) -> None:
    """Vẽ chồng một hoặc nhiều chỉ số của nhiều thí nghiệm, mỗi thí nghiệm một đường (nhãn = exp_id).

    metric: tên một khoá của history (vd "val_loss") hoặc danh sách khoá -> mỗi khoá một ô.
    """
    metrics = [metric] if isinstance(metric, str) else list(metric)
    fig, axes = plt.subplots(1, len(metrics), figsize=(5.6 * len(metrics), 4.4), squeeze=False)
    for ax, m in zip(axes[0], metrics):
        for r in results:
            h = r["history"]
            label = r["cfg"]["exp_id"] + (" (diverged)" if r["summary"]["diverged"] else "")
            ax.plot(h["epoch"], h[m], "o-", ms=2.5, label=label)
        ax.set_title(m)
        ax.set_xlabel("epoch"); ax.set_ylabel(m)
        if logy or m.startswith("grad_norm"):
            ax.set_yscale("log")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
