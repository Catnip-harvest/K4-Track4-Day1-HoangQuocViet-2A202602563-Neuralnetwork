"""train.py — đặt seed, đánh giá, vòng huấn luyện `run_experiment(cfg, data)`, dự đoán và ghi file nộp.

Mọi thí nghiệm chỉ là *đổi dict cfg* rồi gọi lại run_experiment (xem GUIDE, Part 2).
Mọi chỉ số (loss, accuracy, macro-F1) dùng cùng định nghĩa với scripts/evaluate.py.
"""
from __future__ import annotations

import json
import math
import os
import random
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from data import iterate_batches, N_CLASSES
from model import MLP, EXPECTED_PARAMS, count_params
from optimizer import build_optimizer, build_scheduler, clip_gradients

# Cấu hình mặc định = BASELINE (M-base). `lr` để None: notebook chọn bằng val (Part 2) rồi điền vào cfg.
DEFAULT_CFG = dict(
    exp_id="base-s1", group="baseline", description="Baseline M-base",
    loss="ce",                 # "ce" | "mse"
    optimizer="sgd_momentum",  # "sgd" | "sgd_momentum" | "adam" | "adamw"
    lr=None,                   # chọn bằng val, không dùng eval
    weight_decay=0.0, momentum=0.9, betas=(0.9, 0.999), eps=1e-8,
    batch=512, epochs=20,
    hidden=(256, 128), dropout=0.0, init="he",
    clip_norm=None,            # None = không clip; hoặc số, ví dụ 1.0
    precision="fp32",          # "fp32" | "fp16" | "bf16"
    scheduler=None,            # None | "cosine" (tuỳ chọn, ghi vào notes nếu dùng)
    seed=1,
    notes="",
)

# Train loss mỗi epoch đo ở eval() trên một tập con CỐ ĐỊNH 50 000 mẫu của X_tr
# (cùng tập con cho mọi epoch và mọi thí nghiệm: chọn bằng generator seed 0, không phụ thuộc cfg).
TRAIN_LOSS_SUBSET = 50_000
DIVERGENCE_CHECK_EVERY = 50     # số bước giữa hai lần kiểm tra NaN/inf (mỗi lần kiểm tra phải đồng bộ GPU)


def set_seed(seed: int) -> None:
    """Đặt seed cho random, numpy, torch (và torch.cuda nếu có)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def confusion_matrix_t(y_true: torch.Tensor, y_pred: torch.Tensor, k: int = N_CLASSES) -> np.ndarray:
    """Ma trận nhầm lẫn k×k (hàng = nhãn thật, cột = dự đoán), tính trên GPU bằng bincount."""
    flat = y_true * k + y_pred
    return torch.bincount(flat, minlength=k * k).reshape(k, k).cpu().numpy()


def macro_f1_from_confusion(cm: np.ndarray) -> float:
    """macro-F1 = trung bình cộng F1 của 7 lớp; F1_c = 2PR/(P+R), bằng 0 nếu P+R = 0.

    cm: ma trận nhầm lẫn (7, 7), hàng = nhãn thật, cột = dự đoán.
    TP_c = cm[c,c]; FP_c = tổng cột c − TP_c (dự đoán c nhưng sai); FN_c = tổng hàng c − TP_c (bỏ sót c).
    """
    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    prec = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    rec = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(tp), where=(prec + rec) > 0)
    return float(f1.mean())


@torch.no_grad()
def predict(model, X, batch_size: int = 8192) -> torch.Tensor:
    """Trả về nhãn dự đoán int64 (N,) = argmax của logits, ở chế độ eval() (dropout tắt)."""
    model.eval()
    preds = [model(X[i:i + batch_size]).argmax(dim=1) for i in range(0, len(X), batch_size)]
    return torch.cat(preds)


def compute_loss(logits, y, loss_name: str, reduction: str = "mean"):
    """"ce"  : cross-entropy trên logit thô và nhãn int64: −log softmax(z)_y. Softmax nằm TRONG hàm này.
       "mse" : MSE giữa logit và one-hot của y, lấy trung bình trên MỌI phần tử (B·7 phần tử),
               không có hệ số 1/2 (giống nn.MSELoss mặc định): (1/(B·7)) Σ_i Σ_c (z_ic − onehot_ic)².
    Luôn tính bằng float32 (khi autocast, logits có thể là fp16/bf16).
    """
    logits = logits.float()
    if loss_name == "ce":
        return F.cross_entropy(logits, y, reduction=reduction)
    if loss_name == "mse":
        target = F.one_hot(y, num_classes=logits.shape[1]).float()
        per_elem = F.mse_loss(logits, target, reduction="none")
        if reduction == "sum":
            return per_elem.sum() / logits.shape[1]      # tổng theo mẫu của trung bình theo lớp
        return per_elem.mean()
    raise ValueError(f"loss phải là 'ce' hoặc 'mse', nhận {loss_name!r}")


@torch.no_grad()
def evaluate(model, X, y, loss_name: str = "ce", batch_size: int = 8192) -> dict:
    """Trả về dict(loss, acc, macro_f1) ở chế độ eval() (dropout tắt), no_grad, FP32.

    Loss: cộng dồn tổng theo mẫu rồi chia N (lô cuối nhỏ hơn không làm lệch trung bình).
    """
    model.eval()
    total_loss = torch.zeros((), device=X.device)
    preds = []
    for i in range(0, len(X), batch_size):
        logits = model(X[i:i + batch_size])
        total_loss += compute_loss(logits, y[i:i + batch_size], loss_name, reduction="sum")
        preds.append(logits.argmax(dim=1))
    pred = torch.cat(preds)
    cm = confusion_matrix_t(y, pred)
    return dict(loss=float(total_loss) / len(X),
                acc=float((pred == y).float().mean()),
                macro_f1=macro_f1_from_confusion(cm))


def _fill_cfg(cfg: dict) -> dict:
    full = {**DEFAULT_CFG, **cfg}
    full["hidden"] = tuple(full["hidden"])
    full["betas"] = tuple(full["betas"])
    return full


def build_model(cfg: dict, device) -> MLP:
    """Tạo model theo cfg và assert số tham số khớp bảng quy định."""
    model = MLP(hidden=cfg["hidden"], dropout=cfg["dropout"], init=cfg["init"])
    n = count_params(model)
    assert n == EXPECTED_PARAMS[cfg["hidden"]], f"{cfg['hidden']}: {n} tham số, cần {EXPECTED_PARAMS[cfg['hidden']]}"
    return model.to(device)


def run_experiment(cfg: dict, data: dict, verbose: bool = True) -> dict:
    """Huấn luyện một cấu hình và trả về {"cfg", "history", "summary", "best_state"}.

    - step0_loss: loss trên val TRƯỚC bước cập nhật đầu tiên (kỳ vọng ≈ ln 7 với khởi tạo hợp lý).
    - Mỗi epoch: train_loss (eval mode, tập con cố định 50k), val_loss/val_acc/val_macro_f1,
      grad_norm = trung bình chuẩn L2 toàn cục TRƯỚC khi clip, grad_norm_max (để thấy gai),
      clip_frac = tỉ lệ bước có ‖g‖ > c (clipping thực sự kích hoạt), thời gian huấn luyện của epoch.
    - best_epoch = epoch có val_loss thấp nhất; val_acc/val_macro_f1 của summary lấy tại epoch đó;
      best_state = bản sao state_dict tại epoch đó (giữ trong bộ nhớ để dự đoán eval ở Part 4).
    - diverged: loss NaN/inf -> dừng ngay, giữ lịch sử đến epoch hoàn chỉnh cuối cùng.
    - peak_mem_MB: bộ nhớ GPU cực đại TRONG CÁC VÒNG HUẤN LUYỆN, trừ phần dữ liệu đã nằm sẵn trên GPU
      (~125 MB, như nhau cho mọi lần chạy): tức model + trạng thái optimizer + kích hoạt + gradient (+ bản
      sao FP16/BF16 khi autocast). Không tính lượt đánh giá FP32 (lô 8192) để so sánh AMP có nghĩa.
    TUYỆT ĐỐI không dùng X_eval trong hàm này: chọn epoch chỉ bằng val.
    """
    cfg = _fill_cfg(cfg)
    X_tr, y_tr, X_val, y_val = data["X_tr"], data["y_tr"], data["X_val"], data["y_val"]
    device = X_tr.device
    on_cuda = device.type == "cuda"

    set_seed(cfg["seed"])
    sub_idx = torch.randperm(len(X_tr), generator=torch.Generator().manual_seed(0))[:TRAIN_LOSS_SUBSET]
    X_sub, y_sub = X_tr[sub_idx.to(device)], y_tr[sub_idx.to(device)]
    if on_cuda:
        torch.cuda.synchronize()
        mem_start = torch.cuda.memory_allocated()     # chỉ dữ liệu đã nằm sẵn trên GPU
    train_peak = 0

    model = build_model(cfg, device)
    opt = build_optimizer(cfg["optimizer"], model.parameters(), lr=cfg["lr"],
                          weight_decay=cfg["weight_decay"], momentum=cfg["momentum"],
                          betas=cfg["betas"], eps=cfg["eps"])
    steps_per_epoch = math.ceil(len(X_tr) / cfg["batch"])
    scheduler = build_scheduler(opt, cfg["scheduler"], total_steps=steps_per_epoch * cfg["epochs"])

    amp_dtype = {"fp32": None, "fp16": torch.float16, "bf16": torch.bfloat16}[cfg["precision"]]
    scaler = torch.amp.GradScaler(device.type) if cfg["precision"] == "fp16" else None
    clip = cfg["clip_norm"]

    batch_gen = torch.Generator(device=device).manual_seed(cfg["seed"])        # thứ tự lô theo seed

    step0_loss = evaluate(model, X_val, y_val, cfg["loss"])["loss"]

    keys = ("epoch", "train_loss", "val_loss", "val_acc", "val_macro_f1",
            "grad_norm", "grad_norm_max", "clip_frac", "epoch_time_s")
    history = {k: [] for k in keys}
    best_val_loss, best_epoch, best_state = math.inf, None, None
    diverged, diverged_at = False, None

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        if on_cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()        # đo đỉnh bộ nhớ của riêng phần huấn luyện
        t0 = time.perf_counter()
        gn_sum = torch.zeros((), device=device)
        gn_max = torch.zeros((), device=device)
        n_finite = torch.zeros((), device=device)
        n_clipped = torch.zeros((), device=device)
        bad = torch.zeros((), dtype=torch.bool, device=device)
        n_steps = 0

        for xb, yb in iterate_batches(X_tr, y_tr, cfg["batch"], generator=batch_gen):
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                logits = model(xb)                      # autocast chỉ bọc forward + loss
            loss = compute_loss(logits, yb, cfg["loss"])

            opt.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()           # loss·s để gradient FP16 không bị underflow
                scaler.unscale_(opt)                    # chia lại cho s TRƯỚC khi đo/cắt gradient
            else:
                loss.backward()
            gn = clip_gradients(model.parameters(), clip)   # chuẩn TRƯỚC khi cắt
            if scaler is not None:
                scaler.step(opt)                        # tự bỏ qua bước nếu gradient có inf/NaN
                scaler.update()
            else:
                opt.step()
            if scheduler is not None:
                scheduler.step()

            finite = torch.isfinite(gn)
            gn_clean = torch.where(finite, gn, torch.zeros_like(gn))
            gn_sum += gn_clean
            gn_max = torch.maximum(gn_max, gn_clean)
            n_finite += finite.float()
            if clip is not None:
                n_clipped += (gn_clean > clip).float()
            bad |= ~torch.isfinite(loss.detach())
            n_steps += 1
            if n_steps % DIVERGENCE_CHECK_EVERY == 0 and bool(bad):
                break

        if on_cuda:
            torch.cuda.synchronize()
            train_peak = max(train_peak, torch.cuda.max_memory_allocated())
        epoch_time = time.perf_counter() - t0

        if bool(bad):
            diverged, diverged_at = True, epoch
            if verbose:
                print(f"[{cfg['exp_id']}] DIVERGED: loss NaN/inf ở epoch {epoch}, bước {n_steps}; dừng.")
            break

        tr = evaluate(model, X_sub, y_sub, cfg["loss"])
        va = evaluate(model, X_val, y_val, cfg["loss"])
        if not (math.isfinite(tr["loss"]) and math.isfinite(va["loss"])):
            diverged, diverged_at = True, epoch
            if verbose:
                print(f"[{cfg['exp_id']}] DIVERGED: train/val loss không hữu hạn ở epoch {epoch}; dừng.")
            break

        n_fin = max(float(n_finite), 1.0)
        history["epoch"].append(epoch)
        history["train_loss"].append(tr["loss"])
        history["val_loss"].append(va["loss"])
        history["val_acc"].append(va["acc"])
        history["val_macro_f1"].append(va["macro_f1"])
        history["grad_norm"].append(float(gn_sum) / n_fin)
        history["grad_norm_max"].append(float(gn_max))
        history["clip_frac"].append(float(n_clipped) / n_steps if clip is not None else 0.0)
        history["epoch_time_s"].append(epoch_time)

        if va["loss"] < best_val_loss:
            best_val_loss, best_epoch = va["loss"], epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

        if verbose:
            print(f"[{cfg['exp_id']}] ep {epoch:2d} | train {tr['loss']:.4f} | val {va['loss']:.4f} "
                  f"acc {va['acc']:.4f} f1 {va['macro_f1']:.4f} | gn {history['grad_norm'][-1]:.3f} "
                  f"(max {history['grad_norm_max'][-1]:.2f}) | {epoch_time:.1f}s")

    peak_mem = (train_peak - mem_start) / 2**20 if on_cuda and train_peak else None
    have = len(history["epoch"]) > 0
    b = best_epoch - 1 if best_epoch is not None else None
    summary = dict(
        step0_loss=step0_loss,
        best_val_loss=best_val_loss if have else None,
        best_epoch=best_epoch,
        final_train_loss=history["train_loss"][-1] if have else None,
        final_val_loss=history["val_loss"][-1] if have else None,
        val_acc=history["val_acc"][b] if b is not None else None,
        val_macro_f1=history["val_macro_f1"][b] if b is not None else None,
        time_per_epoch_s=float(np.mean(history["epoch_time_s"])) if have else None,
        peak_mem_MB=peak_mem,
        diverged=diverged,
        diverged_epoch=diverged_at,
        steps_per_epoch=steps_per_epoch,
        epochs_completed=len(history["epoch"]),
        device=torch.cuda.get_device_name(0) if on_cuda else str(device),
    )
    if verbose:
        if have:
            print(f"[{cfg['exp_id']}] step0 {step0_loss:.4f} | best ep {best_epoch} val_loss {best_val_loss:.4f} "
                  f"acc {summary['val_acc']:.4f} f1 {summary['val_macro_f1']:.4f} | "
                  f"{summary['time_per_epoch_s']:.2f}s/ep | diverged={diverged}")
        else:
            print(f"[{cfg['exp_id']}] step0 {step0_loss:.4f} | không có epoch hoàn chỉnh | diverged={diverged}")
    return dict(cfg=cfg, history=history, summary=summary, best_state=best_state)


def write_predictions(row_id, preds, path: str) -> None:
    """Ghi file nộp cho scripts/evaluate.py: CSV có tiêu đề `row_id,pred` (cùng thứ tự với X_eval)."""
    row_id = np.asarray(row_id).astype(np.int64)
    preds = np.asarray(preds).astype(np.int64)
    assert len(row_id) == len(preds), "số dự đoán phải bằng số row_id"
    assert len(np.unique(row_id)) == len(row_id), "row_id bị lặp"
    assert preds.min() >= 0 and preds.max() <= N_CLASSES - 1, "pred phải nằm trong 0..6"
    pd.DataFrame({"row_id": row_id, "pred": preds}).to_csv(path, index=False)


def final_eval(cfg: dict, result: dict, data: dict, pred_path: str,
               repo_root: str | None = None, out_json: str | None = None) -> dict | None:
    """Dùng cho cấu hình cuối cùng (và baseline): nạp best_state, dự đoán eval, ghi predictions.

    Nếu có repo_root và out_json: chạy đúng scripts/evaluate.py (script chấm chính thức) và trả về
    nội dung eval_result.json. Không chỉnh gì dựa trên kết quả này.
    """
    cfg = _fill_cfg(cfg)
    pred_path = os.path.abspath(pred_path)
    device = data["X_eval"].device
    model = build_model(cfg, device)
    model.load_state_dict(result["best_state"])
    preds = predict(model, data["X_eval"])          # FP32, eval mode
    write_predictions(data["eval_row_id"], preds.cpu().numpy(), pred_path)
    print(f"đã ghi {pred_path} ({len(preds)} dòng; cấu hình {cfg['exp_id']}, seed {cfg['seed']}, "
          f"best_epoch {result['summary']['best_epoch']})")
    if repo_root is None or out_json is None:
        return None
    out_json = os.path.abspath(out_json)
    proc = subprocess.run([sys.executable, "scripts/evaluate.py", "--pred", pred_path, "--out", out_json],
                          cwd=repo_root, capture_output=True, text=True, encoding="utf-8")
    print(proc.stdout)
    if proc.returncode != 0:
        print(proc.stderr)
        raise RuntimeError("scripts/evaluate.py báo lỗi; xem thông báo ở trên")
    with open(out_json, encoding="utf-8") as f:
        return json.load(f)
