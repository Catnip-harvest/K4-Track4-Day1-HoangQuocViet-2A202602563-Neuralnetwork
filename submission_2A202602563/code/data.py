"""data.py — nạp train/eval đã chia sẵn, tách validation từ train, chuẩn hoá, đưa lên thiết bị.

Điều kiện trước: đã chạy `python scripts/split_data.py` (tạo data/processed/train.npz, eval.npz).

Quy ước dữ liệu (xem README mục 2 và 3):
    X : float32, shape (N, 54)   — 10 cột đầu là số liên tục, 44 cột sau là nhị phân (one-hot)
    y : int64,   shape (N,)      — nhãn 0..6
Tập eval CHỈ dùng để chấm điểm cuối. Không dùng nó để chọn cấu hình, chuẩn hoá hay dừng sớm.
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.model_selection import train_test_split

N_NUMERIC = 10    # số cột liên tục cần chuẩn hoá (cột 0..9)
N_FEATURES = 54
N_CLASSES = 7


def _check_xy(X, y, name: str) -> None:
    assert X.ndim == 2 and X.shape[1] == N_FEATURES, f"{name}: X phải có {N_FEATURES} cột, hiện {X.shape}"
    assert X.dtype == np.float32, f"{name}: X phải là float32, hiện {X.dtype}"
    assert y.shape == (len(X),), f"{name}: y phải có shape (N,), hiện {y.shape}"
    assert y.dtype == np.int64, f"{name}: y phải là int64, hiện {y.dtype}"
    assert y.min() >= 0 and y.max() <= N_CLASSES - 1, f"{name}: nhãn phải nằm trong 0..6"


def load_split(processed_dir: str = "data/processed"):
    """Nạp train và eval từ file .npz.

    Trả về: X_train_full, y_train_full, X_eval, y_eval, eval_row_id
    Nhãn đã được split_data.py trừ 1 (0..6), nên ở đây KHÔNG trừ thêm.
    """
    train = np.load(f"{processed_dir}/train.npz")
    evals = np.load(f"{processed_dir}/eval.npz")
    X_train_full, y_train_full = train["X"], train["y"]
    X_eval, y_eval, eval_row_id = evals["X"], evals["y"], evals["row_id"]

    _check_xy(X_train_full, y_train_full, "train")
    _check_xy(X_eval, y_eval, "eval")
    assert eval_row_id.shape == (len(X_eval),), "eval: row_id phải có một giá trị cho mỗi mẫu"
    assert len(np.unique(eval_row_id)) == len(eval_row_id), "eval: row_id bị lặp"
    return X_train_full, y_train_full, X_eval, y_eval, eval_row_id


def make_val_split(X, y, val_fraction: float = 0.2, seed: int = 42):
    """Tách validation TỪ train (không đụng eval), phân tầng theo nhãn.

    Trả về: X_tr, y_tr, X_val, y_val
    Seed tách (42) cố định cho mọi thí nghiệm; seed huấn luyện là một seed KHÁC (cfg["seed"]),
    nên đổi seed huấn luyện để đo nhiễu không làm thay đổi tập val.
    """
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=val_fraction, stratify=y, random_state=seed)
    return X_tr, y_tr, X_val, y_val


def fit_standardizer(X_tr):
    """mean và std của 10 cột số, tính CHỈ trên X_tr (phần train còn lại sau khi tách val).

    Nếu tính trên val/eval, thống kê của chính dữ liệu dùng để đánh giá sẽ "rò" vào phép biến
    đổi đầu vào: điểm val/eval không còn là ước lượng trung thực cho dữ liệu chưa thấy.
    Tính bằng float64 để tránh sai số cộng dồn trên ~370k dòng.
    """
    numeric = X_tr[:, :N_NUMERIC].astype(np.float64)
    mean = numeric.mean(axis=0)
    std = numeric.std(axis=0)
    std = np.where(std < 1e-12, 1.0, std)   # cột hằng số: giữ nguyên thay vì chia cho 0
    return mean.astype(np.float32), std.astype(np.float32)


def apply_standardizer(X, mean, std):
    """Trả về BẢN SAO của X: 10 cột đầu thành (x - mean) / std; 44 cột nhị phân giữ nguyên."""
    X_out = X.copy()
    X_out[:, :N_NUMERIC] = (X_out[:, :N_NUMERIC] - mean) / std
    return X_out


def prepare_data(device: str, val_fraction: float = 0.2, seed: int = 42,
                 processed_dir: str = "data/processed", verbose: bool = True) -> dict:
    """Gộp các bước trên và đưa TOÀN BỘ dữ liệu lên `device` một lần (không dùng DataLoader).

    Trả về dict gồm các tensor trên device: X_tr, y_tr, X_val, y_val, X_eval, y_eval (y là int64),
    mảng numpy eval_row_id (cùng thứ tự với X_eval), và mean/std dùng để chuẩn hoá.
    """
    X_full, y_full, X_eval, y_eval, eval_row_id = load_split(processed_dir)
    X_tr, y_tr, X_val, y_val = make_val_split(X_full, y_full, val_fraction, seed)

    mean, std = fit_standardizer(X_tr)              # chỉ X_tr
    X_tr = apply_standardizer(X_tr, mean, std)
    X_val = apply_standardizer(X_val, mean, std)    # cùng mean/std của train
    X_eval = apply_standardizer(X_eval, mean, std)  # áp dụng phép biến đổi, KHÔNG fit trên eval

    def to_x(a):
        return torch.tensor(a, dtype=torch.float32, device=device)

    def to_y(a):
        return torch.tensor(a, dtype=torch.int64, device=device)

    data = dict(X_tr=to_x(X_tr), y_tr=to_y(y_tr), X_val=to_x(X_val), y_val=to_y(y_val),
                X_eval=to_x(X_eval), y_eval=to_y(y_eval),
                eval_row_id=eval_row_id, mean=mean, std=std)

    if verbose:
        majority = int(np.bincount(y_tr, minlength=N_CLASSES).argmax())   # lớp đa số theo TRAIN
        print(f"X_tr   {tuple(X_tr.shape)} {X_tr.dtype} | y_tr   {tuple(y_tr.shape)} {y_tr.dtype}")
        print(f"X_val  {tuple(X_val.shape)} {X_val.dtype} | y_val  {tuple(y_val.shape)} {y_val.dtype}")
        print(f"X_eval {tuple(X_eval.shape)} {X_eval.dtype} | y_eval {tuple(y_eval.shape)} {y_eval.dtype}")
        print(f"Lớp đa số (theo train) = {majority}; accuracy 'luôn đoán lớp đa số' trên val = "
              f"{(y_val == majority).mean():.4f}")
    return data


def iterate_batches(X, y, batch_size: int, generator: torch.Generator | None = None, shuffle: bool = True):
    """Generator trả về từng cặp (xb, yb), thay cho DataLoader.

    Xáo bằng torch.randperm trên chính device của X (generator phải cùng device).
    Lô cuối: GIỮ LẠI dù nhỏ hơn batch_size (371 847 = 726·512 + 135 → lô cuối 135 mẫu). Loss
    lấy trung bình trong lô nên lô nhỏ vẫn cho gradient đúng thang đo, chỉ nhiễu hơn; bỏ đi thì
    mỗi epoch mất một phần dữ liệu khác nhau tuỳ cách xáo.
    """
    n = len(X)
    if shuffle:
        perm = torch.randperm(n, generator=generator, device=X.device)
    else:
        perm = torch.arange(n, device=X.device)
    for i in range(0, n, batch_size):
        idx = perm[i:i + batch_size]
        yield X[idx], y[idx]
