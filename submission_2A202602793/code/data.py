"""Data loading, stratified validation splitting, and train-only standardization.

Nhiệm vụ: nạp tập train/eval đã chia sẵn, tách validation từ train, chuẩn hoá, đưa lên thiết bị.

Điều kiện trước: đã chạy `python scripts/split_data.py` (tạo data/processed/train.npz, eval.npz).

Quy ước dữ liệu (xem README mục 2 và 3):
    X : float32, shape (N, 54)   — 10 cột đầu là số liên tục, 44 cột sau là nhị phân (one-hot)
    y : int64,   shape (N,)      — nhãn 0..6
Tập eval CHỈ dùng để chấm điểm cuối. Không dùng nó để chọn cấu hình, chuẩn hoá hay dừng sớm.
"""
from __future__ import annotations

import numpy as np
import torch

N_NUMERIC = 10  # số cột liên tục cần chuẩn hoá (cột 0..9)


def load_split(processed_dir: str = "data/processed"):
    """Nạp train và eval từ file .npz.

    Trả về: X_train_full, y_train_full, X_eval, y_eval, eval_row_id
    Các bước:
      1. np.load(f"{processed_dir}/train.npz") -> khoá "X", "y"
      2. np.load(f"{processed_dir}/eval.npz")  -> khoá "X", "y", "row_id"
      3. assert shape/dtype đúng quy ước ở đầu file
    """
    from pathlib import Path

    directory = Path(processed_dir)
    with np.load(directory / "train.npz") as train, np.load(directory / "eval.npz") as evaluation:
        X_train = train["X"]
        y_train = train["y"]
        X_eval = evaluation["X"]
        y_eval = evaluation["y"]
        eval_row_id = evaluation["row_id"]

    assert X_train.ndim == 2 and X_train.shape[1] == 54, f"X_train phải có shape (N, 54), nhận {X_train.shape}"
    assert X_eval.ndim == 2 and X_eval.shape[1] == 54, f"X_eval phải có shape (N, 54), nhận {X_eval.shape}"
    assert X_train.dtype == np.float32 and X_eval.dtype == np.float32, "X phải có dtype float32"
    assert y_train.ndim == 1 and y_train.dtype == np.int64 and len(y_train) == len(X_train), "y_train phải là int64, shape (N,)"
    assert y_eval.ndim == 1 and y_eval.dtype == np.int64 and len(y_eval) == len(X_eval), "y_eval phải là int64, shape (N,)"
    assert eval_row_id.ndim == 1 and eval_row_id.dtype == np.int64 and len(eval_row_id) == len(X_eval), "eval row_id phải là int64, shape (N,)"
    assert np.all((0 <= y_train) & (y_train <= 6)) and np.all((0 <= y_eval) & (y_eval <= 6)), "nhãn phải nằm trong 0..6"
    return X_train, y_train, X_eval, y_eval, eval_row_id


def make_val_split(X, y, val_fraction: float = 0.2, seed: int = 42):
    """Tách validation TỪ train (không đụng eval). Phân tầng theo nhãn.

    Trả về: X_tr, y_tr, X_val, y_val
    Gợi ý: sklearn.model_selection.train_test_split(..., stratify=y, random_state=seed)
    Dùng CÙNG seed và val_fraction cho mọi thí nghiệm để so sánh công bằng.
    """
    from sklearn.model_selection import train_test_split

    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction phải nằm giữa 0 và 1")
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=val_fraction, random_state=seed, stratify=y
    )
    return X_tr, y_tr, X_val, y_val


def fit_standardizer(X_tr):
    """Tính mean và std của N_NUMERIC cột đầu CHỈ trên tập train (sau khi tách val).

    Trả về: mean (shape (10,)), std (shape (10,))
    Câu hỏi: vì sao không được tính trên toàn bộ dữ liệu hay trên eval?
    """
    X_tr = np.asarray(X_tr, dtype=np.float32)
    if X_tr.ndim != 2 or X_tr.shape[1] < N_NUMERIC:
        raise ValueError(f"X_tr phải có ít nhất {N_NUMERIC} cột số, nhận shape {X_tr.shape}")
    mean = X_tr[:, :N_NUMERIC].mean(axis=0, dtype=np.float64).astype(np.float32)
    std = X_tr[:, :N_NUMERIC].std(axis=0, dtype=np.float64).astype(np.float32)
    # Hằng số không mang thông tin biến thiên; giữ cột đó ở 0 sau chuẩn hoá.
    std[std == 0] = 1.0
    return mean, std


def apply_standardizer(X, mean, std):
    """Trả về bản sao của X, trong đó 10 cột đầu được (x - mean) / std; 44 cột nhị phân giữ nguyên.

    Chú ý: không sửa X tại chỗ nếu bạn còn dùng lại nó; chú ý std = 0 (nếu có).
    """
    X = np.asarray(X)
    mean = np.asarray(mean, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)
    if mean.shape != (N_NUMERIC,) or std.shape != (N_NUMERIC,):
        raise ValueError(f"mean và std phải có shape ({N_NUMERIC},)")
    if X.ndim != 2 or X.shape[1] < N_NUMERIC:
        raise ValueError(f"X phải có ít nhất {N_NUMERIC} cột số, nhận shape {X.shape}")
    if np.any(std <= 0):
        raise ValueError("std phải dương; hãy dùng fit_standardizer để xử lý cột hằng")
    transformed = X.astype(np.float32, copy=True)
    numeric = X[:, :N_NUMERIC].astype(np.float64, copy=False)
    transformed[:, :N_NUMERIC] = (
        (numeric - mean.astype(np.float64)) / std.astype(np.float64)
    ).astype(np.float32)
    return transformed


def prepare_data(device: str, val_fraction: float = 0.2, seed: int = 42,
                 processed_dir: str = "data/processed") -> dict:
    """Gộp các bước trên và đưa TOÀN BỘ dữ liệu lên `device` một lần (không dùng DataLoader).

    Trả về dict gồm các tensor trên device:
        X_tr, y_tr, X_val, y_val, X_eval, y_eval        (y là int64)
    và các mảng numpy: eval_row_id
    Các bước:
      1. load_split -> make_val_split -> fit_standardizer (chỉ trên X_tr)
      2. apply_standardizer cho X_tr, X_val, X_eval bằng CÙNG mean/std
      3. torch.tensor(..., device=device); X là float32, y là int64
      4. in ra kích thước các tập và accuracy của chiến lược "luôn đoán lớp đa số" trên val
    """
    X_train_full, y_train_full, X_eval, y_eval, eval_row_id = load_split(processed_dir)
    X_tr, y_tr, X_val, y_val = make_val_split(
        X_train_full, y_train_full, val_fraction=val_fraction, seed=seed
    )
    mean, std = fit_standardizer(X_tr)
    X_tr = apply_standardizer(X_tr, mean, std)
    X_val = apply_standardizer(X_val, mean, std)
    X_eval = apply_standardizer(X_eval, mean, std)

    result = {
        "X_tr": torch.as_tensor(X_tr, dtype=torch.float32, device=device),
        "y_tr": torch.as_tensor(y_tr, dtype=torch.int64, device=device),
        "X_val": torch.as_tensor(X_val, dtype=torch.float32, device=device),
        "y_val": torch.as_tensor(y_val, dtype=torch.int64, device=device),
        "X_eval": torch.as_tensor(X_eval, dtype=torch.float32, device=device),
        "y_eval": torch.as_tensor(y_eval, dtype=torch.int64, device=device),
        "eval_row_id": eval_row_id,
    }
    train_counts = np.bincount(y_tr, minlength=7)
    majority_class = int(train_counts.argmax())
    majority_accuracy = float((y_val == majority_class).mean())
    print(f"X_tr={tuple(result['X_tr'].shape)}, X_val={tuple(result['X_val'].shape)}, X_eval={tuple(result['X_eval'].shape)}")
    print(f"val accuracy (luôn đoán lớp đa số train {majority_class}) = {majority_accuracy:.4f}")
    return result


def iterate_batches(X, y, batch_size: int, generator: torch.Generator | None = None, shuffle: bool = True):
    """Generator trả về từng cặp (xb, yb), thay cho DataLoader.

    Các bước:
      1. nếu shuffle: perm = torch.randperm(len(X), generator=generator, device=X.device); ngược lại arange
      2. for i in range(0, N, batch_size): idx = perm[i:i+batch_size]; yield X[idx], y[idx]
    Chú ý: batch cuối có thể nhỏ hơn batch_size; hãy quyết định bạn xử lý thế nào và ghi lại.
    """
    if batch_size <= 0:
        raise ValueError("batch_size phải là số nguyên dương")
    n = len(X)
    if len(y) != n:
        raise ValueError("X và y phải có cùng số mẫu")
    if shuffle:
        perm = torch.randperm(n, generator=generator, device=X.device)
    else:
        perm = torch.arange(n, device=X.device)
    for start in range(0, n, batch_size):
        idx = perm[start:start + batch_size]
        yield X[idx], y[idx]
