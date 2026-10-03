"""Training, validation, metrics, and final prediction utilities.

Gồm: đặt seed, đánh giá, vòng huấn luyện `run_experiment(cfg, data)`, dự đoán và ghi file nộp.
Mọi thí nghiệm chỉ là *đổi dict cfg* rồi gọi lại run_experiment (xem GUIDE, Part 2).

Mọi chỉ số (loss, accuracy, macro-F1) dùng cùng định nghĩa với scripts/evaluate.py.
"""
from __future__ import annotations

import csv
import math
import random
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from data import iterate_batches
from model import MLP, EXPECTED_PARAMS, count_params
from optimizer import build_optimizer, clip_gradients

# Cấu hình mặc định = BASELINE (M-base). `lr` do bạn tự chọn bằng val rồi điền vào.
DEFAULT_CFG = dict(
    exp_id="base-s1", group="baseline", description="Baseline M-base",
    loss="ce",                 # "ce" | "mse"
    optimizer="sgd_momentum",  # "sgd" | "sgd_momentum" | "adam" | "adamw"
    lr=None,                   # TODO: chọn bằng val, không dùng eval
    weight_decay=0.0, momentum=0.9, betas=(0.9, 0.999), eps=1e-8,
    batch=512, epochs=20,
    hidden=(256, 128), dropout=0.0, init="he",
    clip_norm=None,            # None = không clip; hoặc số, ví dụ 1.0
    precision="fp32",          # "fp32" | "fp16" | "bf16"
    seed=1,
)


def set_seed(seed: int) -> None:
    """Đặt seed cho random, numpy, torch (và torch.cuda nếu có)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def macro_f1_from_confusion(cm: np.ndarray) -> float:
    """macro-F1 = trung bình cộng F1 của 7 lớp; F1_c = 2PR/(P+R), bằng 0 nếu P+R = 0.

    cm: ma trận nhầm lẫn (7, 7), hàng = nhãn thật, cột = dự đoán.
    """
    cm = np.asarray(cm)
    if cm.shape != (7, 7):
        raise ValueError(f"confusion matrix must have shape (7, 7), received {cm.shape}")
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0, dtype=np.float64) - tp
    fn = cm.sum(axis=1, dtype=np.float64) - tp
    denominator = 2.0 * tp + fp + fn
    f1 = np.divide(2.0 * tp, denominator, out=np.zeros_like(tp), where=denominator > 0)
    return float(f1.mean())


@torch.no_grad()
def predict(model, X, batch_size: int = 8192) -> torch.Tensor:
    """Trả về nhãn dự đoán int64 (N,) = argmax của logits.

    Các bước: model.eval(); duyệt X theo từng lô (không cần xáo); gom argmax(dim=1); torch.cat.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    model.eval()
    predictions = []
    for start in range(0, len(X), batch_size):
        logits = model(X[start:start + batch_size])
        predictions.append(logits.argmax(dim=1).to(dtype=torch.int64))
    if not predictions:
        return torch.empty((0,), dtype=torch.int64, device=X.device)
    return torch.cat(predictions, dim=0)


@torch.no_grad()
def evaluate(model, X, y, loss_name: str = "ce", batch_size: int = 8192) -> dict:
    """Trả về dict(loss, acc, macro_f1) ở chế độ eval() (dropout tắt) và no_grad.

    Các bước:
      1. model.eval()
      2. tính logits theo từng lô; cộng dồn tổng loss (reduction="sum") rồi chia N cuối cùng
      3. pred = argmax; acc = (pred == y).mean()
      4. dựng ma trận nhầm lẫn 7x7 -> macro_f1_from_confusion
    Dùng hàm này cho: train loss (trên toàn bộ hoặc một tập con CỐ ĐỊNH của train), val, và eval cuối cùng.
    """
    if loss_name not in ("ce", "mse"):
        raise ValueError("loss_name must be 'ce' or 'mse'")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if len(X) == 0 or len(y) != len(X):
        raise ValueError("evaluation needs a nonempty X and matching y")
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_targets = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            xb = X[start:start + batch_size]
            yb = y[start:start + batch_size].to(dtype=torch.int64)
            logits = model(xb)
            loss = compute_loss(logits, yb, loss_name)
            total_loss += float(loss.item()) * len(xb)
            all_predictions.append(logits.argmax(dim=1).to(dtype=torch.int64))
            all_targets.append(yb)
    pred = torch.cat(all_predictions)
    target = torch.cat(all_targets)
    accuracy = float((pred == target).float().mean().item())
    flat = target * 7 + pred
    cm = torch.bincount(flat, minlength=49).reshape(7, 7).cpu().numpy()
    return {
        "loss": total_loss / len(X),
        "acc": accuracy,
        "macro_f1": macro_f1_from_confusion(cm),
    }


def compute_loss(logits, y, loss_name: str):
    """"ce"  : cross-entropy nhận logit thô và nhãn int64 (F.cross_entropy).
       "mse" : MSE giữa logit và one-hot của y (ghi rõ bạn lấy trung bình thế nào).
    """
    if loss_name == "ce":
        return F.cross_entropy(logits, y.to(dtype=torch.int64))
    if loss_name == "mse":
        targets = F.one_hot(y.to(dtype=torch.int64), num_classes=logits.shape[-1]).to(dtype=logits.dtype)
        # Mean trên cả batch và số lớp.
        return F.mse_loss(logits, targets, reduction="mean")
    raise ValueError("loss_name must be 'ce' or 'mse'")


def run_experiment(cfg: dict, data: dict) -> dict:
    """Huấn luyện một cấu hình và trả về lịch sử + tóm tắt.

    Args:
        cfg : dict cấu hình (xem DEFAULT_CFG)
        data: kết quả của data.prepare_data (tensor X_tr, y_tr, X_val, y_val, X_eval, y_eval trên device)

    Trả về dict:
        {"cfg": cfg,
         "history": {"epoch": [...], "train_loss": [...], "val_loss": [...], "val_acc": [...],
                     "val_macro_f1": [...], "grad_norm": [...], "epoch_time_s": [...]},
         "summary": {"step0_loss", "best_val_loss", "best_epoch", "final_train_loss", "final_val_loss",
                     "val_acc", "val_macro_f1", "time_per_epoch_s", "peak_mem_MB", "diverged"},
         "best_state": state_dict của epoch có val_loss thấp nhất (giữ trong RAM để dự đoán eval)}
    (tên khoá của summary trùng tên cột trong experiments.xlsx)

    Các bước:
      0. set_seed(cfg["seed"]); tạo model = MLP(...), assert count_params(model) == EXPECTED_PARAMS[hidden]
         chuyển model lên device; tạo optimizer = build_optimizer(...)
         nếu precision == "fp16": scaler = torch.amp.GradScaler(...)
      1. step0_loss = evaluate(model, X_val, y_val)["loss"]   # TRƯỚC bước cập nhật đầu tiên; kỳ vọng ≈ ln 7
      2. for epoch in 1..epochs:
           model.train()
           for xb, yb in iterate_batches(X_tr, y_tr, cfg["batch"], generator):
               with torch.autocast(...)  nếu precision != "fp32":   # chỉ bọc forward + loss
                   logits = model(xb); loss = compute_loss(logits, yb, cfg["loss"])
               optimizer.zero_grad(set_to_none=True)
               backward (qua scaler nếu fp16)
               nếu fp16 và có clip: scaler.unscale_(optimizer)  TRƯỚC khi clip
               gn = clip_gradients(model.parameters(), cfg["clip_norm"])   # chuẩn TRƯỚC khi cắt; ghi lại
               bước cập nhật (scaler.step(optimizer); scaler.update() nếu fp16, ngược lại optimizer.step())
               nếu loss là NaN/inf: đặt diverged=True và dừng sớm, ĐỪNG để notebook treo
           cuối epoch (dùng evaluate, chế độ eval):
               train_loss trên toàn bộ train (hoặc 1 tập con CỐ ĐỊNH ~50 000 mẫu), val_loss/val_acc/val_macro_f1
               grad_norm trung bình của epoch; thời gian epoch (torch.cuda.synchronize() nếu dùng GPU)
               nếu val_loss tốt nhất từ trước tới giờ: lưu best_state (bản sao state_dict) và best_epoch
      3. tổng hợp summary tại best_epoch (val_acc, val_macro_f1 lấy ở best_epoch); peak_mem_MB nếu có GPU
    TUYỆT ĐỐI không đưa X_eval vào hàm này để chọn epoch/cấu hình. Chỉ dùng val.
    """
    cfg = {**DEFAULT_CFG, **cfg}
    cfg["hidden"] = tuple(cfg["hidden"])
    if cfg["lr"] is None:
        raise ValueError("cfg['lr'] must be selected using validation data")
    if cfg["epochs"] <= 0 or cfg["batch"] <= 0:
        raise ValueError("epochs and batch must be positive")
    if cfg["loss"] not in ("ce", "mse"):
        raise ValueError("loss must be 'ce' or 'mse'")
    if cfg["precision"] not in ("fp32", "fp16", "bf16"):
        raise ValueError("precision must be 'fp32', 'fp16', or 'bf16'")

    X_tr, y_tr = data["X_tr"], data["y_tr"]
    X_val, y_val = data["X_val"], data["y_val"]
    device = X_tr.device
    if device != y_tr.device or device != X_val.device or device != y_val.device:
        raise ValueError("train and validation tensors must all be on the same device")
    if device.type not in ("cpu", "cuda") and cfg["precision"] != "fp32":
        raise ValueError("autocast is configured here for CPU or CUDA devices")
    if cfg["precision"] == "fp16" and device.type != "cuda":
        raise ValueError("fp16 training requires a CUDA device")

    set_seed(int(cfg["seed"]))
    hidden = cfg["hidden"]
    if hidden not in EXPECTED_PARAMS:
        raise ValueError(f"hidden architecture {hidden} is not listed in EXPECTED_PARAMS")
    model = MLP(hidden=hidden, dropout=cfg["dropout"], init=cfg["init"]).to(device)
    assert count_params(model) == EXPECTED_PARAMS[hidden]
    optimizer = build_optimizer(
        cfg["optimizer"], model.parameters(), lr=float(cfg["lr"]),
        weight_decay=float(cfg["weight_decay"]), momentum=float(cfg["momentum"]),
        betas=tuple(cfg["betas"]), eps=float(cfg["eps"])
    )

    use_scaler = cfg["precision"] == "fp16"
    scaler = torch.amp.GradScaler("cuda", enabled=True) if use_scaler else None
    amp_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(cfg["precision"])
    generator = torch.Generator(device=device)
    generator.manual_seed(int(cfg["seed"]))

    step0_metrics = evaluate(model, X_val, y_val, loss_name=cfg["loss"])
    step0_loss = step0_metrics["loss"]
    best_val_loss = step0_loss
    best_epoch = 0
    best_val_acc = step0_metrics["acc"]
    best_val_macro_f1 = step0_metrics["macro_f1"]
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    history = {
        "epoch": [], "train_loss": [], "val_loss": [], "val_acc": [],
        "val_macro_f1": [], "grad_norm": [], "clip_fraction": [], "epoch_time_s": [],
    }
    diverged = False
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for epoch in range(1, int(cfg["epochs"]) + 1):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        epoch_start = time.perf_counter()
        model.train()
        epoch_grad_norms = []
        epoch_batch_count = 0
        clipped_step_count = 0
        stop_epoch = False

        for xb, yb in iterate_batches(X_tr, y_tr, int(cfg["batch"]), generator=generator, shuffle=True):
            optimizer.zero_grad(set_to_none=True)
            amp_context = (
                nullcontext() if amp_dtype is None
                else torch.autocast(device_type=device.type, dtype=amp_dtype)
            )
            with amp_context:
                logits = model(xb)
                loss = compute_loss(logits, yb, cfg["loss"])
            if not bool(torch.isfinite(loss).item()):
                diverged = True
                stop_epoch = True
                break

            if scaler is None:
                loss.backward()
            else:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
            grad_norm = clip_gradients(model.parameters(), cfg["clip_norm"])
            if not math.isfinite(grad_norm):
                diverged = True
                stop_epoch = True
                break
            epoch_grad_norms.append(grad_norm)
            epoch_batch_count += 1
            if cfg["clip_norm"] is not None and grad_norm > float(cfg["clip_norm"]):
                clipped_step_count += 1
            if scaler is None:
                optimizer.step()
            else:
                scaler.step(optimizer)
                scaler.update()

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        train_metrics = evaluate(model, X_tr, y_tr, loss_name=cfg["loss"])
        val_metrics = evaluate(model, X_val, y_val, loss_name=cfg["loss"])
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - epoch_start
        if not math.isfinite(train_metrics["loss"]) or not math.isfinite(val_metrics["loss"]):
            diverged = True
        history["epoch"].append(epoch)
        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["acc"])
        history["val_macro_f1"].append(val_metrics["macro_f1"])
        history["grad_norm"].append(float(np.mean(epoch_grad_norms)) if epoch_grad_norms else 0.0)
        history["clip_fraction"].append(
            float(clipped_step_count / epoch_batch_count) if epoch_batch_count else 0.0
        )
        history["epoch_time_s"].append(elapsed)

        if math.isfinite(val_metrics["loss"]) and val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            best_val_acc = val_metrics["acc"]
            best_val_macro_f1 = val_metrics["macro_f1"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if stop_epoch or diverged:
            break

    peak_mem_mb = (
        float(torch.cuda.max_memory_allocated(device) / (1024 ** 2))
        if device.type == "cuda" else None
    )
    summary = {
        "step0_loss": float(step0_loss),
        "best_val_loss": float(best_val_loss),
        "best_epoch": int(best_epoch),
        "final_train_loss": float(history["train_loss"][-1]) if history["train_loss"] else float(step0_loss),
        "final_val_loss": float(history["val_loss"][-1]) if history["val_loss"] else float(step0_loss),
        "val_acc": float(best_val_acc),
        "val_macro_f1": float(best_val_macro_f1),
        "time_per_epoch_s": float(np.mean(history["epoch_time_s"])) if history["epoch_time_s"] else 0.0,
        "peak_mem_MB": peak_mem_mb,
        "diverged": bool(diverged),
    }
    return {"cfg": cfg, "history": history, "summary": summary, "best_state": best_state}


def write_predictions(row_id, preds, path: str) -> None:
    """Ghi file nộp cho scripts/evaluate.py: CSV có tiêu đề `row_id,pred`.

    row_id : mảng row_id của tập eval (data["eval_row_id"])
    preds  : nhãn dự đoán int64 0..6 (cùng thứ tự với row_id)
    Phải đủ mọi dòng của tập eval, mỗi row_id đúng một lần.
    """
    row_ids = np.asarray(row_id)
    if torch.is_tensor(preds):
        preds = preds.detach().cpu().numpy()
    predictions = np.asarray(preds)
    if row_ids.ndim != 1 or predictions.ndim != 1 or len(row_ids) != len(predictions):
        raise ValueError("row_id and preds must be one-dimensional arrays of equal length")
    if not np.issubdtype(row_ids.dtype, np.integer) or not np.issubdtype(predictions.dtype, np.integer):
        raise ValueError("row_id and preds must contain integers")
    if len(np.unique(row_ids)) != len(row_ids):
        raise ValueError("row_id values must be unique")
    if np.any((predictions < 0) | (predictions > 6)):
        raise ValueError("predictions must be class ids in 0..6")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(("row_id", "pred"))
        writer.writerows(zip(row_ids.astype(np.int64), predictions.astype(np.int64)))


def final_eval(cfg: dict, result: dict, data: dict, pred_path: str) -> None:
    """Dùng MỘT LẦN cho cấu hình cuối cùng (và baseline): nạp best_state, dự đoán eval, ghi predictions.

    Các bước:
      1. model = MLP(...); model.load_state_dict(result["best_state"]); lên device
      2. preds = predict(model, data["X_eval"])  # fp32, eval mode
      3. write_predictions(data["eval_row_id"], preds.cpu().numpy(), pred_path)
      4. chạy `python scripts/evaluate.py --pred <pred_path>` và ghi kết quả vào bảng/báo cáo
    """
    device = data["X_eval"].device
    model = MLP(hidden=tuple(cfg["hidden"]), dropout=cfg["dropout"], init=cfg["init"])
    model.load_state_dict(result["best_state"])
    model.to(device)
    predictions = predict(model, data["X_eval"]).cpu().numpy()
    write_predictions(data["eval_row_id"], predictions, pred_path)
