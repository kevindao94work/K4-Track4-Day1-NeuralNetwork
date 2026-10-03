"""Plot experiment metrics and gradient norms.

Ảnh biểu đồ là sản phẩm nộp (xem README mục 6): mỗi thí nghiệm một ảnh figures/<exp_id>.png.
Khi notebook chạy trong code/, lưu vào "../figures/" (ví dụ path = f"../figures/{exp_id}.png").
"""
from __future__ import annotations

import matplotlib.pyplot as plt
from pathlib import Path


def plot_run(result: dict, path: str) -> None:
    """Vẽ MỘT thí nghiệm thành một ảnh PNG có ít nhất 3 ô:
         (1) train_loss và val_loss theo epoch (cùng một trục)
         (2) val_acc (và nên có val_macro_f1) theo epoch
         (3) grad_norm theo epoch (đo TRƯỚC khi clip)
    Yêu cầu: tiêu đề ghi exp_id và cấu hình chính (optimizer, lr, batch, ...), có nhãn trục và chú thích.
    Các bước: fig, axes = plt.subplots(1, 3, figsize=...); plot; set_title/xlabel/legend;
              fig.savefig(path, dpi=..., bbox_inches="tight"); plt.close(fig)
    Gợi ý: đánh dấu best_epoch bằng đường thẳng đứng.
    """
    cfg = result.get("cfg", {})
    summary = result.get("summary", {})
    history = result.get("history", {})
    epochs = history.get("epoch", [])
    if not epochs:
        epochs = list(range(1, len(history.get("train_loss", [])) + 1))
    exp_id = cfg.get("exp_id", "experiment")
    title_cfg = (
        f"{cfg.get('optimizer', '?')} | lr={cfg.get('lr', '?')} | "
        f"batch={cfg.get('batch', '?')} | init={cfg.get('init', '?')}"
    )

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    axes[0].plot(epochs, history.get("train_loss", []), label="train loss")
    axes[0].plot(epochs, history.get("val_loss", []), label="val loss")
    axes[0].set(title="Loss", xlabel="Epoch", ylabel="Loss")
    axes[0].legend()

    axes[1].plot(epochs, history.get("val_acc", []), label="val accuracy")
    if history.get("val_macro_f1"):
        axes[1].plot(epochs, history["val_macro_f1"], label="val macro-F1")
    axes[1].set(title="Validation metrics", xlabel="Epoch", ylabel="Score")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].legend()

    axes[2].plot(epochs, history.get("grad_norm", []), label="grad norm (before clip)")
    axes[2].set(title="Gradient norm", xlabel="Epoch", ylabel="L2 norm")
    axes[2].legend()

    best_epoch = summary.get("best_epoch")
    if best_epoch is not None:
        for axis in axes:
            axis.axvline(best_epoch, color="gray", linestyle="--", linewidth=1, label="best epoch")
    fig.suptitle(f"{exp_id} — {title_cfg}")
    fig.tight_layout()
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_compare(results: list[dict], metric: str, path: str, title: str = "") -> None:
    """Vẽ chồng một chỉ số (ví dụ "val_loss", "val_macro_f1", "grad_norm") của nhiều thí nghiệm
    trên cùng một trục, mỗi thí nghiệm một đường, chú thích bằng exp_id.

    Dùng cho ảnh figures/compare_<nhóm>.png (ví dụ compare_optimizer.png).
    """
    if not results:
        raise ValueError("results must contain at least one experiment")
    fig, ax = plt.subplots(figsize=(8, 5))
    for result in results:
        history = result.get("history", {})
        values = history.get(metric)
        if values is None:
            raise KeyError(f"metric {metric!r} not found in experiment history")
        epochs = history.get("epoch", list(range(1, len(values) + 1)))
        ax.plot(epochs, values, label=result.get("cfg", {}).get("exp_id", "experiment"))
    ax.set(title=title or metric.replace("_", " "), xlabel="Epoch", ylabel=metric.replace("_", " "))
    ax.legend()
    fig.tight_layout()
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)
