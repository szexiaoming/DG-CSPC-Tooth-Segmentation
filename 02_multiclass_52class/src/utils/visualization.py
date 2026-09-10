"""
Visualization utilities for segmentation results.

"""

import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt


def save_prediction_overlay(
    image: np.ndarray,
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    save_path: str,
    title: Optional[str] = None,
    prob_map: Optional[np.ndarray] = None,
) -> None:
    """
    Save a figure showing image, ground truth mask, prediction overlay, and optionally the probability heatmap.

    """
    n_cols = 4 if prob_map is not None else 3
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))

    if n_cols == 3:
        ax1, ax2, ax3 = axes
    else:
        ax1, ax2, ax3, ax4 = axes

    # Original image
    ax1.imshow(image, cmap="gray")
    ax1.set_title("Input X-ray")
    ax1.axis("off")

    # Ground truth
    ax2.imshow(image, cmap="gray")
    ax2.imshow(ground_truth, alpha=0.5, cmap="Reds")
    ax2.set_title("Ground Truth")
    ax2.axis("off")

    # Prediction overlay
    ax3.imshow(image, cmap="gray")
    # True positive = green, False positive = red, False negative = blue
    tp = np.logical_and(prediction > 0.5, ground_truth > 0.5)
    fp = np.logical_and(prediction > 0.5, ground_truth <= 0.5)
    fn = np.logical_and(prediction <= 0.5, ground_truth > 0.5)

    overlay = np.zeros((*image.shape, 3), dtype=np.float32)
    overlay[..., 0] = fp.astype(np.float32)  # Red channel → FP
    overlay[..., 1] = tp.astype(np.float32)  # Green channel → TP
    overlay[..., 2] = fn.astype(np.float32)  # Blue channel → FN

    ax3.imshow(image, cmap="gray")
    ax3.imshow(overlay, alpha=0.4)
    ax3.set_title("Prediction\n(Green=TP, Red=FP, Blue=FN)")
    ax3.axis("off")

    # Probability heatmap (optional)
    if prob_map is not None:
        im = ax4.imshow(prob_map, cmap="hot", vmin=0, vmax=1)
        ax4.set_title("Probability Map")
        ax4.axis("off")
        plt.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)

    if title:
        fig.suptitle(title, fontsize=12)

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_training_curves(
    metrics_history: List[dict],
    save_path: str,
    title: Optional[str] = None,
) -> None:
    """
    Plot and save training/validation curves.

    """
    if not metrics_history:
        return

    epochs = [row.get("epoch", i + 1) for i, row in enumerate(metrics_history)]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Loss curves
    ax = axes[0, 0]
    train_loss = [row.get("train_loss") for row in metrics_history]
    val_loss = [row.get("val_loss") for row in metrics_history]
    ax.plot(epochs, train_loss, "b-", label="Train Loss", linewidth=1.5)
    ax.plot(epochs, val_loss, "r-", label="Val Loss", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss Curves")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Dice curve
    ax = axes[0, 1]
    val_dice = [row.get("val_dice") for row in metrics_history]
    ax.plot(epochs, val_dice, "g-", label="Val Dice", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Dice")
    ax.set_title("Validation Dice")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # IoU curve
    ax = axes[1, 0]
    val_iou = [row.get("val_iou") for row in metrics_history]
    ax.plot(epochs, val_iou, "m-", label="Val IoU", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("IoU")
    ax.set_title("Validation IoU")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Learning rate curve
    ax = axes[1, 1]
    lr = [row.get("learning_rate") for row in metrics_history]
    ax.plot(epochs, lr, "k-", label="LR", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedule")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)

    if title:
        fig.suptitle(title, fontsize=14)

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_comparison_figure(
    results: List[dict],
    save_path: str,
    title: str = "Model Comparison",
    metric_keys: List[str] = ["dice", "iou", "boundary_f1"],
    model_names_key: str = "model",
) -> None:
    """
    Create a bar chart comparing multiple models on given metrics.

    """
    n_metrics = len(metric_keys)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 5))
    if n_metrics == 1:
        axes = [axes]

    model_names = [r[model_names_key] for r in results]
    colors = plt.cm.Set2(np.linspace(0, 1, len(model_names)))
    x = np.arange(len(model_names))

    for i, metric in enumerate(metric_keys):
        ax = axes[i]
        values = [r.get(metric, 0) for r in results]
        bars = ax.bar(x, values, color=colors, edgecolor="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(metric.upper())

        # Add value labels on bars
        for bar, val in zip(bars, values):
            if val is not None and val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{val:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

        ax.set_ylim(0, 1.0)

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
