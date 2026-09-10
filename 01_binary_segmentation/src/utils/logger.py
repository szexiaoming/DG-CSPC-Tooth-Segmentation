"""
Experiment logger for saving training metrics and checkpoints.

Logs metrics to CSV and optionally prints to console with formatting.
"""

import csv
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


class ExperimentLogger:
    """
    Handles logging of training and validation metrics to CSV files.

    Also saves a JSON summary of the experiment configuration and results.

    """

    def __init__(
        self,
        save_dir: str,
        experiment_name: str,
        resume: bool = False,
    ) -> None:
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_name = experiment_name

        self.csv_path = self.save_dir / "metrics.csv"
        self.config_path = self.save_dir / "config.json"
        self.summary_path = self.save_dir / "summary.json"

        self._epoch = 0
        self._best_val_loss = float("inf")
        self._best_val_dice = 0.0
        self._metrics_history: List[Dict[str, Any]] = []

        # Only create a new CSV if it doesn't exist (never overwrite)
        if not self.csv_path.exists():
            self._init_csv()
        elif resume:
            # Load existing metrics history on resume
            self._load_existing()

    def _init_csv(self) -> None:
        """Create CSV file with header."""
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "epoch",
                    "train_loss",
                    "train_bce",
                    "train_dice_loss",
                    "train_boundary_loss",
                    "val_loss",
                    "val_dice",
                    "val_iou",
                    "val_boundary_f1",
                    "val_precision",
                    "val_recall",
                    "learning_rate",
                    "timestamp",
                ]
            )

    def _load_existing(self) -> None:
        """Load existing metrics from CSV to restore tracking state."""
        if not self.csv_path.exists():
            return
        with open(self.csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                r = {}
                for k, v in row.items():
                    if v == "" or v is None:
                        r[k] = None
                    else:
                        try:
                            r[k] = float(v)
                        except ValueError:
                            r[k] = v
                self._metrics_history.append(r)
                # Restore best tracking
                if r.get("val_dice") is not None and r["val_dice"] > self._best_val_dice:
                    self._best_val_dice = r["val_dice"]
                if r.get("val_loss") is not None and r["val_loss"] < self._best_val_loss:
                    self._best_val_loss = r["val_loss"]
                self._epoch = max(self._epoch, int(r.get("epoch", 0)))

    def log_epoch(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        learning_rate: float,
    ) -> None:
        """
        Log metrics for one epoch.

        """
        self._epoch = epoch

        row = {
            "epoch": epoch,
            "train_loss": train_metrics.get("loss", None),
            "train_bce": train_metrics.get("bce", None),
            "train_dice_loss": train_metrics.get("dice", None),
            "train_boundary_loss": train_metrics.get("boundary", None),
            "val_loss": val_metrics.get("loss", None),
            "val_dice": val_metrics.get("dice", None),
            "val_iou": val_metrics.get("iou", None),
            "val_boundary_f1": val_metrics.get("boundary_f1", None),
            "val_precision": val_metrics.get("precision", None),
            "val_recall": val_metrics.get("recall", None),
            "learning_rate": learning_rate,
            "timestamp": datetime.now().isoformat(),
        }

        self._metrics_history.append(row)

        # Append to CSV
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([row.get(k) for k in [
                "epoch", "train_loss", "train_bce", "train_dice_loss",
                "train_boundary_loss", "val_loss", "val_dice", "val_iou",
                "val_boundary_f1", "val_precision", "val_recall",
                "learning_rate", "timestamp",
            ]])

        # Track best metrics
        val_loss = val_metrics.get("loss", float("inf"))
        val_dice = val_metrics.get("dice", 0.0)
        if val_loss is not None and val_loss < self._best_val_loss:
            self._best_val_loss = val_loss
        if val_dice is not None and val_dice > self._best_val_dice:
            self._best_val_dice = val_dice

    def save_config(self, config: Dict[str, Any]) -> None:
        """
        Save experiment configuration as JSON.
        
        """
        with open(self.config_path, "w") as f:
            json.dump(config, f, indent=2, default=str)

    def save_summary(self, summary: Dict[str, Any]) -> None:
        """Save final experiment summary."""
        summary.update(
            {
                "experiment_name": self.experiment_name,
                "total_epochs": self._epoch,
                "best_val_loss": self._best_val_loss,
                "best_val_dice": self._best_val_dice,
            }
        )
        with open(self.summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)

    @property
    def best_val_loss(self) -> float:
        return self._best_val_loss

    @property
    def best_val_dice(self) -> float:
        return self._best_val_dice


def format_metrics(metrics: Dict[str, Any], prefix: str = "") -> str:
    """
    Format a metrics dict as a human-readable string.

    """
    parts = []
    for k, v in metrics.items():
        if v is not None:
            if isinstance(v, float):
                parts.append(f"{k}={v:.4f}")
            else:
                parts.append(f"{k}={v}")
    result = " ".join(parts)
    if prefix:
        result = f"{prefix} {result}"
    return result
