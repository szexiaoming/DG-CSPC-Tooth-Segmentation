"""
Training script for tooth segmentation models.

Supports YAML-based configuration. Reads data from processed directory,
trains a model, saves checkpoints and logs.

"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.datasets import ToothSegmentationDataset, get_transforms_train, get_transforms_val
from src.models import create_model
from src.losses import build_loss_from_config
from src.utils import ExperimentLogger, set_seed, format_metrics, compute_all_metrics
from src.utils.model_summary import get_model_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a tooth segmentation model."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "cuda:0", "cuda:1"],
        help="Device to train on (default: cpu).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing experiment directory.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from.",
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def validate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion: torch.nn.Module,
    device: str,
) -> Dict[str, float]:
    """
    Run validation on the entire validation set.

    Returns metrics averaged over all samples.
    """
    model.eval()
    total_loss = 0.0
    all_metrics = {"dice": [], "iou": [], "boundary_f1": [], "precision": [], "recall": []}
    total_bce = 0.0
    total_dice = 0.0
    total_boundary = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            # Forward
            logits = model(images)  # (N, 1, H, W)

            # Compute loss
            loss_dict = criterion(logits, masks)
            total_loss += loss_dict["loss"].item()
            if loss_dict.get("bce") is not None:
                total_bce += loss_dict["bce"]
            if loss_dict.get("dice") is not None:
                total_dice += loss_dict["dice"]
            if loss_dict.get("boundary") is not None:
                total_boundary += loss_dict["boundary"]
            n_batches += 1

            # Compute per-sample metrics
            probs = torch.sigmoid(logits)
            for i in range(probs.shape[0]):
                pred_np = probs[i, 0].cpu().numpy()
                mask_np = masks[i, 0].cpu().numpy()
                m = compute_all_metrics(pred_np, mask_np)
                for k in all_metrics:
                    all_metrics[k].append(m[k])

    # Average
    avg_loss = total_loss / max(n_batches, 1)
    avg_bce = total_bce / max(n_batches, 1) if total_bce > 0 else None
    avg_dice_loss = total_dice / max(n_batches, 1) if total_dice > 0 else None
    avg_boundary = total_boundary / max(n_batches, 1) if total_boundary > 0 else None

    return {
        "loss": avg_loss,
        "bce": avg_bce,
        "dice_loss": avg_dice_loss,
        "boundary": avg_boundary,
        "dice": float(np.mean(all_metrics["dice"])),
        "iou": float(np.mean(all_metrics["iou"])),
        "boundary_f1": float(np.mean(all_metrics["boundary_f1"])),
        "precision": float(np.mean(all_metrics["precision"])),
        "recall": float(np.mean(all_metrics["recall"])),
    }


def train_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
    epoch: int,
    total_epochs: int,
) -> Dict[str, float]:
    """
    Train for one epoch.

    Returns averaged training metrics.
    """
    model.train()
    total_loss = 0.0
    total_bce = 0.0
    total_dice = 0.0
    total_boundary = 0.0
    n_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{total_epochs}", leave=False)
    for batch in pbar:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        optimizer.zero_grad()

        logits = model(images)
        loss_dict = criterion(logits, masks)

        loss = loss_dict["loss"]
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # Accumulate
        total_loss += loss.item()
        if loss_dict.get("bce") is not None:
            total_bce += loss_dict["bce"]
        if loss_dict.get("dice") is not None:
            total_dice += loss_dict["dice"]
        if loss_dict.get("boundary") is not None:
            total_boundary += loss_dict["boundary"]
        n_batches += 1

        # Update progress bar
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    n = max(n_batches, 1)
    return {
        "loss": total_loss / n,
        "bce": total_bce / n if total_bce > 0 else None,
        "dice": total_dice / n if total_dice > 0 else None,
        "boundary": total_boundary / n if total_boundary > 0 else None,
    }


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_dice: float,
    save_path: str,
    is_best: bool = False,
) -> None:
    """
    Save a model checkpoint.
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_dice": val_dice,
    }
    torch.save(checkpoint, save_path)
    if is_best:
        best_path = str(Path(save_path).parent / "best_model.pth")
        torch.save(checkpoint, best_path)


def build_optimizer(model: torch.nn.Module, config: dict) -> torch.optim.Optimizer:
    """Build optimizer from config."""
    train_cfg = config.get("training", {})
    opt_name = train_cfg.get("optimizer", "adamw").lower()
    lr = train_cfg.get("learning_rate", 0.0001)
    wd = train_cfg.get("weight_decay", 0.00001)

    if opt_name == "adam":
        return optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    elif opt_name == "adamw":
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    elif opt_name == "sgd":
        return optim.SGD(
            model.parameters(), lr=lr, weight_decay=wd, momentum=0.9
        )
    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")


def build_scheduler(
    optimizer: torch.optim.Optimizer, config: dict, epochs: int
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """Build learning rate scheduler from config."""
    train_cfg = config.get("training", {})
    sched_name = train_cfg.get("scheduler", "cosine").lower()

    if sched_name == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif sched_name == "step":
        step_size = train_cfg.get("scheduler_step_size", 30)
        gamma = train_cfg.get("scheduler_gamma", 0.1)
        return optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif sched_name == "plateau":
        patience = train_cfg.get("scheduler_patience", 10)
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", patience=patience, factor=0.5
        )
    elif sched_name == "none":
        return None
    else:
        raise ValueError(f"Unknown scheduler: {sched_name}")


def main() -> None:
    args = parse_args()

    # Load config
    config = load_config(args.config)
    print(f"Loaded config: {args.config}")

    # Set seed
    seed = config.get("seed", 42)
    set_seed(seed)

    # Setup device
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU.")
        device = "cpu"
    print(f"Device: {device}")

    # Setup experiment directory
    output_cfg = config.get("output", {})
    save_dir = output_cfg.get("save_dir", "experiments/default")
    experiment_name = config.get("experiment_name", "default")

    if args.overwrite and Path(save_dir).exists():
        import shutil
        shutil.rmtree(save_dir)

    logger = ExperimentLogger(save_dir, experiment_name)
    logger.save_config(config)

    # Data config
    data_cfg = config.get("data", {})
    image_size = tuple(data_cfg.get("image_size", [512, 512]))  # (H, W)

    # Create datasets
    train_transform = get_transforms_train(image_size)
    val_transform = get_transforms_val(image_size)

    train_dataset = ToothSegmentationDataset(
        image_dir=data_cfg.get("image_dir", "data/processed/images"),
        mask_dir=data_cfg.get("mask_dir", "data/processed/masks"),
        split_file=data_cfg.get("train_split", "data/splits/train.txt"),
        transform=train_transform,
        image_size=image_size,
    )
    val_dataset = ToothSegmentationDataset(
        image_dir=data_cfg.get("image_dir", "data/processed/images"),
        mask_dir=data_cfg.get("mask_dir", "data/processed/masks"),
        split_file=data_cfg.get("val_split", "data/splits/val.txt"),
        transform=val_transform,
        image_size=image_size,
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples:   {len(val_dataset)}")

    # Create dataloaders
    train_cfg = config.get("training", {})
    batch_size = train_cfg.get("batch_size", 2)
    num_workers = train_cfg.get("num_workers", 0)  # 0 for CPU safety

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,  # Validation uses batch_size=1 for per-sample metrics
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    # Create model
    model_cfg = config.get("model", {})
    model = create_model(
        model_name=model_cfg.get("name", "unet"),
        encoder_name=model_cfg.get("encoder", "resnet34"),
        in_channels=model_cfg.get("in_channels", 1),
        num_classes=model_cfg.get("num_classes", 1),
        pretrained=model_cfg.get("pretrained", True),
    )
    model.to(device)

    # Print model summary
    summary = get_model_summary(model, input_shape=(1, 1, image_size[0], image_size[1]), device=device)
    print(f"\nModel Summary:")
    print(f"  Parameters: {summary['trainable_params']:,} trainable / {summary['total_params']:,} total")
    print(f"  Model size: {summary['model_size_mb']:.2f} MB")
    print(f"  Inference:  {summary['inference_time_ms']:.2f} ms/image ({summary['fps']:.1f} FPS) on {device}\n")

    # Create loss
    loss_cfg = config.get("loss", {})
    criterion = build_loss_from_config(loss_cfg)

    # Create optimizer & scheduler
    optimizer = build_optimizer(model, config)
    epochs = train_cfg.get("epochs", 100)
    scheduler = build_scheduler(optimizer, config, epochs)

    # Resume if specified
    start_epoch = 1
    best_val_dice = 0.0
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_dice = checkpoint.get("val_dice", 0.0)

    # Early stopping
    early_stop_patience = train_cfg.get("early_stopping_patience", 15)
    epochs_no_improve = 0

    # Training loop
    print(f"\nStarting training: {epochs} epochs (from epoch {start_epoch})")
    print(f"Early stopping patience: {early_stop_patience}")
    print(f"Save directory: {save_dir}\n")

    total_start = time.time()

    for epoch in range(start_epoch, epochs + 1):
        epoch_start = time.time()

        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, epochs
        )

        # Validate
        val_metrics = validate(model, val_loader, criterion, device)

        # Get current learning rate
        current_lr = optimizer.param_groups[0]["lr"]

        # Log epoch
        logger.log_epoch(epoch, train_metrics, val_metrics, current_lr)

        # Scheduler step
        if scheduler is not None:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_metrics["dice"])
            else:
                scheduler.step()

        # Save checkpoint
        save_best_only = output_cfg.get("save_best_only", True)
        current_val_dice = val_metrics["dice"]
        is_best = current_val_dice > best_val_dice

        if is_best:
            best_val_dice = current_val_dice
            epochs_no_improve = 0
            save_checkpoint(
                model, optimizer, epoch, best_val_dice,
                str(Path(save_dir) / f"checkpoint_epoch_{epoch:03d}.pth"),
                is_best=True,
            )
            
        else:
            epochs_no_improve += 1
            if not save_best_only:
                save_checkpoint(
                    model, optimizer, epoch, current_val_dice,
                    str(Path(save_dir) / f"checkpoint_epoch_{epoch:03d}.pth"),
                )

        # Print progress
        epoch_time = time.time() - epoch_start
        print(
            f"Epoch {epoch:3d}/{epochs} | "
            f"Train Loss: {train_metrics['loss']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val Dice: {val_metrics['dice']:.4f} | "
            f"Val IoU: {val_metrics['iou']:.4f} | "
            f"Val BF1: {val_metrics['boundary_f1']:.4f} | "
            f"LR: {current_lr:.2e} | "
            f"Time: {epoch_time:.1f}s"
        )

        # Early stopping check
        if epochs_no_improve >= early_stop_patience:
            print(
                f"\nEarly stopping triggered after {early_stop_patience} epochs "
                f"without improvement. Best val Dice: {best_val_dice:.4f}"
            )
            break

    total_time = time.time() - total_start
    print(f"\nTraining complete! Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Best validation Dice: {best_val_dice:.4f}")
    print(f"Results saved to: {save_dir}")

    # Save final summary
    logger.save_summary(
        {
            "total_time_seconds": total_time,
            "best_val_dice": best_val_dice,
            "device": device,
            **summary,
        }
    )


if __name__ == "__main__":
    main()
