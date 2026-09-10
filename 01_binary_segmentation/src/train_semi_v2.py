"""
Exploratory Semi-supervised Training (v2).

Key differences from train_semi.py:
1. Uses teacher probability maps instead of hard binary pseudo-masks.
2. Applies pixel-wise weighting based on teacher prediction uncertainty.
3. Uses a warm-up and ramp-up schedule for the pseudo-label loss weight.
4. Uses all available pseudo-label samples (no pseudo_ratio subsampling).
5. Uses a lower boundary-loss weight (0.02 vs 0.10).

Note: this exploratory SSL v2 experiment is not included in the final dissertation.
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
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from tqdm import tqdm
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.datasets import ToothSegmentationDataset, SoftPseudoDataset, get_transforms_train, get_transforms_val
from src.models import create_model
from src.losses import build_loss_from_config
from src.utils import ExperimentLogger, set_seed, format_metrics, compute_all_metrics
from src.utils.model_summary import get_model_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Improved Semi-supervised training v2.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--overwrite", action="store_true",
                        help="Delete existing experiment directory and start fresh.")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from checkpoint in save_dir.")
    parser.add_argument("--prob_dir", type=str, default="data/pseudo_probs",
                        help="Directory with soft pseudo-label .npy files.")
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def compute_confidence_map(soft_target: torch.Tensor) -> torch.Tensor:
    """
    Compute per-pixel confidence from soft target.
    c = 1 - 2 * |p - 0.5|  -> high at extremes, low at uncertainty.
    """
    return 1.0 - 2.0 * torch.abs(soft_target - 0.5)


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_metrics = {"dice": [], "iou": [], "boundary_f1": [], "precision": [], "recall": []}
    n_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            logits = model(images)
            loss_dict = criterion(logits, masks)
            total_loss += loss_dict["loss"].item()
            n_batches += 1
            probs = torch.sigmoid(logits)
            for i in range(probs.shape[0]):
                m = compute_all_metrics(probs[i, 0].cpu().numpy(), masks[i, 0].cpu().numpy())
                for k in all_metrics:
                    all_metrics[k].append(m[k])

    n = max(n_batches, 1)
    return {
        "loss": total_loss / n,
        "dice": float(np.mean(all_metrics["dice"])),
        "iou": float(np.mean(all_metrics["iou"])),
        "boundary_f1": float(np.mean(all_metrics["boundary_f1"])),
        "precision": float(np.mean(all_metrics["precision"])),
        "recall": float(np.mean(all_metrics["recall"])),
    }


def train_epoch_v2(
    model, labeled_loader, pseudo_loader, criterion, optimizer, device,
    epoch, total_epochs, pseudo_weight: float,
):
    """
    Train one epoch with both labeled and pseudo-labeled data.
    Pseudo-labeled data uses confidence-weighted loss.

    """
    model.train()
    total_labeled_loss = 0.0
    total_pseudo_loss = 0.0
    n_labeled = 0
    n_pseudo = 0

    # Create iterators
    pseudo_iter = iter(pseudo_loader) if pseudo_loader is not None else None

    pbar = tqdm(labeled_loader, desc=f"Epoch {epoch}/{total_epochs}", leave=False)
    for batch in pbar:
        # Labeled batch (standard loss)
        images_l = batch["image"].to(device)
        masks_l = batch["mask"].to(device)
        optimizer.zero_grad()
        logits_l = model(images_l)
        loss_l = criterion(logits_l, masks_l)["loss"]
        total_labeled_loss += loss_l.item()
        n_labeled += 1

        # Pseudo-labeled batch (confidence-weighted loss)
        if pseudo_iter is not None and pseudo_weight > 0:
            try:
                batch_p = next(pseudo_iter)
            except StopIteration:
                pseudo_iter = iter(pseudo_loader)
                batch_p = next(pseudo_iter)

            images_p = batch_p["image"].to(device)
            soft_target_p = batch_p["soft_target"].to(device)
            confidence_p = batch_p["confidence"].to(device)

            logits_p = model(images_p)
            loss_p = criterion(logits_p, soft_target_p, confidence_p)["loss"]
            total_pseudo_loss += loss_p.item()
            n_pseudo += 1

            # Combined loss
            total_loss = loss_l + pseudo_weight * loss_p
        else:
            total_loss = loss_l

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        pbar.set_postfix({
            "l_loss": f"{loss_l.item():.4f}",
            "p_loss": f"{loss_p.item():.4f}" if pseudo_weight > 0 else "N/A",
            "p_w": f"{pseudo_weight:.2f}",
        })

    result = {"labeled_loss": total_labeled_loss / max(n_labeled, 1)}
    if n_pseudo > 0:
        result["pseudo_loss"] = total_pseudo_loss / n_pseudo
    return result


def get_pseudo_weight(epoch: int, config: dict) -> float:
    """
    Compute current pseudo-label weight using warmup + ramp schedule.

    Phase 1 (warmup): epochs 1..warmup_epochs -> weight = 0 (labeled only)
    Phase 2 (ramp):   warmup_epochs+1 .. warmup_epochs+ramp_epochs -> linear 0->max
    Phase 3 (full):   rest -> weight = max_weight
    """
    ssl_cfg = config.get("ssl", {})
    warmup = ssl_cfg.get("warmup_epochs", 5)
    ramp = ssl_cfg.get("ramp_epochs", 10)
    max_w = ssl_cfg.get("max_pseudo_weight", 1.0)

    if epoch <= warmup:
        return 0.0
    elif epoch <= warmup + ramp:
        return max_w * (epoch - warmup) / ramp
    else:
        return max_w


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(config.get("seed", 42))

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    print(f"Device: {device}")

    output_cfg = config.get("output", {})
    save_dir = output_cfg.get("save_dir", "experiments/ours_lightweight_ssl_v2")
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        import shutil
        shutil.rmtree(save_dir)
        Path(save_dir).mkdir(parents=True, exist_ok=True)

    logger = ExperimentLogger(save_dir, config.get("experiment_name", "ours_ssl_v2"),
                              resume=args.resume)
    logger.save_config(config)

    data_cfg = config.get("data", {})
    image_size = tuple(data_cfg.get("image_size", [512, 512]))
    image_dir = data_cfg.get("image_dir", "data/processed/images")
    mask_dir = data_cfg.get("mask_dir", "data/processed/masks")

    train_cfg = config.get("training", {})
    batch_size = train_cfg.get("batch_size", 2)
    num_workers = train_cfg.get("num_workers", 0)

    ssl_cfg = config.get("ssl", {})

    # Labeled dataset
    labeled_ds = ToothSegmentationDataset(
        image_dir=image_dir, mask_dir=mask_dir,
        split_file=data_cfg.get("train_split", "data/splits/train.txt"),
        transform=get_transforms_train(image_size), image_size=image_size,
    )
    print(f"Labeled samples: {len(labeled_ds)}")

    # Soft pseudo-labeled dataset
    prob_dir = Path(args.prob_dir)
    # Exclude images already in train/val/test
    labeled_names = set()
    for split_file in [
        data_cfg.get("train_split", "data/splits/train.txt"),
        data_cfg.get("val_split", "data/splits/val.txt"),
        data_cfg.get("test_split", "data/splits/test.txt"),
    ]:
        with open(split_file, "r", encoding="utf-8") as f:
            labeled_names |= set(line.strip() for line in f if line.strip())

    # Find valid pseudo-labeled samples
    prob_files = set(f.stem for f in prob_dir.glob("*.npy"))
    img_files = set(f.stem for f in Path(image_dir).glob("*.png"))
    valid_names = sorted(prob_files & img_files - labeled_names)
    print(f"Soft pseudo-labeled samples: {len(valid_names)}")

    # Limit if configured
    max_pseudo = ssl_cfg.get("max_pseudo_samples", 0)
    if max_pseudo > 0 and len(valid_names) > max_pseudo:
        np.random.seed(config.get("seed", 42))
        valid_names = np.random.choice(valid_names, max_pseudo, replace=False).tolist()
        print(f"  Limited to {len(valid_names)} pseudo samples")

    pseudo_ds = SoftPseudoDataset(
        image_dir=image_dir, prob_dir=str(prob_dir),
        image_names=valid_names,
        transform=get_transforms_train(image_size),
        image_size=image_size,
    )

    # Data loaders
    labeled_loader = DataLoader(labeled_ds, batch_size=batch_size, shuffle=True,
                                 num_workers=num_workers, drop_last=True)
    pseudo_loader = DataLoader(pseudo_ds, batch_size=batch_size, shuffle=True,
                                num_workers=num_workers, drop_last=True)

    # Validation dataset
    val_ds = ToothSegmentationDataset(
        image_dir=image_dir, mask_dir=mask_dir,
        split_file=data_cfg.get("val_split", "data/splits/val.txt"),
        transform=get_transforms_val(image_size), image_size=image_size,
    )
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=num_workers)
    print(f"Val samples: {len(val_ds)}")

    # Model
    model_cfg = config.get("model", {})
    model = create_model(
        model_name=model_cfg.get("name", "unet"),
        encoder_name=model_cfg.get("encoder", "mobilenet_v2"),
        in_channels=model_cfg.get("in_channels", 1),
        num_classes=model_cfg.get("num_classes", 1),
        pretrained=True,
    )
    model.to(device)
    summary = get_model_summary(model, (1, 1, image_size[0], image_size[1]), device)
    print(f"Params: {summary['trainable_params']:,}")

    # Loss & optimizer
    loss_cfg = config.get("loss", {})
    criterion = build_loss_from_config(loss_cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.get("learning_rate", 0.0002),
                                   weight_decay=train_cfg.get("weight_decay", 0.00001))
    epochs = train_cfg.get("epochs", 50)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Training loop
    start_epoch = 1
    best_val_dice = 0.0
    epochs_no_improve = 0
    early_stop_patience = train_cfg.get("early_stopping_patience", 15)

    checkpoint_path = Path(save_dir) / "best_model.pth"
    if args.resume and checkpoint_path.exists():
        print(f"Resuming from checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val_dice = ckpt.get("val_dice", 0.0)
        print(f"Resumed from epoch {ckpt['epoch']}, best Val Dice: {best_val_dice:.4f}")

    print(f"\nStarting improved SSL training: epochs {start_epoch}-{epochs}")
    print(f"  Warmup: epochs 1-{ssl_cfg.get('warmup_epochs', 5)} (labeled only)")
    print(f"  Ramp:   epochs {ssl_cfg.get('warmup_epochs', 5)+1}-{ssl_cfg.get('warmup_epochs', 5)+ssl_cfg.get('ramp_epochs', 10)} (gradual)")
    print(f"  Full:   epochs {ssl_cfg.get('warmup_epochs', 5)+ssl_cfg.get('ramp_epochs', 10)+1}-{epochs}")
    total_start = time.time()

    for epoch in range(start_epoch, epochs + 1):
        epoch_start = time.time()

        pseudo_w = get_pseudo_weight(epoch, config)
        train_metrics = train_epoch_v2(
            model, labeled_loader, pseudo_loader, criterion, optimizer, device,
            epoch, epochs, pseudo_w,
        )
        val_metrics = validate(model, val_loader, criterion, device)
        current_lr = optimizer.param_groups[0]["lr"]

        # Log combined loss
        log_metrics = {"loss": train_metrics.get("labeled_loss", 0)}
        logger.log_epoch(epoch, log_metrics, val_metrics, current_lr)
        scheduler.step()

        if val_metrics["dice"] > best_val_dice:
            best_val_dice = val_metrics["dice"]
            epochs_no_improve = 0
            checkpoint = {
                "epoch": epoch, "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_dice": best_val_dice,
            }
            torch.save(checkpoint, checkpoint_path)
        else:
            epochs_no_improve += 1

        epoch_time = time.time() - epoch_start
        pl = train_metrics.get("pseudo_loss", 0)
        print(f"Epoch {epoch:3d}/{epochs} | Label Loss: {train_metrics['labeled_loss']:.4f} | "
              f"Pseudo Loss: {pl:.4f} (w={pseudo_w:.2f}) | "
              f"Val Dice: {val_metrics['dice']:.4f} | Val IoU: {val_metrics['iou']:.4f} | "
              f"LR: {current_lr:.2e} | Time: {epoch_time:.1f}s")

        if epochs_no_improve >= early_stop_patience:
            print(f"Early stopping at epoch {epoch}")
            break

    total_time = time.time() - total_start
    print(f"\nImproved SSL training complete! {total_time:.0f}s ({total_time/60:.1f} min)")
    print(f"Best Val Dice: {best_val_dice:.4f}")
    print(f"Results saved to: {save_dir}")


if __name__ == "__main__":
    main()
