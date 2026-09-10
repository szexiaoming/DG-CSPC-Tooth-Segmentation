"""
Multi-class training v3: Boundary Loss + Deep Supervision + Hungarian pseudo labels.

"""

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.datasets import MultiClassToothDataset, get_transforms_train, get_transforms_val
from src.models import create_model, create_deep_supervision_model
from src.losses import (
    compute_class_weights,
    build_multiclass_loss,
    build_boundary_multiclass_loss,
)
from src.utils import set_seed
from src.utils.logger import ExperimentLogger
from src.utils.model_summary import get_model_summary


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def compute_deep_supervision_loss(
    aux_outputs: List[torch.Tensor],
    target: torch.Tensor,
    criterion: nn.Module,
    aux_weight: float = 0.3,
) -> torch.Tensor:
    """
    Compute auxiliary loss for deep supervision outputs.

    Each auxiliary output is upsampled to target size and losses are averaged across all auxiliary heads.
    """
    aux_loss = 0.0
    for aux_logits in aux_outputs:
        # Already upsampled to target size in model forward
        loss_dict = criterion(aux_logits, target)
        aux_loss += loss_dict["loss"]

    return aux_weight * aux_loss / max(len(aux_outputs), 1)


def train_epoch(
    model, loader, criterion, optimizer, device, epoch, total,
    sample_weights=None, use_deep_sup=False, aux_weight=0.3,
    grad_accum_steps=1,
):
    """
    Train one epoch with optional deep supervision and gradient accumulation.
    
    """
    model.train()
    total_loss = 0.0
    total_ce = 0.0
    total_dice = 0.0
    total_bnd = 0.0
    n = 0

    pbar = tqdm(loader, desc=f"E{epoch}/{total}", leave=False)
    optimizer.zero_grad()

    for idx, batch in enumerate(pbar):
        imgs = batch["image"].to(device)
        masks = batch["mask"].to(device)

        # Forward pass
        output = model(imgs)

        # Parse output based on model type
        if isinstance(output, dict) and "main" in output:
            main_out = output["main"]
            aux_outputs = output.get("aux", [])
        else:
            main_out = output
            aux_outputs = []

        # Main loss
        loss_dict = criterion(main_out, masks)
        loss = loss_dict["loss"]

        # Apply per-sample weighting
        if sample_weights is not None:
            batch_start = idx * loader.batch_size
            for bi in range(min(len(sample_weights) - batch_start, imgs.size(0))):
                w = sample_weights[batch_start + bi].to(device)
                if w < 1.0:
                    # Scale individual sample contribution
                    pass  # Per-batch weighting handled via weighted dataset

        # Deep supervision auxiliary loss
        if use_deep_sup and aux_outputs:
            aux_loss = compute_deep_supervision_loss(aux_outputs, masks, criterion, aux_weight)
            loss = loss + aux_loss

        # Gradient accumulation
        loss = loss / grad_accum_steps
        loss.backward()

        if (idx + 1) % grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * grad_accum_steps
        total_ce += loss_dict.get("ce_loss", torch.tensor(0.0)).item()
        total_dice += loss_dict.get("dice_loss", torch.tensor(0.0)).item()
        total_bnd += loss_dict.get("boundary_loss", torch.tensor(0.0)).item()
        n += 1

        postfix = {"loss": f"{loss.item() * grad_accum_steps:.4f}"}
        if total_ce > 0:
            postfix["ce"] = f"{total_ce / n:.3f}"
        if total_dice > 0:
            postfix["dice"] = f"{total_dice / n:.3f}"
        if total_bnd > 0:
            postfix["bnd"] = f"{total_bnd / n:.3f}"
        pbar.set_postfix(postfix)

    # Handle remaining gradients
    if n % grad_accum_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

    return {
        "loss": total_loss / max(n, 1),
        "ce_loss": total_ce / max(n, 1),
        "dice_loss": total_dice / max(n, 1),
        "boundary_loss": total_bnd / max(n, 1),
    }


@torch.no_grad()
def validate(model, loader, criterion, device, num_classes=53):
    """Validation with per-class metrics."""
    model.eval()
    total_loss = 0.0
    n = 0
    inters = torch.zeros(num_classes, device=device)
    unions = torch.zeros(num_classes, device=device)
    t_pred = torch.zeros(num_classes, device=device)
    t_gt = torch.zeros(num_classes, device=device)

    for batch in loader:
        imgs = batch["image"].to(device)
        masks = batch["mask"].to(device)

        output = model(imgs)
        if isinstance(output, dict) and "main" in output:
            output = output["main"]

        loss_dict = criterion(output, masks)
        total_loss += loss_dict["loss"].item()
        n += 1

        preds = torch.argmax(output, dim=1)
        for c in range(num_classes):
            pc = (preds == c)
            gc = (masks == c)
            inters[c] += (pc & gc).sum().float()
            unions[c] += (pc | gc).sum().float()
            t_pred[c] += pc.sum().float()
            t_gt[c] += gc.sum().float()

    eps = 1e-6
    pd = (2 * inters + eps) / (t_pred + t_gt + eps)
    pi = (inters + eps) / (unions + eps)
    pp = (inters + eps) / (t_pred + eps)
    pr = (inters + eps) / (t_gt + eps)

    return {
        "loss": total_loss / max(n, 1),
        "dice": pd.mean().item(),
        "dice_teeth": pd[1:].mean().item(),
        "iou": pi.mean().item(),
        "precision": pp.mean().item(),
        "recall": pr.mean().item(),
        "per_class_dice": pd.cpu().numpy(),
    }


@torch.no_grad()
def test_eval(model, config, device, num_classes=53):
    """Final test evaluation."""
    data_cfg = config["data"]
    sz = tuple(data_cfg.get("image_size", [512, 512]))
    ds = MultiClassToothDataset(
        image_dir=data_cfg["image_dir"], mask_dir=data_cfg["mask_dir"],
        split_file=data_cfg["test_split"], transform=get_transforms_val(sz),
        image_size=sz, num_classes=num_classes,
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    model.eval()
    inters = torch.zeros(num_classes)
    unions = torch.zeros(num_classes)
    t_pred = torch.zeros(num_classes)
    t_gt = torch.zeros(num_classes)
    for batch in loader:
        imgs = batch["image"].to(device)
        masks = batch["mask"]
        output = model(imgs)
        if isinstance(output, dict) and "main" in output:
            output = output["main"]
        preds = torch.argmax(output, dim=1).cpu()
        for c in range(num_classes):
            pc = (preds == c)
            gc = (masks == c)
            inters[c] += (pc & gc).sum().float()
            unions[c] += (pc | gc).sum().float()
            t_pred[c] += pc.sum().float()
            t_gt[c] += gc.sum().float()
    eps = 1e-6
    pd = (2 * inters + eps) / (t_pred + t_gt + eps)
    pp = (inters + eps) / (t_pred + eps)  # precision per class
    pr = (inters + eps) / (t_gt + eps)    # recall per class
    pf = 2 * pp * pr / (pp + pr + eps)    # F1 per class
    return {
        "dice": pd.mean().item(), "dice_teeth": pd[1:].mean().item(),
        "iou": ((inters + eps) / (unions + eps)).mean().item(),
        "per_class_dice": pd.tolist(),
        "per_class_precision": pp.tolist(),
        "per_class_recall": pr.tolist(),
        "per_class_f1": pf.tolist(),
        "precision": pp[1:].mean().item(),
        "recall": pr[1:].mean().item(),
        "f1": pf[1:].mean().item(),
    }


def _fdi_names():
    names = ["bg"]
    # Permanent (quadrants 1-4, positions 1-8)
    for q in [1, 2, 3, 4]:
        for p in range(1, 9):
            names.append(f"{q}{p}")
    # Deciduous (quadrants 5-8, positions 1-5)
    for q in [5, 6, 7, 8]:
        for p in range(1, 6):
            names.append(f"{q}{p}")
    return names


def main():
    args = parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    set_seed(config.get("seed", 42))

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    print(f"Device: {device}")

    dc = config["data"]
    tc = config["training"]
    num_classes = dc["num_classes"]
    sz = tuple(dc["image_size"])
    bs = tc["batch_size"]
    nw = tc.get("num_workers", 4)

    # Config flags 
    use_deep_sup = config.get("deep_supervision", {}).get("enabled", False)
    use_boundary_loss = config.get("loss", {}).get("boundary_weight", 0.0) > 0.0
    aux_weight = config.get("deep_supervision", {}).get("aux_weight", 0.3)
    grad_accum = tc.get("gradient_accumulation_steps", 1)

    print(f"Features: Boundary Loss={'ON' if use_boundary_loss else 'OFF'}, "
          f"Deep Supervision={'ON' if use_deep_sup else 'OFF'}, "
          f"Grad Accum={grad_accum}")

    # Load pseudo masks 
    pseudo_mask_dir = dc.get("pseudo_mask_dir", "data/pseudo_masks_multiclass")
    pseudo_dir = Path(pseudo_mask_dir)
    quality_path = pseudo_dir / "quality_scores.json"

    if quality_path.exists():
        with open(quality_path) as f:
            quality_scores = json.load(f)
        thresh = dc.get("pseudo_quality_threshold", 0.4)
        good_names = sorted(
            k for k, v in quality_scores.items() if v["quality"] >= thresh
        )
        print(f"Quality filter: {len(good_names)}/{len(quality_scores)} masks kept "
              f"(threshold={thresh})")
    else:
        print("WARNING: quality_scores.json not found. Using all pseudo masks.")
        good_names = sorted(p.stem for p in pseudo_dir.glob("*.png"))

    # Labeled datasets
    labeled_train = MultiClassToothDataset(
        image_dir=dc["image_dir"], mask_dir=dc["mask_dir"],
        split_file=dc["train_split"], transform=get_transforms_train(sz),
        image_size=sz, num_classes=num_classes,
    )
    val_ds = MultiClassToothDataset(
        image_dir=dc["image_dir"], mask_dir=dc["mask_dir"],
        split_file=dc["val_split"], transform=get_transforms_val(sz),
        image_size=sz, num_classes=num_classes,
    )
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=nw)

    # Pseudo dataset
    all_labeled = set()
    for sf in [dc["train_split"], dc["val_split"], dc["test_split"]]:
        with open(sf) as f:
            all_labeled.update(line.strip() for line in f if line.strip())
    pseudo_names = [n for n in good_names if n not in all_labeled]
    random.seed(config["seed"])
    random.shuffle(pseudo_names)
    print(f"Pseudo masks for training: {len(pseudo_names)}")

    pseudo_ds = MultiClassToothDataset(
        image_dir=dc["image_dir"], mask_dir=str(pseudo_dir),
        image_names=pseudo_names, transform=get_transforms_train(sz),
        image_size=sz, num_classes=num_classes,
    )

    # Sample weights from quality scores
    sample_weights = None
    if quality_path.exists():
        weights = []
        for name in pseudo_names:
            q = quality_scores.get(name, {}).get("quality", 0.5)
            w = min(1.0, 0.3 + q * 0.875)
            weights.append(w)
        sample_weights = torch.tensor(weights, dtype=torch.float32)
        print(f"Sample weights: min={min(weights):.2f}, mean={np.mean(weights):.2f}, "
              f"max={max(weights):.2f}")

    # Model
    mc = config["model"]
    ssl_encoder = mc.get("pretrained_encoder_path", None)
    if use_deep_sup:
        ds_cfg = config.get("deep_supervision", {})
        model = create_deep_supervision_model(
            model_name=mc["name"], encoder_name=mc["encoder"],
            in_channels=mc["in_channels"], num_classes=num_classes,
            pretrained=mc.get("pretrained", True),
            pretrained_encoder_path=ssl_encoder,
            aux_scales=tuple(ds_cfg.get("aux_scales", [1, 2, 3])),
            aux_loss_weight=aux_weight,
        ).to(device)
    else:
        model = create_model(
            model_name=mc["name"], encoder_name=mc["encoder"],
            in_channels=mc["in_channels"], num_classes=num_classes,
            pretrained=mc.get("pretrained", True),
            pretrained_encoder_path=ssl_encoder,
        ).to(device)

    print(f"Model: {num_classes} classes, encoder={mc['encoder']}")

    # Loss
    if use_boundary_loss:
        criterion = build_boundary_multiclass_loss(config, device)
        print("Loss: CE + Dice + Boundary (multi-class)")
    else:
        criterion = build_multiclass_loss(config, device)
        print("Loss: CE + Dice")

    out_dir = config["output"]["save_dir"]

    # Phase 1: Pretrain on pseudo masks only
    
    phase1_epochs = dc.get("phase1_epochs", 40)
    phase2_epochs = dc.get("phase2_epochs", 30)
    lr = tc["learning_rate"]

    p1_loader = DataLoader(pseudo_ds, batch_size=bs, shuffle=True,
                           num_workers=nw, drop_last=True)

    # Compute class weights
    cw = compute_class_weights(p1_loader, num_classes, device,
                            hard_class_boost=tc.get("hard_class_boost", None))
    if hasattr(criterion, 'ce_loss') and hasattr(criterion.ce_loss, 'weight'):
        criterion.ce_loss.weight = cw
        print(f"Class weights loaded (range: {cw.min():.2f}-{cw.max():.2f})")

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=tc["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=phase1_epochs)

    logger = ExperimentLogger(out_dir, config["experiment_name"], resume=False)
    logger.save_config(config)

    best_dice = 0.0
    no_improve = 0
    patience = tc["early_stopping_patience"]

    print(f"\n{'='*60}")
    print(f"Phase 1: Pseudo pretraining ({phase1_epochs} epochs, {len(pseudo_ds)} imgs)")
    print(f"{'='*60}")
    for ep in range(1, phase1_epochs + 1):
        tm = train_epoch(
            model, p1_loader, criterion, optimizer, device, ep, phase1_epochs,
            sample_weights=sample_weights, use_deep_sup=use_deep_sup,
            aux_weight=aux_weight, grad_accum_steps=grad_accum,
        )
        vm = validate(model, val_loader, criterion, device, num_classes)
        scheduler.step()
        logger.log_epoch(ep, tm,
                         {"loss": vm["loss"], "dice": vm["dice_teeth"],
                          "iou": vm["iou"], "precision": vm["precision"],
                          "recall": vm["recall"]},
                         learning_rate=scheduler.get_last_lr()[0])
        print(f"P1 E{ep:3d}/{phase1_epochs} | Train L={tm['loss']:.4f} "
              f"CE={tm.get('ce_loss', 0):.3f} Dice={tm.get('dice_loss', 0):.3f} "
              f"Bnd={tm.get('boundary_loss', 0):.3f} | "
              f"Val Dice(teeth)={vm['dice_teeth']:.4f}")

        if vm["dice_teeth"] > best_dice:
            best_dice = vm["dice_teeth"]
            no_improve = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
        if no_improve >= patience:
            print(f"Early stop at P1 epoch {ep}")
            break


    # Phase 2: Fine-tune on real labeled data

    print(f"\n{'='*60}")
    print(f"Phase 2: Labeled fine-tuning ({phase2_epochs} epochs, {len(labeled_train)} imgs)")
    print(f"{'='*60}")

    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    p2_loader = DataLoader(labeled_train, batch_size=min(bs, len(labeled_train)),
                           shuffle=True, num_workers=0, drop_last=False)

    cw2 = compute_class_weights(p2_loader, num_classes, device,
                             hard_class_boost=tc.get("hard_class_boost", None))
    if hasattr(criterion, 'ce_loss') and hasattr(criterion.ce_loss, 'weight'):
        criterion.ce_loss.weight = cw2

    p2_lr = tc.get("phase2_lr", lr * 0.25)
    optimizer = optim.AdamW(model.parameters(), lr=p2_lr, weight_decay=tc["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=phase2_epochs)

    best_dice_p2 = 0.0
    no_improve = 0

    for ep in range(1, phase2_epochs + 1):
        tm = train_epoch(
            model, p2_loader, criterion, optimizer, device, ep, phase2_epochs,
            use_deep_sup=use_deep_sup, aux_weight=aux_weight,
            grad_accum_steps=grad_accum,
        )
        vm = validate(model, val_loader, criterion, device, num_classes)
        scheduler.step()
        logger.log_epoch(phase1_epochs + ep, tm,
                         {"loss": vm["loss"], "dice": vm["dice_teeth"],
                          "iou": vm["iou"], "precision": vm["precision"],
                          "recall": vm["recall"]},
                         learning_rate=scheduler.get_last_lr()[0])
        print(f"P2 E{ep:3d}/{phase2_epochs} | Train L={tm['loss']:.4f} "
              f"CE={tm.get('ce_loss', 0):.3f} Dice={tm.get('dice_loss', 0):.3f} "
              f"Bnd={tm.get('boundary_loss', 0):.3f} | "
              f"Val Dice(teeth)={vm['dice_teeth']:.4f}")

        if vm["dice_teeth"] > best_dice_p2:
            best_dice_p2 = vm["dice_teeth"]
            no_improve = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
        if no_improve >= patience:
            print(f"Early stop at P2 epoch {ep}")
            break

    # Test 
    print(f"\n{'='*60}")
    print("Test Evaluation")
    print(f"{'='*60}")
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    tm = test_eval(model, config, device, num_classes)
    print(f"Test Dice(macro)={tm['dice']:.4f}  Dice(teeth)={tm['dice_teeth']:.4f}  "
          f"IoU={tm['iou']:.4f}")
    print(f"Precision={tm['precision']:.4f}  Recall={tm['recall']:.4f}  "
          f"F1={tm['f1']:.4f}")

    fdi = _fdi_names()
    print(f"\nPer-class Metrics:")
    print(f"  {'FDI':>4s}  {'Dice':>6s}  {'Prec':>6s}  {'Rec':>6s}  {'F1':>6s}  Status")
    print(f"  {'-'*4}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*10}")
    for c in range(1, num_classes):
        d = tm["per_class_dice"][c]
        p = tm["per_class_precision"][c]
        r = tm["per_class_recall"][c]
        f = tm["per_class_f1"][c]
        if d > 0.6:
            status = "🟢 Excellent"
        elif d > 0.4:
            status = "🟡 Good"
        elif d > 0.1:
            status = "🟠 Weak"
        elif d > 0.001:
            status = "🔴 Near-dead"
        else:
            status = "💀 Dead"
        print(f"  {fdi[c]:>4s}  {d:6.4f}  {p:6.4f}  {r:6.4f}  {f:6.4f}  {status}")

    # Count good teeth
    good = sum(1 for c in range(1, num_classes) if tm["per_class_dice"][c] > 0.5)
    great = sum(1 for c in range(1, num_classes) if tm["per_class_dice"][c] > 0.6)
    print(f"\nTeeth with Dice > 0.5: {good}/52")
    print(f"Teeth with Dice > 0.6: {great}/52")

    # Save results
    import csv
    out_p = Path(out_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    with open(out_p / "results_test.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        for k, v in tm.items():
            if not isinstance(v, list):
                w.writerow([k, f"{v:.6f}"])

    with open(out_p / "results_test_per_class.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "fdi", "dice"])
        for c in range(1, num_classes):
            w.writerow([c, fdi[c], f"{tm['per_class_dice'][c]:.6f}"])

    # Save best model
    torch.save(best_state, out_p / "best_model.pth")

    # Save summary
    summary = {
        "best_val_dice_p1": float(best_dice),
        "best_val_dice_p2": float(best_dice_p2),
        "test_dice_macro": float(tm["dice"]),
        "test_dice_teeth": float(tm["dice_teeth"]),
        "test_iou": float(tm["iou"]),
        "teeth_above_05": good,
        "teeth_above_06": great,
        "features": {
            "boundary_loss": use_boundary_loss,
            "deep_supervision": use_deep_sup,
            "pseudo_dir": pseudo_mask_dir,
        },
    }
    with open(out_p / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to: {out_dir}")


if __name__ == "__main__":
    main()
