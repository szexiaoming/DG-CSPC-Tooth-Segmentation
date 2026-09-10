"""
Improved Self-Training v2 — per-class confidence thresholds + adaptive merging.

"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models import create_deep_supervision_model
from src.utils import set_seed


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--output-name", type=str, default="pseudo_masks_iter4_v2",
                   help="Output directory under data/")
    p.add_argument("--per-class-history", type=str, default=None,
                   help="JSON with per-class Dice from previous rounds, "
                        "e.g. '{\"51\": [0.0, 0.144, 0.376], ...}'")
    return p.parse_args()



def get_per_class_thresholds(per_class_dice_latest, num_classes=53):
    """
    Assign confidence thresholds based on latest per-class Dice.

    """
    thresholds = {}
    for c in range(1, num_classes):
        d = per_class_dice_latest.get(str(c), 0.0)
        if d > 0.6:
            thresholds[c] = 0.85
        elif d > 0.4:
            thresholds[c] = 0.75
        elif d > 0.2:
            thresholds[c] = 0.60
        elif d > 0.05:
            thresholds[c] = 0.45
        else:
            thresholds[c] = 0.30  # nearly dead — keep almost everything
    return thresholds


def get_frozen_classes(per_class_history, num_classes=53):
    """
    Classes that are dead or declining → freeze from best round.

    """
    frozen = set()
    for c in range(1, num_classes):
        key = str(c)
        history = per_class_history.get(key, [])
        if len(history) < 2:
            continue
        latest = history[-1]
        # Dead
        if latest < 0.001:
            frozen.add(c)
        # Declining two rounds in a row
        elif len(history) >= 3:
            if history[-1] < history[-2] - 0.03 and history[-2] < history[-3] - 0.03:
                frozen.add(c)
    return frozen


def predict_multiclass(model, image_path, image_size, device):
    """
    Predict multi-class mask and confidence map.
    
    """
    img = Image.open(image_path).convert("L")
    img = img.resize((image_size[1], image_size[0]), Image.BILINEAR)
    img_np = np.array(img, dtype=np.float32) / 255.0
    img_tensor = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        logits = model(img_tensor)
        if isinstance(logits, tuple):
            logits = logits[0]
        # TTA: horizontal flip
        probs = F.softmax(logits, dim=1)
        probs_flip = F.softmax(model(torch.flip(img_tensor, dims=[-1]))[0]
                               if isinstance(model(torch.flip(img_tensor, dims=[-1])), tuple)
                               else model(torch.flip(img_tensor, dims=[-1])), dim=1)
        probs_flip = torch.flip(probs_flip, dims=[-1])
        probs = (probs + probs_flip) / 2.0

    probs_np = probs[0].cpu().numpy()
    pred = probs_np.argmax(axis=0).astype(np.uint8)
    conf = probs_np.max(axis=0).astype(np.float32)

    # Also keep per-class max probability for per-class thresholding
    return pred, conf, probs_np


def apply_per_class_threshold(pred, conf, probs, thresholds, frozen_classes):
    """
    Apply per-class confidence threshold.
    - Normal classes: keep pixels where confidence > per-class pi
    - Frozen classes: keep nothing (will be filled from previous round)
    """
    mask = pred.copy()
    for c in range(1, probs.shape[0]):
        if c in frozen_classes:
            mask[pred == c] = 0  # Remove frozen class predictions
            continue
        tau = thresholds.get(c, 0.70)
        class_pixels = pred == c
        if class_pixels.sum() == 0:
            continue
        # For pixels predicted as class c, check whether confidence exceeds τ
        low_conf = class_pixels & (conf < tau)
        mask[low_conf] = 0
    return mask


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parent.parent

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    set_seed(config.get("seed", 42))

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    data_cfg = config.get("data", {})
    image_size = tuple(data_cfg.get("image_size", [512, 512]))
    num_classes = data_cfg.get("num_classes", 53)

    # Load per-class history
    per_class_history = {}
    if args.per_class_history:
        with open(args.per_class_history, "r") as f:
            per_class_history = json.load(f)
        print(f"Loaded per-class history for {len(per_class_history)} classes")
    else:
        print("WARNING: No per-class history provided. Using default thresholds (τ=0.70 for all).")

    # Compute per-class latest Dice
    latest_dice = {}
    for c_str, history in per_class_history.items():
        if history:
            latest_dice[c_str] = history[-1]

    thresholds = get_per_class_thresholds(latest_dice, num_classes)
    frozen = get_frozen_classes(per_class_history, num_classes)

    print(f"Frozen classes (dead or declining): {sorted(frozen)}")
    print(f"Per-class τ range: [{min(thresholds.values()):.2f}, {max(thresholds.values()):.2f}]")

    # Load model
    model_cfg = config.get("model", {})
    ds_cfg = config.get("deep_supervision", {})
    model = create_deep_supervision_model(
        model_name=model_cfg.get("name", "unet"),
        encoder_name=model_cfg.get("encoder", "mobilenet_v2"),
        in_channels=model_cfg.get("in_channels", 1),
        num_classes=num_classes,
        pretrained=False,
        aux_scales=ds_cfg.get("aux_scales", [1, 2, 3]),
    )
    state_dict = torch.load(args.checkpoint, map_location=device)
    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print(f"Model loaded from: {args.checkpoint}")

    # Directories
    image_dir = project_root / "data" / "processed" / "images"
    old_pseudo_dir = project_root / data_cfg.get("pseudo_mask_dir", "data/pseudo_masks_iter3")
    frozen_pseudo_dir = project_root / "data" / "pseudo_masks_frozen"
    output_dir = project_root / "data" / args.output_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get unlabeled images
    splits_dir = project_root / "data" / "splits"
    labeled_names = set()
    for sf in ["train.txt", "val.txt", "test.txt"]:
        with open(splits_dir / sf, "r") as f:
            labeled_names.update(line.strip() for line in f if line.strip())

    all_images = sorted(p.stem for p in image_dir.glob("*.png"))
    unlabeled = [n for n in all_images if n not in labeled_names]
    print(f"Processing {len(unlabeled)} unlabeled images")

    quality_scores = {}
    stats = {"total": 0, "frozen_pixels": 0, "model_pixels": 0}

    for name in tqdm(unlabeled, desc="Self-training v2"):
        img_path = image_dir / f"{name}.png"
        if not img_path.exists():
            img_path = image_dir / f"{name}.jpg"
        if not img_path.exists():
            continue

        pred, conf, probs = predict_multiclass(model, img_path, image_size, device)
        model_mask = apply_per_class_threshold(pred, conf, probs, thresholds, frozen)

        # Merge with old pseudo-labels
        old_path = old_pseudo_dir / f"{name}.png"
        if old_path.exists():
            old_mask = np.array(Image.open(old_path))
            final_mask = old_mask.copy()
            # Use model predictions where confident (per-class τ applied)
            final_mask[model_mask > 0] = model_mask[model_mask > 0]
            # For frozen classes: keep old pseudo-labels (best available)
            for fc in frozen:
                if (old_mask == fc).sum() > 0:
                    final_mask[old_mask == fc] = fc
        else:
            final_mask = model_mask

        n_teeth = len(np.unique(final_mask)) - 1
        # Quality score: wider range for mixed dentition
        if 20 <= n_teeth <= 56:
            q = 0.7 + 0.3 * min(1.0, n_teeth / 52.0)
        elif 14 <= n_teeth < 20:
            q = 0.4 + 0.3 * min(1.0, n_teeth / 52.0)
        else:
            q = max(0.1, n_teeth / 20.0)

        quality_scores[name] = {"quality": round(q, 3), "n_teeth": n_teeth}
        Image.fromarray(final_mask.astype(np.uint8)).save(output_dir / f"{name}.png")
        stats["total"] += 1

    # Report
    n_teeth_list = [s["n_teeth"] for s in quality_scores.values()]
    good = sum(1 for s in quality_scores.values() if s["quality"] >= 0.6)
    ok = sum(1 for s in quality_scores.values() if 0.4 <= s["quality"] < 0.6)
    bad = sum(1 for s in quality_scores.values() if s["quality"] < 0.4)

    with open(output_dir / "quality_scores.json", "w") as f:
        json.dump(quality_scores, f, indent=2)

    print(f"\n=== Self-Training v2 Results ===")
    print(f"Total: {stats['total']}")
    print(f"Avg teeth/img: {np.mean(n_teeth_list):.1f} (median: {np.median(n_teeth_list):.0f})")
    print(f"Quality: Good={good}, OK={ok}, Bad={bad}")
    print(f"Frozen classes: {sorted(frozen)}")
    print(f"Saved to: {output_dir}")
    print(f"\nNext: update config pseudo_mask_dir to 'data/{args.output_name}'")


if __name__ == "__main__":
    main()
