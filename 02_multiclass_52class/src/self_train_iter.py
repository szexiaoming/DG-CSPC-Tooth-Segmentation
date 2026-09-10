"""
Iterative Self-Training: use trained v3 model to generate better pseudo labels.

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

from src.models import create_model, create_deep_supervision_model
from src.utils import set_seed


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--conf-threshold", type=float, default=0.7,
                   help="Minimum softmax confidence to accept a pixel")
    p.add_argument("--output-name", type=str, default="pseudo_masks_iter2",
                   help="Output directory name under data/")
    p.add_argument("--merge-existing", action="store_true", default=True,
                   help="Merge with existing pseudo labels")
    return p.parse_args()


def predict_multiclass(model, image_path, image_size, device, tta=False):
    """Predict multi-class mask for one image. Returns (H, W) class indices 0-32."""
    img = Image.open(image_path).convert("L")
    img = img.resize((image_size[1], image_size[0]), Image.BILINEAR)
    img_np = np.array(img, dtype=np.float32) / 255.0
    img_tensor = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0).to(device)

    model.eval()

    def _forward(x):
        with torch.no_grad():
            logits = model(x)
            # Handle deep supervision output (tuple)
            if isinstance(logits, tuple):
                logits = logits[0]
            probs = F.softmax(logits, dim=1)
        return probs

    probs = _forward(img_tensor)

    # TTA: horizontal flip
    if tta:
        probs_flip = _forward(torch.flip(img_tensor, dims=[-1]))
        probs_flip = torch.flip(probs_flip, dims=[-1])
        probs = (probs + probs_flip) / 2.0

    probs_np = probs[0].cpu().numpy()  # (num_classes, H, W)
    pred = probs_np.argmax(axis=0)      # (H, W), values 0-32
    conf = probs_np.max(axis=0)          # (H, W), confidence

    return pred.astype(np.uint8), conf.astype(np.float32)


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

    # Load model
    model_cfg = config.get("model", {})
    ds_cfg = config.get("deep_supervision", {})

    if ds_cfg.get("enabled", False):
        model = create_deep_supervision_model(
            model_name=model_cfg.get("name", "unet"),
            encoder_name=model_cfg.get("encoder", "mobilenet_v2"),
            in_channels=model_cfg.get("in_channels", 1),
            num_classes=num_classes,
            pretrained=False,
            aux_scales=ds_cfg.get("aux_scales", [1, 2, 3]),
        )
    else:
        model = create_model(
            model_name=model_cfg.get("name", "unet"),
            encoder_name=model_cfg.get("encoder", "resnet34"),
            in_channels=model_cfg.get("in_channels", 1),
            num_classes=num_classes,
            pretrained=False,
        )

    state_dict = torch.load(args.checkpoint, map_location=device)
    # Handle both raw state_dict and {"model_state_dict": ...} format
    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print(f"Model loaded from: {args.checkpoint}")

    # Input directories
    image_dir = project_root / "data" / "processed" / "images"
    # Use previous pseudo labels for merging (read from config)
    old_pseudo_dir = project_root / data_cfg.get("pseudo_mask_dir", "data/pseudo_masks_yolo_sam")
    output_dir = project_root / "data" / args.output_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get labeled and unlabeled sets
    splits_dir = project_root / "data" / "splits"
    labeled_names = set()
    for sf in ["train.txt", "val.txt", "test.txt"]:
        with open(splits_dir / sf, "r") as f:
            labeled_names.update(line.strip() for line in f if line.strip())

    all_images = sorted(p.stem for p in image_dir.glob("*.png"))
    unlabeled = [n for n in all_images if n not in labeled_names]
    print(f"Processing {len(unlabeled)} unlabeled images (conf > {args.conf_threshold})")

    quality_scores = {}
    stats = {"from_model": 0, "from_old": 0, "merged": 0}

    for name in tqdm(unlabeled, desc="Self-training"):
        img_path = image_dir / f"{name}.png"
        if not img_path.exists():
            img_path = image_dir / f"{name}.jpg"
        if not img_path.exists():
            continue

        # Model prediction
        pred, conf = predict_multiclass(model, img_path, image_size, device, tta=True)

        # Build high-confidence pseudo mask
        model_mask = pred.copy()
        model_mask[conf < args.conf_threshold] = 0  # zero out uncertain pixels

        # Try to merge with old pseudo labels
        old_path = old_pseudo_dir / f"{name}.png"
        if args.merge_existing and old_path.exists():
            old_mask = np.array(Image.open(old_path))

            # Where model is confident, use model; otherwise keep old
            final_mask = old_mask.copy()
            confident = conf >= args.conf_threshold
            final_mask[confident] = model_mask[confident]

            # Count pixels from each source
            n_model = (confident & (model_mask > 0)).sum()
            n_old = ((~confident) & (old_mask > 0)).sum()
            stats["merged"] += 1
        else:
            final_mask = model_mask
            n_model = (model_mask > 0).sum()
            n_old = 0

        n_teeth = len(np.unique(final_mask)) - 1

        # Quality score (up to 52 teeth for mixed dentition)
        if 20 <= n_teeth <= 52:
            q = 0.7 + 0.3 * min(1.0, n_teeth / 48.0)
        elif 14 <= n_teeth < 20:
            q = 0.4 + 0.3 * min(1.0, n_teeth / 48.0)
        else:
            q = max(0.1, n_teeth / 20.0)

        quality_scores[name] = {
            "quality": round(q, 3),
            "n_teeth": n_teeth,
            "method": "iter2_self_train",
        }

        Image.fromarray(final_mask.astype(np.uint8)).save(output_dir / f"{name}.png")

    # Save quality scores
    with open(output_dir / "quality_scores.json", "w") as f:
        json.dump(quality_scores, f, indent=2)

    # Report
    n_teeth_list = [s["n_teeth"] for s in quality_scores.values()]
    good = sum(1 for s in quality_scores.values() if s["quality"] >= 0.6)
    ok = sum(1 for s in quality_scores.values() if 0.4 <= s["quality"] < 0.6)
    bad = sum(1 for s in quality_scores.values() if s["quality"] < 0.4)

    print(f"\n=== Iter 2 Self-Training Results ===")
    print(f"Total: {len(unlabeled)}")
    print(f"Avg teeth/img: {np.mean(n_teeth_list):.1f} (median: {np.median(n_teeth_list):.0f})")
    print(f"Quality: Good={good}, OK={ok}, Bad={bad}")
    print(f"Merged with old: {stats['merged']}")
    print(f"Saved to: {output_dir}")
    print(f"\nNext: update config pseudo_mask_dir to 'data/pseudo_masks_iter2'")


if __name__ == "__main__":
    main()
