"""
Generate pseudo-labels for unlabeled images using a trained model.

Filters predictions by confidence threshold and saves high-quality
pseudo-labels for semi-supervised retraining.

"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.datasets import ToothSegmentationDataset, get_transforms_val
from src.models import create_model
from src.utils import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate pseudo-labels.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint.")
    parser.add_argument("--threshold", type=float, default=0.8,
                        help="Confidence threshold for pseudo-label acceptance (hard mode only).")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output_mode", type=str, default="hard", choices=["hard", "soft"],
                        help="hard: binary PNG masks (legacy). soft: probability .npy maps + hard masks.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    set_seed(config.get("seed", 42))

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    data_cfg = config.get("data", {})
    image_size = tuple(data_cfg.get("image_size", [512, 512]))

    # Load model
    model_cfg = config.get("model", {})
    model = create_model(
        model_name=model_cfg.get("name", "unet"),
        encoder_name=model_cfg.get("encoder", "mobilenet_v2"),
        in_channels=model_cfg.get("in_channels", 1),
        num_classes=model_cfg.get("num_classes", 1),
        pretrained=False,
    )
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    print(f"Model loaded. Val Dice: {checkpoint.get('val_dice', 'N/A')}")

    # List all unlabeled images (all processed images without masks)
    image_dir = Path(data_cfg.get("image_dir", "data/processed/images"))
    mask_dir = Path(data_cfg.get("mask_dir", "data/processed/masks"))

    # Get labeled image names (have masks)
    labeled_names = set(f.stem for f in mask_dir.glob("*.png"))

    # Unlabeled = all images - labeled
    all_images = set(f.stem for f in image_dir.glob("*.png"))
    unlabeled_names = sorted(all_images - labeled_names)
    print(f"Unlabeled images: {len(unlabeled_names)}")

    if len(unlabeled_names) == 0:
        print("No unlabeled images found!")
        return

    # Create output directories
    pseudo_mask_dir = Path("data/pseudo_masks")
    pseudo_mask_dir.mkdir(parents=True, exist_ok=True)

    soft_output_dir = None
    if args.output_mode == "soft":
        soft_output_dir = Path("data/pseudo_probs")
        soft_output_dir.mkdir(parents=True, exist_ok=True)

    # Generate pseudo-labels
    accepted = 0
    rejected = 0
    confidences = []

    transform = get_transforms_val(image_size)

    for name in tqdm(unlabeled_names, desc="Generating pseudo-labels"):
        # Load and preprocess image
        img_path = image_dir / f"{name}.png"
        img = np.array(Image.open(img_path))

        if img.ndim == 3:
            img = img[:, :, 0]
        if img.dtype == np.uint8:
            img = img.astype(np.float32) / 255.0

        # Apply transform
        transformed = transform(image=img, mask=np.zeros_like(img))
        img_tensor = torch.from_numpy(transformed["image"]).float()
        if img_tensor.ndim == 2:
            img_tensor = img_tensor.unsqueeze(0)
        img_tensor = img_tensor.unsqueeze(0).to(device)  # (1, 1, H, W)

        # Predict
        with torch.no_grad():
            logits = model(img_tensor)
            prob = torch.sigmoid(logits)
            prob_np = prob[0, 0].cpu().numpy()

        # Mean confidence
        mean_conf = float(prob_np.mean())
        confidences.append(mean_conf)

        # Thresholding (always generate hard mask)
        pseudo_mask = (prob_np > args.threshold).astype(np.uint8) * 255

        # Only accept if at least 0.5% of pixels are foreground
        fg_ratio = pseudo_mask.sum() / pseudo_mask.size / 255.0
        if fg_ratio > 0.005:
            accepted += 1
            # Always save hard mask
            Image.fromarray(pseudo_mask).save(pseudo_mask_dir / f"{name}.png")
            # Save soft probability map if requested
            if soft_output_dir is not None:
                np.save(soft_output_dir / f"{name}.npy", prob_np.astype(np.float16))
        else:
            rejected += 1

    print(f"\n{'='*50}")
    print(f"Pseudo-label generation complete!")
    print(f"  Accepted: {accepted} ({100*accepted/(accepted+rejected):.1f}%)")
    print(f"  Rejected: {rejected} ({100*rejected/(accepted+rejected):.1f}%)")
    print(f"  Mean confidence: {np.mean(confidences):.4f} ± {np.std(confidences):.4f}")
    print(f"  Hard masks saved to: {pseudo_mask_dir}")
    if soft_output_dir is not None:
        print(f"  Soft probs saved to: {soft_output_dir}")
        # Save confidence stats
        import json
        stats = {
            "teacher_checkpoint": args.checkpoint,
            "threshold": args.threshold,
            "num_pseudo_labels": accepted,
            "mean_confidence": float(np.mean(confidences)),
            "std_confidence": float(np.std(confidences)),
            "confidence_percentiles": {
                "p10": float(np.percentile(confidences, 10)),
                "p25": float(np.percentile(confidences, 25)),
                "p50": float(np.percentile(confidences, 50)),
                "p75": float(np.percentile(confidences, 75)),
                "p90": float(np.percentile(confidences, 90)),
            },
        }
        with open(soft_output_dir / "confidence_stats.json", "w") as f:
            json.dump(stats, f, indent=2)
        print(f"  Confidence stats saved to: {soft_output_dir / 'confidence_stats.json'}")


if __name__ == "__main__":
    main()
