"""
Evaluate a trained model on the test or validation split.

Saves per-sample metrics to CSV and prints aggregate statistics.

"""

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.datasets import ToothSegmentationDataset, get_transforms_val
from src.models import create_model
from src.utils import compute_all_metrics, set_seed
from src.utils.model_summary import get_model_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained segmentation model."
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to YAML config."
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint."
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["val", "test"],
        help="Which data split to evaluate on.",
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="Device for inference."
    )
    parser.add_argument(
        "--save_predictions",
        action="store_true",
        help="Save prediction masks as PNG files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Load config
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    set_seed(config.get("seed", 42))

    # Device
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    # Data config
    data_cfg = config.get("data", {})
    image_size = tuple(data_cfg.get("image_size", [512, 512]))

    # Determine split file
    if args.split == "test":
        split_file = data_cfg.get("test_split", "data/splits/test.txt")
    else:
        split_file = data_cfg.get("val_split", "data/splits/val.txt")

    # Create dataset (no augmentation)
    transform = get_transforms_val(image_size)
    dataset = ToothSegmentationDataset(
        image_dir=data_cfg.get("image_dir", "data/processed/images"),
        mask_dir=data_cfg.get("mask_dir", "data/processed/masks"),
        split_file=split_file,
        transform=transform,
        image_size=image_size,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    print(f"Evaluating on {args.split} split: {len(dataset)} samples")

    # Load model
    model_cfg = config.get("model", {})
    model = create_model(
        model_name=model_cfg.get("name", "unet"),
        encoder_name=model_cfg.get("encoder", "resnet34"),
        in_channels=model_cfg.get("in_channels", 1),
        num_classes=model_cfg.get("num_classes", 1),
        pretrained=False,  # We're loading weights
    )
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}, "
          f"val Dice: {checkpoint.get('val_dice', 'N/A')}")

    # Evaluation loop
    all_metrics = []
    pred_dir = Path(config.get("output", {}).get("save_dir", "experiments/default")) / "predictions"
    if args.save_predictions:
        pred_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            images = batch["image"].to(device)
            masks = batch["mask"]
            name = batch["name"][0]

            logits = model(images)
            probs = torch.sigmoid(logits)

            pred_np = probs[0, 0].cpu().numpy()
            mask_np = masks[0, 0].cpu().numpy()

            m = compute_all_metrics(pred_np, mask_np)
            m["name"] = name
            all_metrics.append(m)

            # Save predictions
            if args.save_predictions:
                from PIL import Image
                pred_binary = (pred_np > 0.5).astype(np.uint8) * 255
                Image.fromarray(pred_binary).save(pred_dir / f"{name}_pred.png")

    # Aggregate
    avg_metrics = {}
    for key in ["dice", "iou", "precision", "recall", "boundary_f1"]:
        values = [m[key] for m in all_metrics]
        avg_metrics[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }

    # Print results
    print(f"\n{'='*60}")
    print(f"Evaluation Results ({args.split} split, {len(all_metrics)} samples)")
    print(f"{'='*60}")
    for metric, stats in avg_metrics.items():
        print(
            f"  {metric:14s}: {stats['mean']:.4f} ± {stats['std']:.4f} "
            f"[{stats['min']:.4f}, {stats['max']:.4f}]"
        )

    # Save results CSV
    output_cfg = config.get("output", {})
    save_dir = Path(output_cfg.get("save_dir", "experiments/default"))
    results_csv = save_dir / f"results_{args.split}.csv"
    results_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["name", "dice", "iou", "precision", "recall", "boundary_f1"]
    with open(results_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for m in all_metrics:
            writer.writerow({k: m[k] for k in fieldnames})

    print(f"\nPer-sample results saved to: {results_csv}")

    # Also save aggregate summary
    summary_csv = save_dir / f"summary_{args.split}.csv"
    with open(summary_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "mean", "std", "min", "max"])
        for metric, stats in avg_metrics.items():
            writer.writerow(
                [metric, stats["mean"], stats["std"], stats["min"], stats["max"]]
            )
    print(f"Aggregate summary saved to: {summary_csv}")


if __name__ == "__main__":
    main()
