"""
Generate prediction masks and visual overlays using a trained model.

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
from src.utils import set_seed, save_prediction_overlay


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate predictions and visual overlays."
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
        default=None,
        choices=["train", "val", "test"],
        help="Data split to predict on (uses all images in split).",
    )
    parser.add_argument(
        "--image_path",
        type=str,
        default=None,
        help="Path to a single image to predict on (alternative to --split).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for inference.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Binarization threshold for predictions.",
    )
    parser.add_argument(
        "--save_overlays",
        action="store_true",
        help="Save visualization overlays (image + prediction).",
    )
    return parser.parse_args()


def predict_single(
    model: torch.nn.Module,
    image_path: str,
    device: str,
    image_size: tuple,
    threshold: float,
) -> np.ndarray:
    """
    Predict on a single image.
    
    """
    # Load and preprocess
    img = Image.open(image_path)
    img = np.array(img)

    if img.ndim == 3:
        img = img[:, :, 0]

    # Normalize
    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0

    # Resize
    from PIL import Image as PILImage
    img_pil = PILImage.fromarray((img * 255).astype(np.uint8))
    img_pil = img_pil.resize((image_size[1], image_size[0]), PILImage.BILINEAR)
    img = np.array(img_pil).astype(np.float32) / 255.0

    # Add batch and channel dims
    img_tensor = torch.from_numpy(img).float()
    img_tensor = img_tensor.unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, H, W)

    # Predict
    model.eval()
    with torch.no_grad():
        logits = model(img_tensor)
        prob = torch.sigmoid(logits)
        pred = (prob > threshold).float()

    return pred[0, 0].cpu().numpy()


def main() -> None:
    args = parse_args()

    if args.split is None and args.image_path is None:
        print("Error: specify either --split or --image_path.")
        return

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
        encoder_name=model_cfg.get("encoder", "resnet34"),
        in_channels=model_cfg.get("in_channels", 1),
        num_classes=model_cfg.get("num_classes", 1),
        pretrained=False,
    )
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    print(f"Model loaded from: {args.checkpoint}")
    print(f"Device: {device}")

    # Output directory
    output_cfg = config.get("output", {})
    save_dir = Path(output_cfg.get("save_dir", "experiments/default"))
    pred_dir = save_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    # Single image prediction
    if args.image_path:
        print(f"Predicting on: {args.image_path}")
        pred = predict_single(model, args.image_path, device, image_size, args.threshold)
        pred_img = Image.fromarray((pred * 255).astype(np.uint8))
        out_path = pred_dir / f"{Path(args.image_path).stem}_pred.png"
        pred_img.save(out_path)
        print(f"Prediction saved to: {out_path}")
        return

    # Batch prediction on split
    split_file = data_cfg.get(f"{args.split}_split", f"data/splits/{args.split}.txt")
    transform = get_transforms_val(image_size)

    dataset = ToothSegmentationDataset(
        image_dir=data_cfg.get("image_dir", "data/processed/images"),
        mask_dir=data_cfg.get("mask_dir", "data/processed/masks"),
        split_file=split_file,
        transform=transform,
        image_size=image_size,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    print(f"Predicting on {args.split} split: {len(dataset)} samples")

    with torch.no_grad():
        for batch in tqdm(loader, desc="Predicting"):
            images = batch["image"].to(device)
            name = batch["name"][0]

            logits = model(images)
            probs = torch.sigmoid(logits)
            pred_binary = (probs > args.threshold).float()

            pred_np = pred_binary[0, 0].cpu().numpy()

            # Save prediction mask
            pred_img = Image.fromarray((pred_np * 255).astype(np.uint8))
            pred_img.save(pred_dir / f"{name}_pred.png")

            # Save overlay if requested
            if args.save_overlays:
                mask = batch["mask"]
                mask_np = mask[0, 0].cpu().numpy()
                img_np = images[0, 0].cpu().numpy()
                prob_np = probs[0, 0].cpu().numpy()

                overlay_path = pred_dir / f"{name}_overlay.png"
                save_prediction_overlay(
                    image=img_np,
                    ground_truth=mask_np,
                    prediction=pred_np,
                    save_path=str(overlay_path),
                    title=name,
                    prob_map=prob_np,
                )

    print(f"\nPredictions saved to: {pred_dir}")
    if args.save_overlays:
        print(f"Overlays saved to: {pred_dir}")


if __name__ == "__main__":
    main()
