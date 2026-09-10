"""
Train YOLOv8 detector on 30 labeled tooth images.

Trains a lightweight YOLOv8-nano for tooth detection.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--model", type=str, default="yolov8n.pt",
                   help="YOLOv8 model: yolov8n.pt, yolov8s.pt, etc.")
    return p.parse_args()


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parent.parent

    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: ultralytics not installed!")
        print("Run: pip install ultralytics")
        return

    data_yaml = project_root / "data" / "yolo_dataset" / "dataset.yaml"
    if not data_yaml.exists():
        print("ERROR: YOLO dataset not found!")
        print("Run first: python src/convert_to_yolo.py")
        return

    output_dir = project_root / "data" / "yolo_model"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Training YOLOv8 on {args.device}...")
    print(f"Data: {data_yaml}")

    # Load pretrained YOLOv8
    model = YOLO(args.model)

    # Train
    results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=512,
        batch=8,
        device=args.device,
        workers=4,
        project=str(output_dir),
        name="tooth_detector",
        exist_ok=True,
        # Light augmentations (small dataset)
        hsv_h=0.01, hsv_s=0.1, hsv_v=0.1,
        degrees=5, translate=0.05, scale=0.1,
        fliplr=0.5,
        # Validation
        val=True,
    )

    # Copy best model
    best_src = output_dir / "tooth_detector" / "weights" / "best.pt"
    best_dst = output_dir / "best.pt"
    if best_src.exists():
        import shutil
        shutil.copy(best_src, best_dst)
        print(f"\nModel saved: {best_dst}")
    else:
        print(f"\nWARNING: best.pt not found at {best_src}")
        print("Check training output for errors")

    print("Done!")


if __name__ == "__main__":
    main()
