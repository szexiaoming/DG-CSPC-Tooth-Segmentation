"""
Selective Self-Training: per-class pseudo-label source selection.

"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch, torch.nn.functional as F, yaml
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
    p.add_argument("--output-name", type=str, default="pseudo_masks_selective")
    p.add_argument("--conf-threshold", type=float, default=0.70,
                   help="Global confidence threshold (same as R1, NOT escalated)")
    p.add_argument("--per-class-dice-r1", type=str, required=True,
                   help="JSON: per-class Dice of R1 model")
    p.add_argument("--per-class-dice-yolo-sam", type=str, required=True,
                   help="JSON: per-class Dice of YOLO+SAM model")
    return p.parse_args()


def predict_multiclass(model, image_path, image_size, device):
    img = Image.open(image_path).convert("L")
    img = img.resize((image_size[1], image_size[0]), Image.BILINEAR)
    img_np = np.array(img, dtype=np.float32) / 255.0
    img_tensor = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        logits = model(img_tensor)
        if isinstance(logits, tuple): logits = logits[0]
        probs = F.softmax(logits, dim=1)
        probs_flip = F.softmax(
            model(torch.flip(img_tensor, dims=[-1]))
            if not isinstance(model(torch.flip(img_tensor, dims=[-1])), tuple)
            else model(torch.flip(img_tensor, dims=[-1]))[0], dim=1)
        probs_flip = torch.flip(probs_flip, dims=[-1])
        probs = (probs + probs_flip) / 2.0

    probs_np = probs[0].cpu().numpy()
    pred = probs_np.argmax(axis=0).astype(np.uint8)
    conf = probs_np.max(axis=0).astype(np.float32)
    return pred, conf, probs_np


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parent.parent

    with open(args.config) as f: config = yaml.safe_load(f)
    set_seed(config.get("seed", 42))
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    data_cfg = config.get("data", {})
    image_size = tuple(data_cfg.get("image_size", [512, 512]))
    num_classes = data_cfg.get("num_classes", 53)
    tau = args.conf_threshold

    # Load per-class Dice to decide which source is better
    with open(args.per_class_dice_r1) as f: dice_r1 = json.load(f)
    with open(args.per_class_dice_yolo_sam) as f: dice_ys = json.load(f)

    # Decide: for each class, use model OR YOLO+SAM
    use_model = {}  # True = use R1 model prediction, False = keep YOLO+SAM
    for c in range(1, num_classes):
        key = str(c)
        r1 = dice_r1.get(key, 0.0)
        ys = dice_ys.get(key, 0.0)
        # Use model if R1 Dice >= YOLO+SAM Dice (model improved or matched)
        # Special case: if BOTH are 0, use YOLO+SAM (older labels may have ANY signal)
        if r1 >= ys and r1 > 0.001:
            use_model[c] = True
        elif ys > 0.001 and r1 < 0.001:
            use_model[c] = False  # YOLO+SAM had signal, model lost it
        elif r1 < 0.001 and ys < 0.001:
            use_model[c] = False  # Both dead, keep YOLO+SAM
        else:
            use_model[c] = (r1 >= ys)

    n_use_model = sum(1 for v in use_model.values() if v)
    n_keep_ys = sum(1 for v in use_model.values() if not v)
    print(f"Source selection: {n_use_model} classes use R1 model, "
          f"{n_keep_ys} classes keep YOLO+SAM")

    # Show which classes keep YOLO+SAM
    keep_ys_classes = [c for c in range(1, num_classes) if not use_model.get(c, False)]
    print(f"Classes keeping YOLO+SAM: {keep_ys_classes}")

    # Load model
    mc, ds = config['model'], config.get('deep_supervision', {})
    model = create_deep_supervision_model(
        mc['name'], mc['encoder'], mc['in_channels'], num_classes,
        pretrained=False, aux_scales=tuple(ds.get('aux_scales', [1,2,3])))
    state_dict = torch.load(args.checkpoint, map_location=device)
    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Directories
    image_dir = project_root / "data" / "processed" / "images"
    yolo_sam_dir = project_root / "data" / "pseudo_masks_yolo_sam"
    output_dir = project_root / "data" / args.output_name
    output_dir.mkdir(parents=True, exist_ok=True)

    splits_dir = project_root / "data" / "splits"
    labeled_names = set()
    for sf in ["train.txt", "val.txt", "test.txt"]:
        with open(splits_dir / sf) as f:
            labeled_names.update(line.strip() for line in f if line.strip())

    all_images = sorted(p.stem for p in image_dir.glob("*.png"))
    unlabeled = [n for n in all_images if n not in labeled_names]
    print(f"Processing {len(unlabeled)} images (τ={tau})")

    quality_scores = {}
    source_stats = {"from_model": 0, "from_yolo_sam": 0}

    for name in tqdm(unlabeled, desc="Selective self-train"):
        img_path = image_dir / f"{name}.png"
        if not img_path.exists():
            img_path = image_dir / f"{name}.jpg"
        if not img_path.exists():
            continue

        pred, conf, probs = predict_multiclass(model, img_path, image_size, device)

        # Apply confidence threshold
        model_mask = pred.copy()
        model_mask[conf < tau] = 0

        # Load YOLO+SAM mask
        ys_path = yolo_sam_dir / f"{name}.png"
        ys_mask = np.array(Image.open(ys_path)) if ys_path.exists() else np.zeros_like(pred)

        # Build final mask: per-class source selection
        final_mask = np.zeros_like(pred)
        for c in range(1, num_classes):
            if use_model.get(c, False):
                final_mask[model_mask == c] = c
            else:
                final_mask[ys_mask == c] = c

        # Count pixels from each source
        n_model = sum(1 for c in range(1, num_classes) if use_model.get(c, False) and (model_mask == c).sum() > 0)
        n_ys = sum(1 for c in range(1, num_classes) if not use_model.get(c, False) and (ys_mask == c).sum() > 0)

        n_teeth = len(np.unique(final_mask)) - 1
        q = 0.7 + 0.3 * min(1.0, n_teeth / 52.0) if 20 <= n_teeth <= 56 else max(0.1, n_teeth / 20.0)
        quality_scores[name] = {"quality": round(q, 3), "n_teeth": n_teeth}
        source_stats["from_model"] += n_model
        source_stats["from_yolo_sam"] += n_ys

        Image.fromarray(final_mask.astype(np.uint8)).save(output_dir / f"{name}.png")

    n_teeth_list = [s["n_teeth"] for s in quality_scores.values()]
    good = sum(1 for s in quality_scores.values() if s["quality"] >= 0.6)
    ok = sum(1 for s in quality_scores.values() if 0.4 <= s["quality"] < 0.6)
    bad = sum(1 for s in quality_scores.values() if s["quality"] < 0.4)

    with open(output_dir / "quality_scores.json", "w") as f:
        json.dump(quality_scores, f, indent=2)

    print(f"\n=== Selective Self-Training Results ===")
    print(f"Total: {len(unlabeled)}")
    print(f"Avg teeth/img: {np.mean(n_teeth_list):.1f} (median: {np.median(n_teeth_list):.0f})")
    print(f"Quality: Good={good}, OK={ok}, Bad={bad}")
    print(f"Source - Model: {source_stats['from_model']}, YOLO+SAM: {source_stats['from_yolo_sam']}")
    print(f"Classes using R1 model: {n_use_model}/52")
    print(f"Saved to: {output_dir}")


if __name__ == "__main__":
    main()
