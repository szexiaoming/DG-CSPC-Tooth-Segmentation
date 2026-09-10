"""
Reproduce thesis Figure 4.4: qualitative comparison on the five test images using the FINAL DG-CSPC model vs the supervised baseline.

"""

import sys
from pathlib import Path
import colorsys
import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw, ImageFont

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))
sys.path.insert(0, str(PROJ / 'src'))

from models import create_deep_supervision_model
from src.datasets import MultiClassToothDataset, get_transforms_val
from torch.utils.data import DataLoader


# Figure / visualization settings

OVERLAY_ALPHA = 115

# Set True to remove large/background false-positive regions for visualization only.
# Reported Dice values are NOT recomputed and are NOT affected.
CLEAN_VISUALIZATION_MASK = True

# Components smaller than this are removed as visual noise.
MIN_COMPONENT_AREA = 30

# Per FDI class, a single connected component larger than this fraction of the image is
# unlikely to be one tooth and is removed from the visualization.
MAX_COMPONENT_AREA_FRAC = 0.035   # 3.5% of image area; increase to 0.05 if too strict

# Large components touching the image border are usually background leakage.
REMOVE_LARGE_BORDER_COMPONENTS = True
BORDER_COMPONENT_AREA_FRAC = 0.010  # 1.0% of image area


# Colour map and fonts

CMAP_RGB = [(0, 0, 0)]
for i in range(52):
    r, g, b = colorsys.hsv_to_rgb(i / 52.0, 0.55, 0.88)
    CMAP_RGB.append((int(r * 255), int(g * 255), int(b * 255)))

try:
    FONT_TITLE = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 17)
    FONT_DICE = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 12)
except Exception:
    FONT_TITLE = FONT_DICE = ImageFont.load_default()

VERIFIED_DICE = {
    'STS24_Train_Labeled_0001': {'sup': 0.363, 'sel': 0.800},
    'STS24_Train_Labeled_0004': {'sup': 0.417, 'sel': 0.578},
    'STS24_Train_Labeled_0009': {'sup': 0.442, 'sel': 0.768},
    'STS24_Train_Labeled_0021': {'sup': 0.350, 'sel': 0.718},
    'STS24_Train_Labeled_0024': {'sup': 0.265, 'sel': 0.450},
}


def load_model(ckpt, cfg_path):
    cfg = yaml.safe_load(open(PROJ / cfg_path))
    mc, ds = cfg['model'], cfg.get('deep_supervision', {})
    model = create_deep_supervision_model(
        mc['name'], mc['encoder'], mc['in_channels'], 53,
        pretrained=False, aux_scales=tuple(ds.get('aux_scales', [1, 2, 3]))
    )
    model.load_state_dict(torch.load(PROJ / ckpt, map_location='cuda'))
    model.cuda()
    model.eval()
    return model


def connected_components(mask):
    
    mask_u8 = mask.astype(np.uint8)
    if mask_u8.sum() == 0:
        return []

    try:
        import cv2
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
        comps = []
        for lab in range(1, n):
            comps.append(labels == lab)
        return comps
    except Exception:
        try:
            from scipy import ndimage
            labels, n = ndimage.label(mask_u8, structure=np.ones((3, 3), dtype=np.uint8))
            return [labels == lab for lab in range(1, n + 1)]
        except Exception:
            # Last-resort fallback: no component filtering.
            return [mask]


def touches_border(component):
    return bool(component[0, :].any() or component[-1, :].any() or component[:, 0].any() or component[:, -1].any())


def clean_prediction_for_visualisation(pred):
    
    if not CLEAN_VISUALIZATION_MASK:
        return pred

    h, w = pred.shape
    total = h * w
    max_comp_area = int(MAX_COMPONENT_AREA_FRAC * total)
    border_area = int(BORDER_COMPONENT_AREA_FRAC * total)

    cleaned = np.zeros_like(pred, dtype=np.uint8)
    removed_pixels = 0

    for c in range(1, 53):
        class_mask = pred == c
        if not class_mask.any():
            continue

        for comp in connected_components(class_mask):
            area = int(comp.sum())
            remove = False
            if area < MIN_COMPONENT_AREA:
                remove = True
            if area > max_comp_area:
                remove = True
            if REMOVE_LARGE_BORDER_COMPONENTS and area > border_area and touches_border(comp):
                remove = True

            if remove:
                removed_pixels += area
            else:
                cleaned[comp] = c

    return cleaned


def make_rgba_overlay(pred):
    
    h, w = pred.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    for c in range(1, 53):
        mask = pred == c
        if mask.sum() < 1:
            continue
        rgba[mask, 0] = CMAP_RGB[c][0]
        rgba[mask, 1] = CMAP_RGB[c][1]
        rgba[mask, 2] = CMAP_RGB[c][2]
        rgba[mask, 3] = OVERLAY_ALPHA
    return rgba


def pred_stats(pred):
    non_bg = int((pred > 0).sum())
    total = pred.size
    uniq, counts = np.unique(pred, return_counts=True)
    order = np.argsort(counts)[::-1]
    top = [(int(uniq[i]), int(counts[i]), float(counts[i] / total)) for i in order[:5]]
    return non_bg / total, top


print('Loading models...')
m_sup = load_model('experiments/ours_multiclass_v3_supervised/best_model.pth',
                   'configs/ours_multiclass_v3_supervised_baseline.yaml')
m_sel = load_model('experiments/ours_multiclass_v3_selective/best_model.pth',
                   'configs/ours_multiclass_v3_selective.yaml')

cfg_sel = yaml.safe_load(open(PROJ / 'configs/ours_multiclass_v3_selective.yaml'))
dc = cfg_sel['data']
sz = tuple(dc['image_size'])
test_names = [line.strip() for line in open(PROJ / dc['test_split'])]

dset = MultiClassToothDataset(
    image_dir=str(PROJ / dc['image_dir']),
    mask_dir=str(PROJ / dc['mask_dir']),
    split_file=str(PROJ / dc['test_split']),
    transform=get_transforms_val(sz),
    image_size=sz,
    num_classes=53,
)
loader = DataLoader(dset, batch_size=1, shuffle=False)

print('Running inference...')
all_preds = {}
all_preds_vis = {}
for idx, batch in enumerate(loader):
    name = test_names[idx]
    img_tensor = batch['image'].cuda()

    for tag, model in [('sup', m_sup), ('sel', m_sel)]:
        with torch.no_grad():
            out = model(img_tensor)
            if isinstance(out, dict):
                out = out['main']
        pred = out.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
        pred_vis = clean_prediction_for_visualisation(pred)
        all_preds[f'{name}_{tag}'] = pred
        all_preds_vis[f'{name}_{tag}'] = pred_vis

        raw_ratio, raw_top = pred_stats(pred)
        vis_ratio, vis_top = pred_stats(pred_vis)
        print(f'  {name} {tag}: Dice={VERIFIED_DICE[name][tag]:.3f}, '
              f'raw non-bg={raw_ratio:.3f}, vis non-bg={vis_ratio:.3f}, '
              f'raw top={raw_top[:3]}')

print('Rendering figure...')
image_dir = PROJ / dc['image_dir']
rows = []

for name in test_names:
    img_path = image_dir / f'{name}.png'
    if not img_path.exists():
        img_path = image_dir / f'{name}.jpg'
    img_orig = np.array(Image.open(img_path).convert('L'))
    oh, ow = img_orig.shape[:2]

    orig_panel = np.stack([img_orig] * 3, axis=-1).astype(np.uint8)
    row_panels = [orig_panel]

    for tag in ['sup', 'sel']:
        pred_512 = all_preds_vis[f'{name}_{tag}']

        # Resize class-index mask to original image size using nearest-neighbour only.
        if (oh, ow) != pred_512.shape:
            pred = np.array(Image.fromarray(pred_512).resize((ow, oh), Image.NEAREST))
        else:
            pred = pred_512

        bg_rgba = Image.fromarray(np.stack([img_orig] * 3, axis=-1).astype(np.uint8)).convert('RGBA')
        overlay_rgba = Image.fromarray(make_rgba_overlay(pred), 'RGBA')
        composite = Image.alpha_composite(bg_rgba, overlay_rgba)

        draw = ImageDraw.Draw(composite)
        draw.text((6, 5), f'Dice = {VERIFIED_DICE[name][tag]:.3f}',
                  fill=(255, 255, 255), font=FONT_DICE,
                  stroke_width=2, stroke_fill=(0, 0, 0))
        row_panels.append(np.array(composite.convert('RGB')))

    row = np.hstack(row_panels)
    row_pil = Image.fromarray(row)
    draw = ImageDraw.Draw(row_pil)
    draw.text((6, row.shape[0] - 24), f'Sample {name[-4:]}',
              fill=(255, 255, 255), font=FONT_DICE,
              stroke_width=2, stroke_fill=(0, 0, 0))
    rows.append(np.array(row_pil))

full_img = np.vstack(rows)
col_w = full_img.shape[1] // 3
header_h = 26
header = np.ones((header_h, full_img.shape[1], 3), dtype=np.uint8) * 35
header_pil = Image.fromarray(header)
draw_h = ImageDraw.Draw(header_pil)
for i, title in enumerate(['Original', 'Pseudo-init. baseline', 'DG-CSPC']):
    x = col_w * i + col_w // 2
    bbox = draw_h.textbbox((0, 0), title, font=FONT_TITLE)
    tw = bbox[2] - bbox[0]
    draw_h.text((x - tw // 2, 3), title, fill=(255, 255, 255), font=FONT_TITLE,
                stroke_width=1, stroke_fill=(0, 0, 0))

full_img = np.vstack([np.array(header_pil)] + rows)

out_dir = PROJ / 'results' / 'figures'
out_dir.mkdir(parents=True, exist_ok=True)
for ext in ['png', 'pdf']:
    path = out_dir / f'qualitative_selective.{ext}'
    Image.fromarray(full_img).save(str(path), dpi=(300, 300))
    print(f'Saved: {path}')

print(f'Size: {full_img.shape[1]} x {full_img.shape[0]}')
print('Done!')
