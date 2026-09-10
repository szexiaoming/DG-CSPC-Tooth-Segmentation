"""
Per-sample Dice using MACRO-average (per-class mean), matching training test_eval().
This is consistent with the aggregate table in the thesis.

"""

import sys, yaml, torch
from pathlib import Path
import numpy as np

PROJ = Path(__file__).resolve().parent.parent / '02_multiclass_52class'
sys.path.insert(0, str(PROJ))
sys.path.insert(0, str(PROJ / 'src'))

from models import create_deep_supervision_model
from src.datasets import MultiClassToothDataset, get_transforms_val
from torch.utils.data import DataLoader


def load_model(ckpt, cfg_path):
    cfg = yaml.safe_load(open(PROJ / cfg_path))
    mc = cfg['model']
    ds = cfg.get('deep_supervision', {})
    m = create_deep_supervision_model(
        mc['name'], mc['encoder'], mc['in_channels'], 53,
        pretrained=False, aux_scales=tuple(ds.get('aux_scales', [1, 2, 3])))
    m.load_state_dict(torch.load(PROJ / ckpt, map_location='cuda'))
    m.cuda()
    m.eval()
    return m, cfg


m_sup, c_sup = load_model(
    'experiments/ours_multiclass_v3_supervised/best_model.pth',
    'configs/ours_multiclass_v3_supervised_baseline.yaml')
m_sel, c_sel = load_model(
    'experiments/ours_multiclass_v3_selective/best_model.pth',
    'configs/ours_multiclass_v3_selective.yaml')

dc = c_sel['data']
sz = tuple(dc['image_size'])
test_names = [l.strip() for l in open(PROJ / dc['test_split'])]

dset = MultiClassToothDataset(
    image_dir=str(PROJ / dc['image_dir']),
    mask_dir=str(PROJ / dc['mask_dir']),
    split_file=str(PROJ / dc['test_split']),
    transform=get_transforms_val(sz), image_size=sz, num_classes=53)
loader = DataLoader(dset, batch_size=1, shuffle=False)

# Accumulate per-class stats for aggregate macro-average
all_inter_b = np.zeros(53, dtype=np.float64)
all_pred_b = np.zeros(53, dtype=np.float64)
all_gt_b = np.zeros(53, dtype=np.float64)
all_inter_s = np.zeros(53, dtype=np.float64)
all_pred_s = np.zeros(53, dtype=np.float64)
all_gt_s = np.zeros(53, dtype=np.float64)

print(f"{'Sample':<6} {'B_macro':>8} {'S_macro':>8} {'Delta':>8}  "
      f"{'B_micro':>8} {'S_micro':>8}  {'B_teeth':>7} {'S_teeth':>7}")
print('-' * 78)

all_per_sample = []

for idx, batch in enumerate(loader):
    img = batch['image'].cuda()
    mask = batch['mask']
    name = test_names[idx]

    sample_results = {}
    for tag, model in [('B', m_sup), ('S', m_sel)]:
        with torch.no_grad():
            out = model(img)
            if isinstance(out, dict):
                out = out['main']
        pred = out.argmax(dim=1).cpu()

        # Per-class intersection / pred / gt for this sample
        inter = np.zeros(53, dtype=np.float64)
        t_pred = np.zeros(53, dtype=np.float64)
        t_gt = np.zeros(53, dtype=np.float64)
        for c in range(1, 53):
            pc = (pred == c).sum().item()
            gc = (mask == c).sum().item()
            ic = ((pred == c) & (mask == c)).sum().item()
            inter[c] = ic
            t_pred[c] = pc
            t_gt[c] = gc

        # Per-class Dice then average (MACRO)
        eps = 1e-6
        per_class_dice = (2 * inter + eps) / (t_pred + t_gt + eps)
        macro_dice = per_class_dice[1:].mean()  # exclude background

        # Per-sample micro (for comparison)
        total_i = inter[1:].sum()
        total_p = t_pred[1:].sum()
        total_g = t_gt[1:].sum()
        micro_dice = 2 * total_i / (total_p + total_g + eps)

        n_active = int((t_pred[1:] > 0).sum())

        sample_results[tag] = {
            'macro': macro_dice, 'micro': micro_dice,
            'inter': inter, 'pred': t_pred, 'gt': t_gt,
            'n_active': n_active,
        }

    bm = sample_results['B']['macro']
    sm = sample_results['S']['macro']
    bmi = sample_results['B']['micro']
    smi = sample_results['S']['micro']
    bt = sample_results['B']['n_active']
    st = sample_results['S']['n_active']

    all_per_sample.append((name, bm, sm, bmi, smi, bt, st))

    # Accumulate for aggregate
    all_inter_b += sample_results['B']['inter']
    all_pred_b += sample_results['B']['pred']
    all_gt_b += sample_results['B']['gt']
    all_inter_s += sample_results['S']['inter']
    all_pred_s += sample_results['S']['pred']
    all_gt_s += sample_results['S']['gt']

    print(f"{name:<6} {bm:8.4f} {sm:8.4f} {sm-bm:+8.4f}  "
          f"{bmi:8.4f} {smi:8.4f}  {bt:6d} {st:6d}")

# Aggregate macro
eps = 1e-6
per_class_b = (2 * all_inter_b + eps) / (all_pred_b + all_gt_b + eps)
agg_b_macro = per_class_b[1:].mean()
per_class_s = (2 * all_inter_s + eps) / (all_pred_s + all_gt_s + eps)
agg_s_macro = per_class_s[1:].mean()

# Aggregate micro
agg_b_micro = 2 * all_inter_b[1:].sum() / (all_pred_b[1:].sum() + all_gt_b[1:].sum() + eps)
agg_s_micro = 2 * all_inter_s[1:].sum() / (all_pred_s[1:].sum() + all_gt_s[1:].sum() + eps)

print(f"\n{'-'*78}")
print(f"{'AGGREGATE':<6} {agg_b_macro:8.4f} {agg_s_macro:8.4f} {agg_s_macro-agg_b_macro:+8.4f}  "
      f"{agg_b_micro:8.4f} {agg_s_micro:8.4f}")
print(f"\nExpected (training test_eval):  B_macro={c_sup.get('_test_dice', 0.3084):.4f}  "
      f"S_macro={c_sel.get('_test_dice', 0.6096):.4f}")

# Count active/dead classes
b_active = int((per_class_b[1:] > 0.001).sum())
s_active = int((per_class_s[1:] > 0.001).sum())
b_good = int((per_class_b[1:] > 0.5).sum())
s_good = int((per_class_s[1:] > 0.5).sum())
print(f"\nActive classes: B={b_active}/52  S={s_active}/52")
print(f"Classes > 0.5:  B={b_good}/52  S={s_good}/52")
