"""
Per-sample Dice using EXACT same pipeline as test_eval().

"""

import sys, yaml, torch
from pathlib import Path

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

print(f'{"Sample":<6} {"Baseline":>8} {"DG-CSPC":>8} {"Delta":>8}')
print('-' * 34)

is_b, us_b = 0.0, 0.0
is_s, us_s = 0.0, 0.0
idx = 0

for batch in loader:
    img = batch['image'].cuda()
    mask = batch['mask']
    name = test_names[idx]
    idx += 1

    results = {}
    for tag, model in [('B', m_sup), ('S', m_sel)]:
        with torch.no_grad():
            out = model(img)
            if isinstance(out, dict):
                out = out['main']
        pred = out.argmax(dim=1).cpu()
        i, u = 0.0, 0.0
        for c in range(1, 53):
            pc = pred == c
            gc = mask == c
            i += (pc & gc).sum().float().item()
            u += (pc.sum().float() + gc.sum().float()).item()
        results[tag] = (2.0 * i / (u + 1e-6), i, u)

    db, ib, ub = results['B']
    ds_val, i_sel, u_sel = results['S']
    is_b += ib; us_b += ub
    is_s += i_sel; us_s += u_sel

    print(f'{name:<6} {db:8.4f} {ds_val:8.4f} {ds_val-db:+8.4f}')

agg_b = 2.0 * is_b / (us_b + 1e-6)
agg_s = 2.0 * is_s / (us_s + 1e-6)
print(f'\nAggregate Baseline: {agg_b:.4f}  (expected 0.3084)')
print(f'Aggregate DG-CSPC: {agg_s:.4f}  (expected 0.6096)')
