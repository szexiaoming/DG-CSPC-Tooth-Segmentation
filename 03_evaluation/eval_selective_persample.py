"""
Per-sample evaluation using same DatLoader pipeline as test_eval().
"""

import sys, yaml, torch
from pathlib import Path
base = str(Path(__file__).resolve().parent.parent / '02_multiclass_52class')
sys.path.insert(0, base)
sys.path.insert(0, base + '/src')
from models import create_deep_supervision_model
from src.datasets import MultiClassToothDataset, get_transforms_val
from torch.utils.data import DataLoader



def load_model(ckpt, cfg_path):
    cfg = yaml.safe_load(open(cfg_path))
    mc = cfg['model']; ds = cfg.get('deep_supervision', {})
    m = create_deep_supervision_model(mc['name'], mc['encoder'], mc['in_channels'], 53,
        pretrained=False, aux_scales=tuple(ds.get('aux_scales', [1,2,3])))
    m.load_state_dict(torch.load(ckpt, map_location='cuda'))
    m.cuda(); m.eval()
    return m, cfg

m_sel, c_sel = load_model(base+'/experiments/ours_multiclass_v3_selective/best_model.pth',
                           base+'/configs/ours_multiclass_v3_selective.yaml')
m_sup, c_sup = load_model(base+'/experiments/ours_multiclass_v3_supervised/best_model.pth',
                           base+'/configs/ours_multiclass_v3_supervised_baseline.yaml')

dc = c_sel['data']; sz = tuple(dc['image_size'])

# Single DataLoader with all 5 test samples
ds = MultiClassToothDataset(dc['image_dir'], dc['mask_dir'], dc['test_split'],
                             transform=get_transforms_val(sz), image_size=sz, num_classes=53)
loader = DataLoader(ds, batch_size=1, shuffle=False)

print(f'{"Sample":<6} {"Baseline":>8} {"Selective":>8} {"Delta":>8} {"Teeth":>6}')
print('-' * 42)

test_names = [l.strip() for l in open(base+'/'+dc['test_split'])]
sample_idx = 0

for batch in loader:
    img, mask = batch['image'].cuda(), batch['mask'].cuda()
    name = test_names[sample_idx]
    sample_idx += 1

    def dice_scores(model):
        with torch.no_grad():
            out = model(img)
            if isinstance(out, dict): out = out['main']
        pred = out.argmax(dim=1)
        inter, union = 0.0, 0.0
        for c in range(1, 53):
            p = (pred == c); g = (mask == c)
            inter += (p & g).sum().float().item()
            union += (p.sum().float() + g.sum().float()).item()
        n_t = len(torch.unique(pred)) - 1
        return 2 * inter / (union + 1e-6), n_t

    db, _ = dice_scores(m_sup)
    ds_val, nt = dice_scores(m_sel)
    print(f'{name:<6} {db:8.4f} {ds_val:8.4f} {ds_val-db:+8.4f} {nt:6d}')
