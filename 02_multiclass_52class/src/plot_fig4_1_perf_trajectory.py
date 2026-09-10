"""
Reproduce thesis Figure 4.1  Dice(teeth) performance trajectory.

"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
import numpy as np

OUT_DIR = Path(__file__).resolve().parent / 'figures'
OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
})

experiments = ['Pseudo-init', 'Old\npipeline', 'YOLO+SAM\n+Hungarian',
               'Self-train\nR1', 'Self-train\nR2', 'DG-CSPC']
dice_teeth = [0.308, 0.248, 0.387, 0.505, 0.415, 0.610]

GREY      = '#8c8c8c'
DARK_GREY = '#595959'
ORANGE    = '#d4a76a'
GREEN     = '#7b9e6d'
RED       = '#c27a7a'
BLUE      = '#5b7db5'
colors = [GREY, DARK_GREY, ORANGE, GREEN, RED, BLUE]

x = np.arange(len(experiments))

fig, ax = plt.subplots(figsize=(7.5, 4.2))
bars = ax.bar(x, dice_teeth, color=colors, edgecolor='white', linewidth=0.6, width=0.58)

for bar, v in zip(bars, dice_teeth):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.010,
            f'{v:.3f}', ha='center', va='bottom', fontsize=9)

ax.axhline(y=0.308, color='#b0b0b0', linestyle='--', linewidth=0.7, alpha=0.7,
           label='Pseudo-init baseline')
ax.set_xticks(x)
ax.set_xticklabels(experiments)
ax.set_ylabel('Dice(teeth)')
ax.set_ylim(0, 0.70)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
ax.yaxis.set_major_locator(mticker.MultipleLocator(0.10))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(fontsize=9, loc='upper right', frameon=False)

plt.tight_layout(pad=0.5)
fig.savefig(OUT_DIR / 'perf_trajectory.png', dpi=300, bbox_inches='tight')
fig.savefig(OUT_DIR / 'perf_trajectory.pdf', dpi=300, bbox_inches='tight')
plt.close()
print(f'Saved: {OUT_DIR / "perf_trajectory.png"} + .pdf')
