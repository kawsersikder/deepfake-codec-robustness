"""Summary figures for the README (assets/*.png), drawn from the result files in results/.

Run from the repository root: python analysis/make_figures.py
"""
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.use('Agg')
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'assets'

SURFACE, INK, INK2, MUTED, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#898781', '#e1e0d9'
ACCENT = '#2a78d6'
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK2,
    'xtick.color': INK2, 'ytick.color': INK, 'axes.titlecolor': INK, 'axes.titlesize': 11,
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
})


def style(ax):
    for side in ('top', 'right', 'left'):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis='y', length=0)
    ax.grid(axis='x', color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def foundation_models():
    names = {'clip_l14': 'CLIP ViT-L/14', 'dinov2_g14': 'DINOv2 ViT-g/14', 'eva02_l14': 'EVA-02-L/14'}
    folds = [('celebdf', 'Celeb-DF'), ('ffpp', 'FaceForensics++'), ('dfdc', 'DFDC')]
    t = pd.concat(pd.read_csv(ROOT / f'results/phase13_foundation_lora/{b}/faculty_table_{b}.csv') for b in names)
    order = ['dinov2_g14', 'clip_l14', 'eva02_l14']   # bottom to top: EVA-02 (best) on top
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 2.9), sharey=True)
    for ax, (fold, title) in zip(axes, folds):
        style(ax)
        for y, b in enumerate(order):
            r = t[(t.backbone == b) & (t.held_out == fold)].iloc[0]
            ax.plot([r.probe_auc, r.lora_auc], [y, y], color=GRID, linewidth=2, zorder=1, solid_capstyle='round')
            ax.scatter(r.probe_auc, y, s=60, facecolor=SURFACE, edgecolor=MUTED, linewidth=2, zorder=2)
            ax.scatter(r.lora_auc, y, s=70, color=ACCENT, edgecolor=SURFACE, linewidth=2, zorder=3)
            ax.annotate(f'{r.lora_auc:.3f}', (r.lora_auc, y), xytext=(8, 0), textcoords='offset points',
                        va='center', fontsize=9, color=INK)
        ax.set_title(f'held out: {title}', loc='left', pad=8)
        ax.set_xlim(0.5, 1.03)
        ax.set_yticks(range(len(order)), [names[b] for b in order])
        ax.set_xlabel('video-level AUC on the unseen dataset')
    probe = plt.Line2D([], [], marker='o', linestyle='', markerfacecolor=SURFACE, markeredgecolor=MUTED,
                       markeredgewidth=2, markersize=8, label='frozen linear probe')
    lora = plt.Line2D([], [], marker='o', linestyle='', color=ACCENT, markersize=8, label='+ LoRA on q, v (rank 16)')
    fig.legend(handles=[probe, lora], loc='upper right', ncol=2, frameon=False, fontsize=9, bbox_to_anchor=(0.99, 1.02))
    fig.suptitle('Leave-one-dataset-out: LoRA adaptation of three foundation models', x=0.01, ha='left',
                 fontsize=12, color=INK, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / 'foundation_models_lora.png', dpi=200, bbox_inches='tight')
    plt.close(fig)


def codec_augmentation():
    e = pd.read_csv(ROOT / 'results/d1_strength_analysis/estimands_mean.csv')
    arms = {'randmask': 'Random DCT mask', 'invmask': 'Inverted-table DCT mask', 'qmask': 'JPEG-table DCT mask',
            'realh264': 'Real H.264 training', 'realjpeg': 'Real JPEG training'}
    rows = {c: e[e.condition == c].set_index('estimand') for c in ('h264_crf40', 'clean')}
    order = sorted(arms, key=lambda a: rows['h264_crf40'].loc[f'{a} - noaug', 'estimate'])
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.0), sharey=True, gridspec_kw=dict(wspace=0.08))
    panels = [('h264_crf40', 'Gain under heavy compression (H.264 CRF 40)'), ('clean', 'Change on uncompressed video')]
    for ax, (cond, title) in zip(axes, panels):
        style(ax)
        ax.axvline(0, color=MUTED, linewidth=1, zorder=1)
        for y, a in enumerate(order):
            r = rows[cond].loc[f'{a} - noaug']
            color = ACCENT if a == 'randmask' else MUTED
            ax.plot([r.lo95, r.hi95], [y, y], color=color, linewidth=2, solid_capstyle='round', zorder=2)
            ax.scatter(r.estimate, y, s=70, color=color, edgecolor=SURFACE, linewidth=2, zorder=3)
            ax.annotate(f'{r.estimate:+.3f}', (r.hi95, y), xytext=(6, 0), textcoords='offset points', va='center',
                        fontsize=9, color=INK if a == 'randmask' else INK2)
        ax.set_title(title, loc='left', pad=8)
        ax.set_xlabel('AUC difference vs. retrained no-augmentation model')
        ax.set_xlim(-0.06, 0.085)
    axes[0].set_yticks(range(len(order)), [arms[a] for a in order])
    fig.suptitle('Codec-aware training of EVA-02 + LoRA (mean over 3 held-out datasets, 95% CI)', x=0.01,
                 ha='left', fontsize=12, color=INK, y=1.03)
    fig.savefig(OUT / 'codec_aware_training.png', dpi=200, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    OUT.mkdir(exist_ok=True)
    foundation_models()
    codec_augmentation()
    print('wrote', sorted(p.name for p in OUT.glob('*.png')))
