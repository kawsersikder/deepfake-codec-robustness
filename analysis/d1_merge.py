"""Merge D1 session 1 (results_d1) and session 2 (results_d1_session2) and compute the comparisons
that need both: every arm against the retrained no-augmentation control (noaug), the mask-type
contrasts, a same-config replicate (realjpeg DFDC ran in both sessions) and gap closure from noaug.
Uses the notebook's own codec7 bootstrap (torch stubbed; only numpy/pandas code paths run)."""
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
S1, S2, OUT = ROOT / 'results/d1_session1', ROOT / 'results/d1_session2', ROOT / 'results/d1_merged_analysis'
torch = MagicMock(); torch.nn.Module = object; torch.utils.data.Dataset = object
for name, mod in {'torch': torch, 'torch.nn': torch.nn, 'torch.nn.functional': torch.nn.functional,
                  'torch.utils': torch.utils, 'torch.utils.data': torch.utils.data}.items():
    sys.modules[name] = mod
sys.path.insert(0, str(ROOT / 'src'))
import codec7 as C7  # noqa: E402

FOLDS = ['dfdc', 'ffpp', 'celebdf']
CONDS = ['clean', 'resize224', 'jpeg50', 'jpeg20', 'h264_crf23', 'h264_crf40']
SEED_SD, N_BOOT = 0.006, 1000
# Which session's predictions stand for each arm. plain was scored identically in both (checked below);
# realjpeg uses session 2 on every fold (same session as noaug), session 1's DFDC run is the replicate.
SOURCE = {'plain': S1, 'realh264': S1, 'qmask': S1, 'randmask': S1,
          'noaug': S2, 'invmask': S2, 'realjpeg': S2}


def pred(arm, fold, cond, root=None):
    return C7.read_pred((root or SOURCE[arm]) / f'pred_{arm}_{fold}_{cond}.csv')


def main():
    OUT.mkdir(exist_ok=True)
    # 0. plain must be byte-for-byte the same scoring in both sessions
    same = all((S1 / f'pred_plain_{f}_{c}.csv').read_bytes() == (S2 / f'pred_plain_{f}_{c}.csv').read_bytes()
               for f in FOLDS for c in CONDS)
    print('plain predictions identical in both sessions:', same)

    # 1. AUC table, every arm, fold and condition
    rows = []
    for arm, root in SOURCE.items():
        for f in FOLDS:
            for c in CONDS:
                p = pred(arm, f, c)
                rows.append(dict(arm=arm, fold=f, condition=c, auc=C7.macro_auc(p), videos=len(p),
                                 session=1 if root == S1 else 2))
    auc = pd.DataFrame(rows)
    auc.to_csv(OUT / 'auc_all_arms.csv', index=False)
    wide = auc.pivot_table(index='arm', columns='condition', values='auc', aggfunc='mean')[CONDS]
    wide = wide.loc[['plain', 'noaug', 'realh264', 'realjpeg', 'qmask', 'randmask', 'invmask']]
    wide.to_csv(OUT / 'auc_mean_over_folds.csv')

    # 2. paired comparisons
    pairs = ([(a, 'noaug') for a in ['plain', 'realh264', 'realjpeg', 'qmask', 'randmask', 'invmask']] +
             [('noaug', 'plain'), ('qmask', 'randmask'), ('invmask', 'randmask'), ('invmask', 'qmask'),
              ('realh264', 'realjpeg'), ('randmask', 'realh264'), ('qmask', 'realh264')])
    rows = []
    for a, b in pairs:
        for f in FOLDS:
            for c in CONDS:
                r = C7.paired_bootstrap(pred(a, f, c), pred(b, f, c), N_BOOT, 42)
                rows.append(dict(comparison=f'{a} - {b}', fold=f, condition=c, delta=r['delta'], lo=r['ci_low'],
                                 hi=r['ci_high'], videos=r['videos'],
                                 ci_excludes_0=bool(r['ci_low'] > 0 or r['ci_high'] < 0),
                                 beyond_seed_noise=bool(abs(r['delta']) >= 2 * SEED_SD)))
    pdl = pd.DataFrame(rows)
    pdl.to_csv(OUT / 'paired_deltas.csv', index=False)
    mean = (pdl.groupby(['comparison', 'condition'], sort=False)
            .agg(mean_delta=('delta', 'mean'), folds_ci_above_0=('lo', lambda v: int((v > 0).sum())),
                 folds_ci_below_0=('hi', lambda v: int((v < 0).sum())))
            .reset_index())
    mean.to_csv(OUT / 'paired_deltas_mean.csv', index=False)

    # 3. replicate: realjpeg DFDC trained twice with identical settings (session 1 and session 2)
    rep = []
    for c in CONDS:
        r = C7.paired_bootstrap(pred('realjpeg', 'dfdc', c, S2), pred('realjpeg', 'dfdc', c, S1), N_BOOT, 42)
        rep.append(dict(condition=c, auc_session2=r['auc_a'], auc_session1=r['auc_b'], delta=r['delta'],
                        lo=r['ci_low'], hi=r['ci_high']))
    rep = pd.DataFrame(rep)
    rep.to_csv(OUT / 'replicate_realjpeg_dfdc.csv', index=False)

    # 4. gap closure measured from noaug: (mask - noaug) / (ref - noaug), same cluster resamples
    rows = []
    keys = ['dataset', 'video_id', 'group_id', 'label']
    for f in FOLDS:
        for c in ['h264_crf40', 'jpeg20']:
            for ref in ['realh264', 'realjpeg']:
                for m in ['qmask', 'randmask', 'invmask']:
                    p0, pm, pr = pred('noaug', f, c), pred(m, f, c), pred(ref, f, c)
                    d = (p0.merge(pm, on=keys, suffixes=('', '_m'), validate='one_to_one')
                         .merge(pr, on=keys, suffixes=('', '_r'), validate='one_to_one'))
                    y = d.label.to_numpy() == 1
                    w = C7.cluster_weights(d.dataset.astype(str) + '||' + d.group_id.astype(str), N_BOOT, 42)
                    a0, am, ar = (C7.weighted_auc(y, d[k], w) for k in ('probability', 'probability_m', 'probability_r'))
                    b0, bm, br = (C7.weighted_auc(y, d[k])[0] for k in ('probability', 'probability_m', 'probability_r'))
                    ref_lo, ref_hi = np.quantile(ar - a0, [0.025, 0.975])
                    defined = ref_lo > 0  # reference gain must be clearly positive
                    with np.errstate(divide='ignore', invalid='ignore'):
                        boot = (am - a0) / (ar - a0)
                    boot = boot[np.isfinite(boot)]
                    lo, hi = np.quantile(boot, [0.025, 0.975]) if defined else (np.nan, np.nan)
                    rows.append(dict(fold=f, condition=c, mask_arm=m, reference_arm=ref,
                                     mask_gain=bm - b0, reference_gain=br - b0, ref_lo=ref_lo, ref_hi=ref_hi,
                                     ratio=(bm - b0) / (br - b0) if defined else np.nan, ratio_lo=lo, ratio_hi=hi,
                                     note='defined' if defined else 'not defined: reference gain interval not above 0'))
    gap = pd.DataFrame(rows)
    gap.to_csv(OUT / 'gap_closure_from_noaug.csv', index=False)

    pd.set_option('display.width', 200)
    print('\nAUC, mean over 3 folds\n', wide.round(4).to_string())
    print('\nPaired deltas, mean over folds (folds with CI above / below 0)')
    show = mean[mean.condition.isin(['clean', 'jpeg20', 'h264_crf40'])]
    print(show.round(4).to_string(index=False))
    print('\nReplicate realjpeg DFDC, session 2 - session 1\n', rep.round(4).to_string(index=False))
    print('\nGap closure from noaug (defined rows)\n',
          gap[gap.note == 'defined'].round(3).to_string(index=False))
    print('\nper-fold, h264_crf40, against noaug')
    print(pdl[(pdl.condition == 'h264_crf40') & pdl.comparison.str.endswith('- noaug')]
          .round(4).to_string(index=False))
    json.dump(dict(plain_identical=same, sources={k: str(v.name) for k, v in SOURCE.items()}, n_boot=N_BOOT,
                   seed=42, seed_sd=SEED_SD), open(OUT / 'merge_meta.json', 'w'), indent=2)


if __name__ == '__main__':
    main()
