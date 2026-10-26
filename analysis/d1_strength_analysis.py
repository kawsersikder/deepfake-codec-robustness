"""Pre-registered 2x2 analysis of the strength-matched run (RESEARCH_PLAN_MASTER section 6, notebook Cell 0).

Ordering {JPEG table, uniform} x strength {qmask's, randmask's}: qmask, randmask (session 1), qmask_highE,
randmask_lowE (session 3). Primary endpoint H.264 CRF 40, secondary JPEG q20, mean over the 3 folds.
  ordering    = ((qmask_highE - randmask) + (qmask - randmask_lowE)) / 2
  strength    = ((qmask_highE - qmask) + (randmask - randmask_lowE)) / 2
  interaction = (qmask_highE - randmask) - (qmask - randmask_lowE)
One cluster-bootstrap resample shared by all arms within a fold; interval widened by training noise
sigma_run = 0.008 per run (variance k * sigma^2 / 3 over 3 folds: k = 1 main effects, 4 interaction, 2 pairs).
Decision: "profile worse/better" if the ordering 95% CI excludes 0 and |ordering| >= 0.012;
"strength explains it" if the ordering 90% CI lies inside +/-0.010 and the strength 95% CI lies above 0;
otherwise inconclusive (seed 43 decides).

Caveat found on 26 Oct: session 3 drew its training frames from a different file order (Kaggle now lists
files alphabetically), so its arms trained on a different data sample than sessions 1 and 2 (FF++: 46% of
videos in a different split; FF++ val also changed). Cross-session contrasts therefore carry a data-resample
effect that the sigma_run widening was not designed for. Because session 3 holds one diagonal of the 2x2 and
session 1 the other, the interaction is exactly 2 x the session offset (pure confound, never report it); the
ordering and strength main effects cancel a constant offset but not an arm-dependent one. FF++ group names differ
between sessions (same partition): every arm's group_id is dropped and clusters come from session 1's plain file,
after check_partition asserts the partitions are equal.
Run from the repository root: python analysis/d1_strength_analysis.py (torch is stubbed; numpy, pandas, scikit-learn)."""
import json
import re
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / 'results/d1_strength_analysis'
torch = MagicMock(); torch.nn.Module = object; torch.utils.data.Dataset = object
sys.modules.update({'torch': torch, 'torch.nn': torch.nn, 'torch.nn.functional': torch.nn.functional,
                    'torch.utils': torch.utils, 'torch.utils.data': torch.utils.data})
import importlib.util  # noqa: E402
spec = importlib.util.spec_from_file_location('codec7', ROOT / 'src/codec7.py')
C7 = importlib.util.module_from_spec(spec); spec.loader.exec_module(C7)

S1, S2, S3 = ROOT / 'results/d1_session1', ROOT / 'results/d1_session2', ROOT / 'results/d1_session3_strength'
SOURCE = {'plain': S1, 'realh264': S1, 'qmask': S1, 'randmask': S1, 'noaug': S2, 'invmask': S2, 'realjpeg': S2,
          'qmask_highE': S3, 'randmask_lowE': S3, 'invmask_highE': S3}
FOLDS = ['dfdc', 'ffpp', 'celebdf']
CONDS = ['clean', 'resize224', 'jpeg50', 'jpeg20', 'h264_crf23', 'h264_crf40']
KEYS = ['dataset', 'video_id', 'label']
N_BOOT, SIGMA = 1000, 0.008
Z95, Z90 = 1.959964, 1.644854


def load(arm, fold, cond):
    p = C7.read_pred(SOURCE[arm] / f'pred_{arm}_{fold}_{cond}.csv')
    return p.drop(columns='group_id')


def clusters(fold, identity=False):
    """Session 1 cluster names per video (FF++ identity groups; Celeb-DF / DFDC one per video).
    identity=True: Celeb-DF clustered by the first celebrity id in the name (sensitivity check)."""
    ref = C7.read_pred(S1 / f'pred_plain_{fold}_clean.csv').set_index('video_id').group_id
    if identity and fold == 'celebdf':
        ref = pd.Series({v: (m.group(0) if (m := re.search(r'id\d+', v.split('::')[-1])) else v) for v in ref.index})
    return ref


def check_partition():
    """Session 3's FF++ group names must describe the same partition as session 1's."""
    a = C7.read_pred(S1 / 'pred_plain_ffpp_clean.csv').set_index('video_id').group_id
    b = C7.read_pred(S3 / 'pred_plain_ffpp_clean.csv').set_index('video_id').group_id.loc[a.index]
    n = len(set(zip(a, b)))
    assert n == a.nunique() == b.nunique(), 'FF++ partitions differ between sessions'
    for cond in CONDS:   # plain itself must be identical in both sessions
        x, y = load('plain', 'ffpp', cond).set_index('video_id'), C7.read_pred(S3 / f'pred_plain_ffpp_{cond}.csv').set_index('video_id')
        assert np.array_equal(x.probability.loc[a.index].to_numpy(), y.probability.loc[a.index].to_numpy())
    return n


def fold_boot(fold, cond, arms, identity=False):
    """Point AUC and bootstrap AUCs (shared resamples) for every arm on one fold and condition."""
    d = None
    for arm in arms:
        p = load(arm, fold, cond).rename(columns={'probability': arm})
        d = p if d is None else d.merge(p, on=KEYS, validate='one_to_one')
    n = len(load(arms[0], fold, cond))
    assert len(d) == n, f'{fold}/{cond}: arms scored different videos'
    cl = clusters(fold, identity).loc[d.video_id].to_numpy()
    w = C7.cluster_weights(pd.Series(d.dataset.to_numpy()) + '||' + pd.Series(cl), N_BOOT, 42)
    y = d.label.to_numpy() == 1
    point = {a: float(C7.weighted_auc(y, d[a])[0]) for a in arms}
    boot = {a: C7.weighted_auc(y, d[a], w) for a in arms}
    return point, boot


ESTIMANDS = {   # name: (coefficients over arms, training-noise variance multiplier k)
    'ordering': ({'qmask_highE': .5, 'randmask': -.5, 'qmask': .5, 'randmask_lowE': -.5}, 1),
    'strength': ({'qmask_highE': .5, 'qmask': -.5, 'randmask': .5, 'randmask_lowE': -.5}, 1),
    'interaction': ({'qmask_highE': 1, 'randmask': -1, 'qmask': -1, 'randmask_lowE': 1}, 4),
}
PAIRS = [('qmask_highE', 'randmask'), ('randmask_lowE', 'qmask'), ('invmask_highE', 'randmask'),
         ('qmask_highE', 'invmask_highE'), ('qmask_highE', 'qmask'), ('randmask', 'randmask_lowE'),
         ('invmask', 'invmask_highE'), ('qmask', 'randmask')] + [
        (a, 'noaug') for a in ['qmask', 'randmask', 'invmask', 'qmask_highE', 'randmask_lowE', 'invmask_highE',
                               'realh264', 'realjpeg', 'plain']]
for a, b in PAIRS:
    ESTIMANDS[f'{a} - {b}'] = ({a: 1, b: -1}, 2)
ARMS = sorted({a for coef, _ in ESTIMANDS.values() for a in coef})


def analyse(cond, identity=False):
    rows, per_fold_boot = [], {}
    for fold in FOLDS:
        point, boot = fold_boot(fold, cond, ARMS, identity)
        for name, (coef, k) in ESTIMANDS.items():
            est = sum(c * point[a] for a, c in coef.items())
            bt = sum(c * boot[a] for a, c in coef.items())
            per_fold_boot.setdefault(name, []).append(bt)
            se_b = float(np.nanstd(bt, ddof=1))
            se_t = float(np.sqrt(se_b ** 2 + k * SIGMA ** 2))   # one fold: k * sigma^2
            rows.append(dict(condition=cond, fold=fold, estimand=name, estimate=est, se_boot=se_b, se_total=se_t,
                             lo95=est - Z95 * se_t, hi95=est + Z95 * se_t))
    folds = pd.DataFrame(rows)
    means = []
    for name, (coef, k) in ESTIMANDS.items():
        f = folds[folds.estimand == name]
        est = float(f.estimate.mean())
        bt = np.nanmean(np.vstack(per_fold_boot[name]), axis=0)   # folds are independent: average the draws
        se_b = float(np.nanstd(bt, ddof=1))
        se_t = float(np.sqrt(se_b ** 2 + k * SIGMA ** 2 / 3))
        means.append(dict(condition=cond, estimand=name, estimate=est, se_boot=se_b, se_total=se_t,
                          lo95=est - Z95 * se_t, hi95=est + Z95 * se_t, lo90=est - Z90 * se_t, hi90=est + Z90 * se_t,
                          folds_positive=int((f.estimate > 0).sum()), folds_ci_above_0=int((f.lo95 > 0).sum()),
                          folds_ci_below_0=int((f.hi95 < 0).sum()),
                          per_fold=' / '.join(f'{v:+.4f}' for v in f.estimate)))
    return folds, pd.DataFrame(means)


def decide(m):
    o = m.set_index('estimand').loc['ordering']
    s = m.set_index('estimand').loc['strength']
    if (o.lo95 > 0 or o.hi95 < 0) and abs(o.estimate) >= 0.012:
        return 'JPEG-table profile ' + ('BETTER' if o.estimate > 0 else 'WORSE') + ' than uniform at equal strength'
    if o.lo90 > -0.010 and o.hi90 < 0.010 and s.lo95 > 0:
        return 'STRENGTH explains the gap (ordering equivalent within +/-0.010, strength effect > 0)'
    return 'INCONCLUSIVE (seed 43 decides)'


def main():
    n = check_partition()
    print(f'FF++: session 3 group names describe the same {n}-group partition as session 1; plain identical in both')
    auc = []
    for arm, root in SOURCE.items():
        for fold in FOLDS:
            for cond in CONDS:
                auc.append(dict(arm=arm, fold=fold, condition=cond, auc=C7.macro_auc(load(arm, fold, cond)),
                                session={S1: 1, S2: 2, S3: 3}[root]))
    auc = pd.DataFrame(auc)
    auc.to_csv(HERE / 'auc_all_arms.csv', index=False)
    order = ['plain', 'noaug', 'realh264', 'realjpeg', 'qmask', 'qmask_highE', 'randmask_lowE', 'randmask',
             'invmask_highE', 'invmask']
    wide = auc.pivot_table(index='arm', columns='condition', values='auc')[CONDS].loc[order]
    wide.to_csv(HERE / 'auc_mean_over_folds.csv')
    out, decisions = {}, {}
    for identity in (False, True):
        tag = '_celebdf_identity' if identity else ''
        F, M = [], []
        for cond in CONDS:
            f, m = analyse(cond, identity)
            F.append(f); M.append(m)
            decisions[(cond, identity)] = decide(m)
        F, M = pd.concat(F), pd.concat(M)
        F.to_csv(HERE / f'estimands_per_fold{tag}.csv', index=False)
        M.to_csv(HERE / f'estimands_mean{tag}.csv', index=False)
        out[identity] = M
    pd.set_option('display.width', 220)
    print('\nMean AUC over 3 folds\n', wide.round(4).to_string())
    for cond in ['h264_crf40', 'jpeg20', 'h264_crf23', 'clean']:
        m = out[False][out[False].condition == cond]
        print(f'\n==== {cond}: {decisions[(cond, False)]}  |  Celeb-DF clustered by identity: {decisions[(cond, True)]}')
        print(m[['estimand', 'estimate', 'lo95', 'hi95', 'lo90', 'hi90', 'per_fold', 'folds_ci_above_0', 'folds_ci_below_0']]
              .round(4).to_string(index=False))
    json.dump({f'{c}{" (Celeb-DF identity clusters)" if i else ""}': d for (c, i), d in decisions.items()},
              open(HERE / 'decisions.json', 'w'), indent=2)


if __name__ == '__main__':
    main()
