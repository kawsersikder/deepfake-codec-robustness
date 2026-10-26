"""Phase 13: CLIP ViT-L/14, DINOv2 ViT-g/14 and EVA-02-L/14 with LoRA on the
attention projections, evaluated leave-one-dataset-out.

Two arms per backbone and held-out corpus, scored on the same frames:
  probe  frozen backbone, cached features, logistic-regression head
  lora   low-rank updates on the attention projections plus a linear head.
         The head starts from the probe (LP-FT), so epoch 0 equals the probe.
Checkpoints are chosen on source validation only. The held-out corpus is
scored once, after training. Definitions only: nothing trains on import.
"""
import faulthandler
import gc
import itertools
import json
import math
import os
import random
import shutil
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from PIL import Image
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import timm
from timm.data import resolve_model_data_config


CORPORA = ('celebdf', 'ffpp', 'dfdc')
BACKBONES = {
    'clip_l14': dict(
        timm_id='vit_large_patch14_clip_224.openai', kwargs={},
        note='CLIP ViT-L/14 (OpenAI), image-text contrastive pretraining, 304M params'),
    'dinov2_g14': dict(
        timm_id='vit_giant_patch14_dinov2.lvd142m', kwargs={'img_size': 224}, half=True,
        note='DINOv2 ViT-g/14, self-distillation on LVD-142M, 1.1B params; '
             'position embeddings resampled from 518 to 224; frozen weights stored in fp16'),
    'eva02_l14': dict(
        timm_id='eva02_large_patch14_224.mim_m38m', kwargs={},
        note='EVA-02-L/14, masked image modelling on Merged-38M, 304M params'),
}
BANK_COLS = ['path', 'dataset', 'video_id', 'group_id', 'label', 'split']
CKPT_OK = {}   # backbone -> gradient checkpointing passed the unit test

# Kaggle's T4 x2 can hang in DataParallel's GPU-to-GPU copies; route them through the host.
os.environ.setdefault('NCCL_P2P_DISABLE', '1')
# In a commit, 90 silent minutes means a hang: dump every thread's stack and exit, so the
# run fails with a traceback instead of idling to the 12 h limit. Off in interactive sessions.
COMMIT = os.environ.get('KAGGLE_KERNEL_RUN_TYPE') == 'Batch'
WATCHDOG_MIN = 90

try:   # kernel start: Kaggle's 12 h limit counts from the session start, not from this cell
    import psutil
    T0 = psutil.Process(os.getpid()).create_time()
except Exception:
    T0 = time.time()


def hours():
    return (time.time() - T0) / 3600


def log(msg):
    print(f'[{hours():6.2f}h] {msg}', flush=True)
    if COMMIT:
        try:
            faulthandler.dump_traceback_later(WATCHDOG_MIN * 60, exit=True, file=sys.__stderr__)
        except Exception:
            pass


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def free():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, Path):
        return str(x)
    return x


# ---------------------------------------------------------------- manifest

def validate_manifest(frame):
    required = {'path', 'dataset', 'video_id', 'group_id', 'label', 'split'}
    if required - set(frame):
        raise ValueError(f'Missing manifest columns: {required - set(frame)}')
    d = frame.copy()
    if d[list(required)].isna().any().any():
        raise ValueError('Manifest contains missing values')
    for col in ['dataset', 'video_id', 'group_id', 'split']:
        d[col] = d[col].astype(str).str.strip()
        if d[col].eq('').any():
            raise ValueError(f'Empty {col}')
    if not set(d.dataset) <= set(CORPORA):
        raise ValueError('Use canonical dataset names: celebdf, ffpp, dfdc')
    if not set(d.split) <= {'train', 'val', 'test'}:
        raise ValueError('Use existing train/val/test assignments')
    if not set(d.label) <= {0, 1}:
        raise ValueError('Labels must be 0=real, 1=fake')
    d['path'] = d.path.map(os.path.normpath)   # string only: touching 230k files costs hours in a commit
    if d.path.duplicated().any():
        raise ValueError('Duplicate frame paths; use one row per file')
    if (d.groupby(['dataset', 'video_id']).label.nunique() > 1).any():
        raise ValueError('A video has conflicting labels')
    if (d.groupby(['dataset', 'video_id']).group_id.nunique() > 1).any():
        raise ValueError('A video has multiple identity/source groups')
    for key in ['video_id', 'group_id']:
        if (d.groupby(['dataset', key]).split.nunique() > 1).any():
            raise ValueError(f'{key} leakage across original splits')
    # Paths come from a directory listing; spot-check a sample instead of every file.
    missing = [p for p in d.path.sample(min(len(d), 200), random_state=0) if not os.path.isfile(p)]
    if missing:
        raise FileNotFoundError(f'{len(missing)} of 200 sampled frames are missing; first: {missing[0]}')
    return d


def diverse_cap(d, cap, seed):
    """Shuffle, then take one frame per video before any video gets a second."""
    d = d.sample(frac=1, random_state=seed).copy()
    d['_round'] = d.groupby(['dataset', 'video_id']).cumcount()
    return d.sort_values('_round', kind='stable').head(cap).drop(columns='_round')


def balanced_cap(d, cap, seed):
    return pd.concat([diverse_cap(d[d.label == y], cap // 2, seed) for y in (0, 1)])


def first_frames(d, k):
    return d.sort_values('path').groupby(['dataset', 'video_id'], group_keys=False).head(k)


def limit_videos(d, n, seed):
    """Smoke mode: at most n whole videos per (dataset, label)."""
    keep = (d[['dataset', 'video_id', 'label']].drop_duplicates()
            .sample(frac=1, random_state=seed).groupby(['dataset', 'label']).head(n))
    return d.merge(keep[['dataset', 'video_id']], on=['dataset', 'video_id'])


def _ids(frame, col):
    return set(zip(frame.dataset, frame[col]))


def fold_frames(d, held, cfg):
    """Train/val come from the two source corpora; test is every video of `held`.
    Frames are drawn with `data_seed`, so a new training seed keeps the same frames."""
    sources = [c for c in CORPORA if c != held]
    data_seed = cfg.get('data_seed', cfg['seed'])
    tr = d[d.dataset.isin(sources) & d.split.eq('train')]
    va = d[d.dataset.isin(sources) & d.split.eq('val')]
    te = d[d.dataset.eq(held)]
    if cfg['smoke']:
        tr, va, te = (limit_videos(x, cfg['smoke_videos'], data_seed) for x in (tr, va, te))
    train = pd.concat([balanced_cap(tr[tr.dataset.eq(s)], cfg['train_per_source'], data_seed)
                       for s in sources])
    val = first_frames(va, cfg['eval_frames_per_video'])
    test = first_frames(te, cfg['eval_frames_per_video'])
    for name, part, need in [('train', train, sources), ('val', val, sources), ('test', test, [held])]:
        for ds in need:
            if part[part.dataset.eq(ds)].label.nunique() != 2:
                raise ValueError(f'{held} fold: {name}/{ds} needs both classes')
    for col in ('video_id', 'group_id'):
        if _ids(train, col) & _ids(val, col):
            raise ValueError(f'{held} fold: train and val share a {col}')
    if held in set(train.dataset) | set(val.dataset):
        raise ValueError(f'{held} fold: held-out corpus leaked into training')
    return [x.reset_index(drop=True) for x in (train, val, test)]


# -------------------------------------------------------------------- data

class Frames(Dataset):
    def __init__(self, frame, tf):
        self.paths = frame.path.tolist()
        self.labels = frame.label.astype(float).tolist()
        self.tf = tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as im:
            x = self.tf(im.convert('RGB'))
        return x, self.labels[i], i


def loader(frame, tf, cfg, train=False):
    return DataLoader(Frames(frame, tf),
                      batch_size=cfg['batch_size'] if train else cfg['eval_batch_size'],
                      shuffle=train, drop_last=train, num_workers=cfg['workers'],
                      timeout=600 if cfg['workers'] else 0,   # a stuck worker raises instead of hanging
                      pin_memory=True, generator=torch.Generator().manual_seed(cfg['seed']))


def backbone_transforms(model, size):
    """Each backbone keeps its own pretraining normalisation; faces are resized, not cropped."""
    dc = resolve_model_data_config(model)
    resize = T.Resize((size, size), interpolation=T.InterpolationMode.BICUBIC, antialias=True)
    tail = [T.ToTensor(), T.Normalize(dc['mean'], dc['std'])]
    return (T.Compose([T.RandomHorizontalFlip(), resize] + tail),
            T.Compose([resize] + tail),
            dict(mean=list(dc['mean']), std=list(dc['std'])))


# -------------------------------------------------------------------- LoRA

class LoRAAdapter(nn.Module):
    """s * B A x with B = 0 at the start, so the wrapped layer is unchanged until trained."""
    def __init__(self, d_in, d_out, r, alpha, dropout):
        super().__init__()
        self.A = nn.Parameter(torch.empty(r, d_in))
        self.B = nn.Parameter(torch.zeros(d_out, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.scale = alpha / r
        self.drop = nn.Dropout(dropout)
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        return F.linear(F.linear(self.drop(x), self.A.to(x.dtype)), self.B.to(x.dtype)) * self.scale


class LoRALinear(nn.Module):
    """Frozen nn.Linear plus a low-rank update: W x + b + s B A x."""
    def __init__(self, base, r, alpha, dropout):
        super().__init__()
        self.base = base
        self.lora = LoRAAdapter(base.in_features, base.out_features, r, alpha, dropout)

    def forward(self, x):
        return self.base(x) + self.lora(x)


class LoRAFusedQKV(nn.Module):
    """Fused qkv projection; low-rank updates on the chosen q/k/v slices only."""
    def __init__(self, base, parts, r, alpha, dropout):
        super().__init__()
        assert base.out_features % 3 == 0
        self.base, self.d = base, base.out_features // 3
        self.lora = nn.ModuleDict({p: LoRAAdapter(base.in_features, self.d, r, alpha, dropout)
                                   for p in parts})

    def forward(self, x):
        out = self.base(x)
        delta = [self.lora[p](x) if p in self.lora else out.new_zeros(*out.shape[:-1], self.d)
                 for p in 'qkv']
        return out + torch.cat(delta, -1).to(out.dtype)


def inject_lora(model, targets, r, alpha, dropout):
    """Freeze the model and wrap every attention block's projections in place.

    timm ViTs (CLIP, DINOv2) fuse q/k/v into one `qkv` Linear; EVA-02 has
    q_proj/k_proj/v_proj. Target `o` is the attention output projection `proj`.
    """
    if not targets or not set(targets) <= set('qkvo'):
        raise ValueError("lora_targets must be letters from 'qkvo', e.g. 'qv'")
    for p in model.parameters():
        p.requires_grad_(False)
    wrapped = []
    for name, m in list(model.named_modules()):
        if isinstance(getattr(m, 'qkv', None), nn.Linear):
            parts = ''.join(p for p in 'qkv' if p in targets)
            if parts:
                m.qkv = LoRAFusedQKV(m.qkv, parts, r, alpha, dropout)
        elif all(isinstance(getattr(m, f'{p}_proj', None), nn.Linear) for p in 'qkv'):
            for p in 'qkv':
                if p in targets:
                    setattr(m, f'{p}_proj', LoRALinear(getattr(m, f'{p}_proj'), r, alpha, dropout))
        else:
            continue
        if 'o' in targets:
            if not isinstance(getattr(m, 'proj', None), nn.Linear):
                raise TypeError(f'{name}: no attention output projection to wrap')
            m.proj = LoRALinear(m.proj, r, alpha, dropout)
        wrapped.append(name)
    if len(wrapped) != len(model.blocks):
        raise RuntimeError(f'Wrapped {len(wrapped)} attention modules, model has {len(model.blocks)} blocks')
    return wrapped


def adapters(model):
    return [m for m in model.modules() if isinstance(m, LoRAAdapter)]


def set_checkpointing(backbone, on):
    backbone.set_grad_checkpointing(on)
    if on and getattr(backbone, '_p13_hook', None) is None:
        # Reentrant checkpointing needs an input that requires grad. The frozen
        # patch embedding produces none, and without this LoRA gets no gradient.
        backbone._p13_hook = backbone.patch_embed.register_forward_hook(
            lambda m, i, o: o.requires_grad_(True) if torch.is_grad_enabled() else o)


# ------------------------------------------------------------------ models

def create_backbone(key, pretrained=True, **override):
    spec = BACKBONES[key]
    try:
        model = timm.create_model(spec['timm_id'], pretrained=pretrained, num_classes=0,
                                  **{**spec['kwargs'], **override})
    except Exception as e:
        raise RuntimeError(f'Could not create {spec["timm_id"]} (timm {timm.__version__}). '
                           f'Check that Internet is ON in the notebook settings. {e}') from e
    # fp16 frozen weights halve the 1.1B model's memory. LoRA weights, added after this, stay fp32.
    return model.half() if pretrained and spec.get('half') else model


class Encoder(nn.Module):
    """Pooled features in fp32. Autocast sits inside forward so DataParallel replicas use it."""
    def __init__(self, backbone, amp=True):
        super().__init__()
        self.backbone, self.amp = backbone, amp
        self.in_dtype = next(backbone.parameters()).dtype

    def forward(self, x):
        if self.amp and x.is_cuda:
            with torch.autocast('cuda', dtype=torch.float16):
                return self.backbone(x).float()
        return self.backbone(x.to(self.in_dtype)).float()


class Detector(nn.Module):
    def __init__(self, encoder, dim):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(dim, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        return self.head(self.encoder(x)).squeeze(1)


def parallel(module, cfg):
    return nn.DataParallel(module) if cfg['multi_gpu'] else module


def build_detector(key, cfg, dim, probe=None):
    backbone = create_backbone(key)
    inject_lora(backbone, cfg['lora_targets'], cfg['lora_rank'], cfg['lora_alpha'], cfg['lora_dropout'])
    det = Detector(Encoder(backbone, cfg['amp']), dim)
    if probe is not None:
        with torch.no_grad():
            det.head.weight.copy_(torch.as_tensor(probe['w'], dtype=torch.float32)[None])
            det.head.bias.fill_(float(probe['b']))
    return det.to(cfg['device'])


def param_audit(det):
    total = sum(p.numel() for p in det.parameters())
    lora = sum(p.numel() for n, p in det.named_parameters() if '.lora.' in n)
    head = sum(p.numel() for p in det.head.parameters())
    trainable = sum(p.numel() for p in det.parameters() if p.requires_grad)
    if trainable != lora + head:
        raise RuntimeError('Something other than LoRA and the head is trainable')
    return dict(total_params=total, lora_params=lora, head_params=head,
                trainable_params=trainable, trainable_pct=100 * trainable / total)


# ------------------------------------------------------------- evaluation

def video_scores(frames, prob):
    d = frames[['dataset', 'video_id', 'group_id', 'label']].copy()
    d['probability'] = prob
    return d.groupby(['dataset', 'video_id', 'group_id', 'label'], as_index=False).probability.mean()


def macro_auc(v):
    return float(np.mean([roc_auc_score(g.label, g.probability) for _, g in v.groupby('dataset')]))


@torch.no_grad()
def predict(det, frames, tf, cfg):
    net = parallel(det, cfg).eval()
    prob = np.full(len(frames), np.nan)
    for x, _, idx in loader(frames, tf, cfg):
        prob[idx.numpy()] = torch.sigmoid(net(x.to(cfg['device'], non_blocking=True))).cpu().numpy()
    if not np.isfinite(prob).all():
        raise FloatingPointError('Non-finite predictions. Set CFG["amp"]=False (fp32, slower).')
    return video_scores(frames, prob)


def paired_delta(a, b, n_boot, seed):
    """AUC(b) - AUC(a) on identical videos; resample (dataset, group) clusters."""
    keys = ['dataset', 'video_id', 'group_id', 'label']
    d = a.merge(b, on=keys, validate='one_to_one', suffixes=('_a', '_b'))
    if not len(d) == len(a) == len(b):
        raise ValueError('The two arms did not score exactly the same videos')
    y, pa, pb = d.label.to_numpy(), d.probability_a.to_numpy(), d.probability_b.to_numpy()
    groups = list(d.groupby(['dataset', 'group_id']).indices.values())
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(n_boot):
        i = np.concatenate([groups[g] for g in rng.integers(0, len(groups), len(groups))])
        if len(np.unique(y[i])) == 2:
            samples.append(roc_auc_score(y[i], pb[i]) - roc_auc_score(y[i], pa[i]))
    if len(samples) < 0.9 * n_boot:
        raise ValueError('Too few valid bootstrap samples; inspect class/group balance')
    lo, hi = np.quantile(samples, [0.025, 0.975])
    return dict(delta=float(roc_auc_score(y, pb) - roc_auc_score(y, pa)),
                lo=float(lo), hi=float(hi))


# ----------------------------------------------------------- frozen probe

@torch.no_grad()
def extract_features(encoder, frames, tf, cfg, what):
    net = parallel(encoder, cfg).eval()
    out, t = None, time.time()
    dl = loader(frames, tf, cfg)
    for k, (x, _, idx) in enumerate(dl):
        f = net(x.to(cfg['device'], non_blocking=True)).cpu().numpy()
        if out is None:
            out = np.full((len(frames), f.shape[1]), np.nan, np.float32)
        out[idx.numpy()] = f
        if k % cfg['log_every'] == 0:
            log(f'  {what}: batch {k + 1}/{len(dl)}')
    if not np.isfinite(out).all():
        raise FloatingPointError(f'{what}: non-finite features. Set CFG["amp"]=False (fp32, slower).')
    log(f'  {what}: {len(frames):,} frames at {len(frames) / (time.time() - t):.0f} img/s')
    return out


def bank_features(key, bank, cfg, out, reuse_dirs):
    for d in [out, *map(Path, reuse_dirs)]:
        idx, feat = d / f'bank_{key}.csv', d / f'features_{key}.npy'
        if idx.is_file() and feat.is_file() and pd.read_csv(idx).path.tolist() == bank.path.tolist():
            log(f'Reusing cached {key} features from {d}')
            if d != out:
                shutil.copy(idx, out / idx.name)
                shutil.copy(feat, out / feat.name)
            return np.load(feat).astype(np.float32)
    backbone = create_backbone(key).to(cfg['device'])
    _, tf_eval, _ = backbone_transforms(backbone, cfg['img_size'])
    feats = extract_features(Encoder(backbone, cfg['amp']), bank, tf_eval, cfg, f'{key} feature bank')
    del backbone
    free()
    feats = feats.astype(np.float16)   # stored half precision; used identically fresh or reused
    np.save(out / f'features_{key}.npy', feats)
    bank[BANK_COLS].to_csv(out / f'bank_{key}.csv', index=False)
    return feats.astype(np.float32)


def fold_scaler(scaler, clf):
    """Logistic regression on standardised features -> one Linear on raw features."""
    w = clf.coef_[0] / scaler.scale_
    b = clf.intercept_[0] - np.sum(clf.coef_[0] * scaler.mean_ / scaler.scale_)
    return w.astype(np.float32), float(b)


def probe_prob(feats, probe):
    return expit(feats.astype(np.float64) @ probe['w'].astype(np.float64) + probe['b'])


def fit_probe(f_tr, y_tr, f_va, val, cfg):
    scaler = StandardScaler().fit(f_tr)
    x_tr, x_va = scaler.transform(f_tr), scaler.transform(f_va)
    grid, best = [], None
    for C in cfg['probe_C']:
        clf = LogisticRegression(C=C, max_iter=5000, class_weight='balanced').fit(x_tr, y_tr)
        auc = macro_auc(video_scores(val, clf.predict_proba(x_va)[:, 1]))
        grid.append(dict(C=C, val_auc=auc))
        if best is None or auc > best[1]:
            best = (clf, auc, C)
    w, b = fold_scaler(scaler, best[0])
    return dict(C=best[2], val_auc=best[1], w=w, b=b, grid=grid)


def run_probes(key, splits, folds, cfg, out, reuse_dirs):
    parts = [x for held in folds for x in splits[held]]
    bank = (pd.concat(parts).drop_duplicates('path').sort_values('path').reset_index(drop=True))
    log(f'Feature bank: {len(bank):,} frames from {bank.video_id.nunique():,} videos')
    feats = bank_features(key, bank, cfg, out, reuse_dirs)
    row_of = pd.Series(np.arange(len(bank)), index=bank.path)

    def get(frame):
        return feats[row_of.loc[frame.path].to_numpy()]

    probes, rows = {}, []
    for held in folds:
        train, val, test = splits[held]
        t = time.time()
        p = fit_probe(get(train), train.label.to_numpy(), get(val), val, cfg)
        p['val_pred'] = video_scores(val, probe_prob(get(val), p))
        p['test_pred'] = video_scores(test, probe_prob(get(test), p))
        p['test_auc'] = macro_auc(p['test_pred'])
        p['test_pred'].to_csv(out / f'pred_probe_{held}.csv', index=False)
        rows.append(dict(backbone=key, held_out=held, C=p['C'], val_auc=p['val_auc'],
                         test_auc=p['test_auc'], train_frames=len(train),
                         test_videos=len(p['test_pred']), fit_sec=round(time.time() - t, 1)))
        probes[held] = p
        log(f'  probe {held}: C={p["C"]:g} source-val AUC {p["val_auc"]:.4f} | '
            f'held-out AUC {p["test_auc"]:.4f}')
    pd.DataFrame(rows).to_csv(out / f'probe_{key}.csv', index=False)
    return probes, feats.shape[1]


# ----------------------------------------------------- memory and runtime

def _is_oom(e):
    return isinstance(e, torch.cuda.OutOfMemoryError) or 'out of memory' in str(e).lower()


def choose_memory_mode(det, batch, cfg, allow_ckpt):
    """Pick checkpointing and a micro-batch whose training step fits on GPU 0.

    Gradients are accumulated over micro-batches up to the full batch, so the
    optimiser update matches the other backbones. Smaller micro-batches are
    tried before checkpointing, which costs an extra forward pass.
    """
    x, y = batch
    total = torch.cuda.get_device_properties(0).total_memory
    sizes = [m for m in (len(x), len(x) // 2, len(x) // 4) if m >= 2]
    plans = [(False, m) for m in sizes] + ([(True, m) for m in sizes] if allow_ckpt else [])
    det.train()
    for ckpt, micro in plans:
        set_checkpointing(det.encoder.backbone, ckpt)
        det.zero_grad(set_to_none=True)
        free()
        torch.cuda.reset_peak_memory_stats(0)
        base = torch.cuda.memory_allocated(0)
        mode = (f'micro-batch {micro} of {len(x)}, checkpointing {"on" if ckpt else "off"}, '
                f'weights and state {base / 2**30:.1f} GiB')
        try:
            F.binary_cross_entropy_with_logits(parallel(det, cfg)(x[:micro]), y[:micro]).backward()
            peak = torch.cuda.max_memory_allocated(0)
            det.zero_grad(set_to_none=True)
            if peak < 0.85 * total:
                log(f'  memory: {mode}, peak {peak / 2**30:.1f} of {total / 2**30:.1f} GiB on GPU 0')
                return ckpt, micro, peak
            log(f'  memory: {mode}, peak {peak / 2**30:.1f} GiB is too close to the limit')
        except RuntimeError as e:
            if not _is_oom(e):
                raise
            log(f'  memory: {mode}, out of memory')
        det.zero_grad(set_to_none=True)
        free()
    raise RuntimeError(f'Even a micro-batch of {sizes[-1]} does not fit on GPU 0')


def benchmark(det, batches, val, tf_eval, cfg, micro):
    """Train speed excludes data loading (optimistic); eval speed includes it."""
    if len(batches) < 2:
        raise ValueError('Need at least two training batches to benchmark')
    net = parallel(det, cfg)
    det.train()
    per_img = []
    for i, (x, y) in enumerate(batches):
        torch.cuda.synchronize()
        t = time.time()
        for xc, yc in zip(x.split(micro), y.split(micro)):
            F.binary_cross_entropy_with_logits(net(xc), yc).backward()
        torch.cuda.synchronize()
        if i:
            per_img.append((time.time() - t) / len(x))
        det.zero_grad(set_to_none=True)
    n = min(len(val), cfg['eval_batch_size'] * 6)
    t = time.time()
    predict(det, val.head(n), tf_eval, cfg)
    return dict(train_img_s=float(1 / np.mean(per_img)), eval_img_s=float(n / (time.time() - t)))


def project_fold_hours(sizes, bench, cfg):
    n_train, n_val, n_test = sizes
    sec = (cfg['epochs'] * (n_train / bench['train_img_s'] + n_val / bench['eval_img_s'])
           + (n_val + n_test) / bench['eval_img_s'])
    return 1.15 * sec / 3600


# ---------------------------------------------------------------- testing

def unit_tests(key, cfg):
    """CPU only, random weights, no downloads. Uses the exact timm class at depth 2."""
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    x = torch.randn(5, 8)
    base = nn.Linear(8, 6)
    w = LoRALinear(base, 4, 8, 0.0)
    assert torch.allclose(w(x), base(x)), 'LoRALinear must start as the base layer'
    qkv = nn.Linear(8, 24)
    f = LoRAFusedQKV(qkv, 'qv', 4, 8, 0.0)
    with torch.no_grad():
        for a in f.lora.values():
            a.B.normal_()
        moved = (f(x) - qkv(x)).abs().reshape(5, 3, 8).amax(dim=(0, 2))
    assert moved[0] > 0 and moved[1] == 0 and moved[2] > 0, 'qv LoRA must leave k untouched'

    m = create_backbone(key, pretrained=False, depth=2).eval()
    xin = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        ref = m(xin)
    names = inject_lora(m, cfg['lora_targets'], 4, 8, 0.0)
    ads = adapters(m)
    with torch.no_grad():
        out = m(xin)
    assert len(names) == 2 and ads, f'{key}: LoRA not attached to both blocks'
    assert all(a.calls > 0 for a in ads), f'{key}: attention forward bypasses LoRA'
    assert torch.allclose(ref, out, atol=1e-5), f'{key}: zero-initialised LoRA changed the output'
    assert not [n for n, p in m.named_parameters() if p.requires_grad and '.lora.' not in n]
    head = nn.Linear(ref.shape[1], 1)
    for ckpt in (False, True):
        m.train()
        m.zero_grad(set_to_none=True)
        try:
            set_checkpointing(m, ckpt)
            head(m(xin)).sum().backward()
        except Exception as e:
            if not ckpt:
                raise
            CKPT_OK[key] = False
            set_checkpointing(m, False)
            print(f'WARNING: gradient checkpointing unavailable for {key} ({type(e).__name__}: {e}). '
                  f'Training will run without it; lower batch_size if memory runs out.')
            continue
        dead = [a for a in ads if a.B.grad is None or not a.B.grad.abs().sum() > 0]
        assert not dead, f'{key}: checkpointing={ckpt} leaves {len(dead)} adapters without gradient'
        if ckpt:
            CKPT_OK[key] = True

    X = rng.normal(size=(300, 12)) * rng.uniform(0.1, 5, 12) + rng.normal(size=12)
    y = (X[:, 0] + rng.normal(size=300) > X[:, 0].mean()).astype(int)
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(max_iter=2000).fit(sc.transform(X), y)
    wv, bv = fold_scaler(sc, clf)
    assert np.allclose(expit(X @ wv.astype(np.float64) + bv),
                       clf.predict_proba(sc.transform(X))[:, 1], atol=1e-5), 'probe folding is wrong'

    rows = [dict(path=f'/x/{ds}/{v}/{k}.png', dataset=ds, video_id=f'{ds}::{v}',
                 group_id=f'{ds}::{v}', label=v % 2,
                 split='train' if v < 24 else ('val' if v < 32 else 'test'))
            for ds in CORPORA for v in range(40) for k in range(10)]
    toy = pd.DataFrame(rows)
    tcfg = dict(cfg, smoke=False, train_per_source=100, eval_frames_per_video=4)
    for held in CORPORA:
        tr, va, te = fold_frames(toy, held, tcfg)
        assert held not in set(tr.dataset) | set(va.dataset)
        assert te.video_id.nunique() == 40 and len(te) == 160
        assert (tr.groupby('dataset').size() == 100).all() and (tr.label.value_counts() == 100).all()
        assert tr.groupby('dataset').video_id.nunique().min() == 24
    v = video_scores(te, rng.uniform(size=len(te)))
    r = paired_delta(v, v, 50, 0)
    assert r['delta'] == 0 and r['lo'] == r['hi'] == 0
    print(f'PASS: unit tests ({key} architecture at depth 2, LoRA wiring, zero init, '
          f'gradients, probe folding, fold leakage, bootstrap)')


def preflight(key, cfg, splits, lora_folds, probe_folds, out):
    """Real pretrained weights: LoRA wiring, zero-init identity, gradients, memory, speed."""
    dev = cfg['device']
    first = splits[lora_folds[0] if lora_folds else probe_folds[0]]
    backbone = create_backbone(key).to(dev).eval()
    tf_train, tf_eval, norm = backbone_transforms(backbone, cfg['img_size'])
    x = torch.stack([tf_eval(Image.open(p).convert('RGB')) for p in first[0].path.head(4)]).to(dev)
    enc32 = Encoder(backbone, amp=False)
    with torch.no_grad():
        before = enc32(x)
    inject_lora(backbone, cfg['lora_targets'], cfg['lora_rank'], cfg['lora_alpha'], cfg['lora_dropout'])
    backbone.to(dev)   # the new LoRA weights are created on the CPU
    ads = adapters(backbone)
    with torch.no_grad():
        after = enc32(x)
        assert all(a.calls > 0 for a in ads), 'Some LoRA adapters never ran'
        assert torch.allclose(before, after, atol=1e-4, rtol=1e-4), 'Zero-initialised LoRA changed features'
        for a in ads:
            a.B.normal_(0, 0.02)
        moved = (enc32(x) - before).abs().max().item()
        for a in ads:
            a.B.zero_()
    assert moved > 1e-4, 'Perturbing LoRA did not change the features'

    dim = before.shape[1]
    det = Detector(Encoder(backbone, amp=False), dim).to(dev)
    with torch.no_grad():
        det.head.weight.normal_(0, 0.05)
    for ckpt in ((False, True) if CKPT_OK.get(key) else (False,)):
        set_checkpointing(backbone, ckpt)
        det.train()
        det.zero_grad(set_to_none=True)
        F.binary_cross_entropy_with_logits(det(x[:2]), torch.tensor([0., 1.], device=dev)).backward()
        dead = [a for a in ads if a.B.grad is None or not a.B.grad.abs().sum() > 0]
        assert not dead, f'checkpointing={ckpt}: {len(dead)} adapters got no gradient'
        assert all(torch.isfinite(p.grad).all() for p in det.parameters() if p.grad is not None)
    audit = param_audit(det)
    det.encoder.amp = cfg['amp']
    det.zero_grad(set_to_none=True)
    log(f'PASS: real {key}: {len(ads)} adapters on {len(backbone.blocks)} blocks all run, '
        f'zero init leaves features unchanged, gradients reach every adapter '
        f'(frozen weights {det.encoder.in_dtype})')
    log(f'  trainable {audit["trainable_params"]:,} of {audit["total_params"]:,} '
        f'({audit["trainable_pct"]:.2f}%), feature dim {dim}, '
        f'pooling {getattr(backbone, "global_pool", "?")!r}, normalisation {norm}')

    batches = [(xb.to(dev), yb.float().to(dev)) for xb, yb, _ in
               itertools.islice(loader(first[0], tf_train, cfg, train=True), 4)]
    ckpt, micro, peak = choose_memory_mode(det, batches[0], cfg, allow_ckpt=CKPT_OK.get(key, False))
    bench = benchmark(det, batches, first[1], tf_eval, cfg, micro)
    pooling = getattr(backbone, 'global_pool', '?')
    del det, backbone, batches, x
    free()

    n_bank = len(pd.concat([p for h in probe_folds for p in splits[h]]).drop_duplicates('path'))
    plan = [dict(step='feature bank (frozen probe)', frames=n_bank, hours=1.15 * n_bank / bench['eval_img_s'] / 3600)]
    for held in lora_folds:
        tr, va, te = splits[held]
        plan.append(dict(step=f'LoRA, {held} held out', frames=len(tr),
                         hours=project_fold_hours((len(tr), len(va), len(te)), bench, cfg)))
    plan = pd.DataFrame(plan)
    log(f'  speed: train {bench["train_img_s"]:.0f} img/s, eval {bench["eval_img_s"]:.0f} img/s '
        f'on {torch.cuda.device_count()} GPU(s)')
    print(plan.round(2).to_string(index=False))
    total = hours() + plan.hours.sum()
    print(f'Projected finish at {total:.1f} h of this session (limit {cfg["time_limit_h"]} h). '
          f'Folds that would cross the limit are skipped and can run in a later session.')
    result = dict(dim=dim, audit=audit, bench=bench, grad_ckpt=ckpt, micro_batch=micro, peak_gib=peak / 2**30,
                  normalisation=norm, pooling=str(pooling), plan=plan.to_dict('records'))
    (out / 'preflight.json').write_text(json.dumps(jsonable(result), indent=2))
    return result


# --------------------------------------------------------------- training

def signature(key, cfg):
    keys = ['seed', 'smoke', 'train_per_source', 'eval_frames_per_video', 'lora_rank', 'lora_alpha',
            'lora_dropout', 'lora_targets', 'epochs', 'batch_size', 'lr_lora', 'lr_head', 'weight_decay']
    return dict(backbone=key, data_seed=cfg.get('data_seed', cfg['seed']), **{k: cfg[k] for k in keys})


def reuse_fold(key, held, cfg, out, reuse_dirs):
    for d in map(Path, reuse_dirs):
        meta = d / f'lora_{held}.json'
        if meta.is_file():
            r = json.loads(meta.read_text())
            if r.get('status') == 'complete' and r.get('signature') == jsonable(signature(key, cfg)):
                for name in [meta.name, f'pred_lora_{held}.csv', f'history_{held}.csv', f'lora_{held}_best.pt']:
                    if (d / name).is_file():
                        shutil.copy(d / name, out / name)
                r['test_pred'] = pd.read_csv(out / f'pred_lora_{held}.csv', dtype={'video_id': str, 'group_id': str})
                log(f'Reusing finished LoRA fold {held} from {d}')
                return r
    return None


def run_lora_fold(key, held, split, probe, dim, cfg, out, bench, reuse_dirs):
    done = reuse_fold(key, held, cfg, out, reuse_dirs)
    if done:
        return done
    train, val, test = split
    projected = project_fold_hours((len(train), len(val), len(test)), bench, cfg)
    if hours() + projected > cfg['time_limit_h']:
        log(f'SKIP LoRA {held}: needs ~{projected:.1f} h, {cfg["time_limit_h"] - hours():.1f} h left. '
            f'Run it in a new session with this output in REUSE_DIRS.')
        return dict(status='skipped_time_budget', projected_h=projected)
    log(f'=== LoRA {key} | {held} held out | {len(train):,} train frames | ~{projected:.1f} h ===')
    seed_all(cfg['seed'])
    dev = cfg['device']
    det = build_detector(key, cfg, dim, probe)
    audit = param_audit(det)
    tf_train, tf_eval, _ = backbone_transforms(det.encoder.backbone, cfg['img_size'])
    train_dl = loader(train, tf_train, cfg, train=True)
    xb, yb, _ = next(iter(train_dl))
    ckpt, micro, _ = choose_memory_mode(det, (xb.to(dev), yb.float().to(dev)), cfg,
                                        allow_ckpt=CKPT_OK.get(key, False))
    del xb, yb
    with torch.no_grad():   # choose_memory_mode ran backward only; make sure nothing moved
        assert all(not a.B.abs().sum() for a in adapters(det))

    net = parallel(det, cfg)
    trainable = [p for p in det.parameters() if p.requires_grad]
    lora_p = [p for n, p in det.named_parameters() if p.requires_grad and not n.startswith('head.')]
    opt = torch.optim.AdamW([dict(params=lora_p, lr=cfg['lr_lora'], weight_decay=cfg['weight_decay']),
                             dict(params=list(det.head.parameters()), lr=cfg['lr_head'], weight_decay=0.0)])
    total = cfg['epochs'] * len(train_dl)
    warm = max(1, int(cfg['warmup_frac'] * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm
        else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    scaler = torch.amp.GradScaler('cuda', enabled=cfg['amp'])
    n_fake = int((train.label == 1).sum())
    pos_weight = torch.tensor(int((train.label == 0).sum()) / max(1, n_fake), device=dev)

    # Epoch 0 is the probe pushed through the network: a check on LoRA zero init and head folding.
    v0 = predict(det, val, tf_eval, cfg)
    gap = v0.merge(probe['val_pred'], on=['dataset', 'video_id', 'group_id', 'label'])
    gap = (gap.probability_x - gap.probability_y).abs()
    log(f'  epoch 0 vs probe on source val: mean |dp| {gap.mean():.5f}, max {gap.max():.4f}')
    if gap.mean() > 0.01 or gap.max() > 0.1:
        raise AssertionError('Epoch-0 LoRA model does not reproduce the probe; do not trust this fold')
    history = [dict(epoch=0, train_loss=np.nan, val_auc=macro_auc(v0), sec=0.0)]
    best_auc, best_epoch, best_state, best_val = -1.0, None, None, None
    status, t_fold = 'complete', time.time()
    for epoch in range(1, cfg['epochs'] + 1):
        det.train()
        t, losses = time.time(), []
        for step, (x, y, _) in enumerate(train_dl):
            x, y = x.to(dev, non_blocking=True), y.float().to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            batch_loss = 0.0
            for xc, yc in zip(x.split(micro), y.split(micro)):   # one chunk unless memory needs more
                loss = F.binary_cross_entropy_with_logits(net(xc), yc, pos_weight=pos_weight) * (len(xc) / len(x))
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'Non-finite loss at epoch {epoch} step {step}. '
                                             f'Try CFG["amp"]=False or a lower lr_lora.')
                scaler.scale(loss).backward()
                batch_loss += loss.item()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(trainable, cfg['grad_clip'])
            scaler.step(opt)
            scaler.update()
            sched.step()
            losses.append(batch_loss)
            if step % cfg['log_every'] == 0:
                log(f'  epoch {epoch} step {step + 1}/{len(train_dl)} loss {np.mean(losses[-50:]):.4f}')
        vp = predict(det, val, tf_eval, cfg)
        auc = macro_auc(vp)
        history.append(dict(epoch=epoch, train_loss=float(np.mean(losses)), val_auc=auc,
                            sec=round(time.time() - t, 1)))
        pd.DataFrame(history).to_csv(out / f'history_{held}.csv', index=False)
        log(f'  epoch {epoch}: train loss {np.mean(losses):.4f} | source-val AUC {auc:.4f}')
        if auc > best_auc:
            best_auc, best_epoch, best_val = auc, epoch, vp
            best_state = {n: p.detach().cpu().clone() for n, p in det.named_parameters() if p.requires_grad}
            torch.save(best_state, out / f'lora_{held}_best.pt')
        if hours() > cfg['time_limit_h'] and epoch < cfg['epochs']:
            status = 'truncated'
            log(f'  time limit reached after epoch {epoch}; scoring the best epoch so far')
            break
    _, unexpected = det.load_state_dict(best_state, strict=False)
    assert not unexpected
    test_pred = predict(det, test, tf_eval, cfg)
    test_pred.to_csv(out / f'pred_lora_{held}.csv', index=False)
    best_val.to_csv(out / f'pred_lora_val_{held}.csv', index=False)
    result = dict(status=status, signature=signature(key, cfg), held_out=held, best_epoch=best_epoch,
                  val_auc=best_auc, test_auc=macro_auc(test_pred), history=history, audit=audit,
                  grad_ckpt=ckpt, micro_batch=micro, train_hours=(time.time() - t_fold) / 3600)
    (out / f'lora_{held}.json').write_text(json.dumps(jsonable(result), indent=2))
    log(f'  LoRA {held}: best epoch {best_epoch} (source-val {best_auc:.4f}) | '
        f'held-out AUC {result["test_auc"]:.4f} | probe {probe["test_auc"]:.4f}')
    del det, net, opt, best_state
    free()
    result['test_pred'] = test_pred
    return result


# -------------------------------------------------------------- reference

def reference_predictions(held, test, cfg, out):
    """Phase 11 dual-stream and baseline checkpoints re-scored on these exact frames.
    Needs Cell 1 (create_model, load_checkpoint) and Cell 2 (DualNet, eval_tf)."""
    refs = {}
    for arm, fname in [('dualstream', f'model_dualstream_loo_{held}.pth'),
                       ('baseline', f'model_baseline_loo_{held}.pth')]:
        hits = [p for p in cfg['checkpoints'] if p.name == fname]
        if not hits:
            log(f'  reference {arm}/{held}: {fname} not attached, skipped')
            continue
        try:
            if arm == 'dualstream':
                model = DualNet(pretrained=False)
                model.load_state_dict(torch.load(hits[0], map_location='cpu', weights_only=True), strict=True)
            else:
                model = load_checkpoint(create_model(pretrained=False), hits[0])
        except Exception as e:
            log(f'  reference {arm}/{held}: could not load {hits[0]} ({e}); skipped')
            continue
        model = model.to(cfg['device']).eval()
        prob = np.full(len(test), np.nan)
        with torch.no_grad():
            for x, _, idx in loader(test, eval_tf, cfg):
                with torch.autocast('cuda', dtype=torch.float16, enabled=cfg['amp']):
                    logit = model(x.to(cfg['device'], non_blocking=True)).float().reshape(-1)
                prob[idx.numpy()] = torch.sigmoid(logit).cpu().numpy()
        if not np.isfinite(prob).all():
            log(f'  reference {arm}/{held}: non-finite outputs, skipped')
            continue
        refs[arm] = video_scores(test, prob)
        refs[arm].to_csv(out / f'pred_{arm}_{held}.csv', index=False)
        log(f'  reference {arm}/{held}: AUC {macro_auc(refs[arm]):.4f} on the same frames ({hits[0]})')
        del model
        free()
    return refs


# ------------------------------------------------------- earlier outputs

def find_input_dirs(filename, skip=()):
    """Folders under /kaggle/input holding `filename`, without listing the face-crop trees."""
    skip, hits = {str(p) for p in skip}, []
    for dp, dns, fns in os.walk('/kaggle/input'):
        if filename in fns:
            hits.append(Path(dp))
        if dp in skip or dp.count(os.sep) >= 9 or any(
                f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')) for f in fns):
            dns[:] = []
    return sorted(hits)


def crop_size_report(d, n, seed):
    """Face-crop sizes from a sample per corpus: decides whether DINOv2 above 224 px is worth testing."""
    rows = []
    for ds, g in d.groupby('dataset'):
        for path in g.path.sample(min(n, len(g)), random_state=seed):
            with Image.open(path) as im:
                rows.append(dict(dataset=ds, short_side=min(im.size), long_side=max(im.size)))
    s = pd.DataFrame(rows)
    return s.groupby('dataset').agg(
        sampled=('short_side', 'size'), short_side_p10=('short_side', lambda v: v.quantile(0.1)),
        short_side_median=('short_side', 'median'), short_side_p90=('short_side', lambda v: v.quantile(0.9)),
        share_at_least_448=('short_side', lambda v: float((v >= 448).mean()))).reset_index()


# ---------------------------------------------------------------- outputs

def summarise(key, probes, lora, refs, cfg, out):
    rows = []
    for held, p in probes.items():
        row = dict(backbone=key, held_out=held, test_videos=len(p['test_pred']),
                   probe_val_auc=p['val_auc'], probe_auc=p['test_auc'])
        r = lora.get(held)
        ok = bool(r) and r['status'] in ('complete', 'truncated')
        if ok:
            c = paired_delta(p['test_pred'], r['test_pred'], cfg['bootstrap'], cfg['seed'])
            row.update(lora_val_auc=r['val_auc'], lora_auc=r['test_auc'], lora_epoch=r['best_epoch'],
                       lora_minus_probe=c['delta'], lora_minus_probe_lo=c['lo'],
                       lora_minus_probe_hi=c['hi'])
        row['lora_status'] = r['status'] if r else 'not_run'
        for arm, pred in refs.get(held, {}).items():
            row[f'{arm}_auc'] = macro_auc(pred)
            if ok:
                c = paired_delta(pred, r['test_pred'], cfg['bootstrap'], cfg['seed'])
                row.update({f'lora_minus_{arm}': c['delta'], f'lora_minus_{arm}_lo': c['lo'],
                            f'lora_minus_{arm}_hi': c['hi']})
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(out / f'faculty_table_{key}.csv', index=False)
    return table


def make_plots(key, table, lora, out):
    import matplotlib.pyplot as plt
    arms = [c for c in ['probe_auc', 'lora_auc', 'dualstream_auc', 'baseline_auc'] if c in table]
    labels = {'probe_auc': 'frozen probe', 'lora_auc': 'LoRA',
              'dualstream_auc': 'dual-stream (Phase 11)', 'baseline_auc': 'baseline (Phase 11)'}
    fig, ax = plt.subplots(figsize=(7, 4))
    width = 0.8 / len(arms)
    for i, arm in enumerate(arms):
        ax.bar(np.arange(len(table)) + i * width, table[arm], width, label=labels[arm])
    ax.axhline(0.5, color='grey', lw=0.8, ls='--')
    ax.set_xticks(np.arange(len(table)) + 0.4 - width / 2)
    ax.set_xticklabels([f'{h} held out' for h in table.held_out])
    ax.set_ylim(0.4, 1.0)
    ax.set_ylabel('video-level AUC (same frames)')
    ax.set_title(f'{BACKBONES[key]["timm_id"]}: unseen-dataset AUC')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / f'heldout_auc_{key}.png', dpi=160)
    plt.close(fig)

    runs = {h: r for h, r in lora.items() if r.get('history')}
    if runs:
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
        for held, r in runs.items():
            h = pd.DataFrame(r['history'])
            axes[0].plot(h.epoch, h.val_auc, marker='o', label=f'{held} held out')
            axes[1].plot(h.epoch[1:], h.train_loss[1:], marker='o', label=f'{held} held out')
        axes[0].set(xlabel='epoch (0 = frozen probe)', ylabel='source-val AUC', title='Model selection')
        axes[1].set(xlabel='epoch', ylabel='train BCE', title='Training loss')
        axes[0].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out / f'lora_curves_{key}.png', dpi=160)
        plt.close(fig)


def write_metadata(key, cfg, pre, lora, split_origin, out):
    meta = dict(
        backbone=key, timm_id=BACKBONES[key]['timm_id'], note=BACKBONES[key]['note'],
        torch=torch.__version__, timm=timm.__version__,
        gpus=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        wall_hours=hours(), split_origin=split_origin,
        config={k: v for k, v in cfg.items() if k != 'checkpoints'},
        preflight=pre, lora={h: {k: v for k, v in r.items() if k != 'test_pred'} for h, r in lora.items()})
    (out / 'run_metadata.json').write_text(json.dumps(jsonable(meta), indent=2))


def zip_results(out):
    """Everything except the cached feature bank (kept in /kaggle/working for reuse)."""
    faulthandler.cancel_dump_traceback_later()   # the run is finished: disarm the hang watchdog
    archive = out.with_suffix('.zip')
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.rglob('*')):
            if p.is_file() and p.suffix != '.npy':
                z.write(p, p.relative_to(out.parent))
    return archive
