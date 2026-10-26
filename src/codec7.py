"""Phase 14b: codec-aware block-DCT masking, the full three-fold experiment.

Every arm continues the Phase 11 checkpoint of one leave-one-dataset-out fold on
the Phase 13 frames with the same schedule, then scores the held-out corpus clean,
after JPEG re-saving and after H.264 re-encoding. Masks act only on the input of
the dual-stream model's `fad` module and only in training; the RGB stream never
sees them. Checkpoints are chosen on source validation only.

Self-contained: model builders, the transform, the log function and the session
clock come in as arguments. Nothing trains on import.
"""
import faulthandler
import gc
import hashlib
import io
import json
import math
import multiprocessing
import os
import random
import shutil
import subprocess
import tempfile
import time
import traceback
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score


CORPORA = ('celebdf', 'ffpp', 'dfdc')
IMG_EXT = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
JPEG_LUMA = [
    [16, 11, 10, 16, 24, 40, 51, 61], [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56], [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77], [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101], [72, 92, 95, 98, 112, 100, 103, 99],
]
MASK_KINDS = ('qtable', 'uniform', 'inverted')
ARM_DEFS = {
    'control':        dict(model='dual', mask=None,     realjpeg=False, trained=True),
    'qmask':          dict(model='dual', mask='qtable', realjpeg=False, trained=True),
    'randmask':       dict(model='dual', mask='uniform', realjpeg=False, trained=True),
    'invmask':        dict(model='dual', mask='inverted', realjpeg=False, trained=True),
    'realjpeg':       dict(model='dual', mask=None,     realjpeg=True,  trained=True),
    'realjpeg_qmask': dict(model='dual', mask='qtable', realjpeg=True,  trained=True),
    'rgb_control':    dict(model='rgb',  mask=None,     realjpeg=False, trained=True),
    'rgb_realjpeg':   dict(model='rgb',  mask=None,     realjpeg=True,  trained=True),
}
ORIGINAL_DEFS = {
    'dual_original': dict(model='dual', mask=None, realjpeg=False, trained=False),
    'rgb_original':  dict(model='rgb',  mask=None, realjpeg=False, trained=False),
}
ARM_ORDER = list(ORIGINAL_DEFS) + list(ARM_DEFS)
# Phase 13b Part A: the untouched checkpoints on exactly these test frames (no flip TTA).
REFERENCE_AUC = {'dual_original': dict(celebdf=0.8081, ffpp=0.8719, dfdc=0.7402),
                 'rgb_original':  dict(celebdf=0.7759, ffpp=0.8320, dfdc=0.7414)}
COMPARISONS = [('qmask', 'control'), ('qmask', 'randmask'), ('qmask', 'invmask'), ('qmask', 'realjpeg'),
               ('realjpeg_qmask', 'realjpeg'), ('randmask', 'control'), ('realjpeg', 'control'),
               ('control', 'rgb_control'), ('rgb_realjpeg', 'rgb_control')]
ENDPOINTS = {('qmask', 'randmask', 'jpeg20'): 'primary', ('qmask', 'control', 'jpeg20'): 'primary',
             ('qmask', 'randmask', 'h264_crf40'): 'secondary', ('qmask', 'control', 'h264_crf40'): 'secondary'}
SIGNATURE_KEYS = ('seed', 'data_seed', 'smoke', 'smoke_videos', 'train_per_source', 'eval_frames_per_video',
                  'epochs', 'lr', 'weight_decay', 'batch_size', 'grad_clip', 'amp', 'deterministic',
                  'max_drop', 'mask_seed_offset', 'realjpeg_p', 'realjpeg_q', 'realjpeg_seed_offset',
                  'jpeg_subsampling', 'eval_jpeg')
H264_KEYS = ('h264_crf', 'h264_videos_per_class', 'h264_seed', 'h264_preset', 'h264_fps', 'h264_size')
# Used only when Cell 6's benchmark has not run; the preflight replaces them with measured speeds.
DEFAULT_BENCH = {'dual': dict(train_img_s=120.0, eval_img_s=350.0), 'rgb': dict(train_img_s=250.0, eval_img_s=500.0)}
H264_SEC_PER_VIDEO = 0.6   # projection only: read 8 crops, encode and decode one clip, one process
WARMUP_SKIP_LIMIT = 20     # 20 halvings take GradScaler from 2**16 to 2**-4: beyond that it is not warm-up


def arm_spec(name):
    if name in ARM_DEFS:
        return ARM_DEFS[name]
    if name in ORIGINAL_DEFS:
        return ORIGINAL_DEFS[name]
    raise KeyError(f'Unknown arm {name!r}; choose from {ARM_ORDER}')


def seed_all(seed, deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic


def free():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, Path):
        return str(x)
    return x


def write_json(path, obj):
    Path(path).write_text(json.dumps(jsonable(obj), indent=2))


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def conditions(cfg, h264=True):
    names = ['clean'] + [f'jpeg{q}' for q in cfg['eval_jpeg']]
    if h264:
        names += ['clean_h264subset'] + [f'h264_crf{c}' for c in cfg['h264_crf']]
    return names


def read_pred(path):
    return pd.read_csv(path, dtype={'dataset': str, 'video_id': str, 'group_id': str})


# ------------------------------------------------------------------ masks

def drop_table(kind, max_drop=0.35):
    """8x8 drop probabilities. All three kinds have the same mean over the 63 AC
    coefficients; DC is never dropped."""
    q = np.array(JPEG_LUMA, dtype=np.float64)
    ac = np.ones((8, 8), bool)
    ac[0, 0] = False
    table = max_drop * q / q.max()
    table[0, 0] = 0.0
    target = table[ac].mean()
    if kind == 'qtable':
        p = table
    elif kind == 'uniform':
        p = np.where(ac, target, 0.0)
    elif kind == 'inverted':
        # Same probabilities, reversed order: the least-quantized coefficient gets the
        # largest probability. Tied table entries share their mean, which keeps the sum.
        qa, pa = q[ac], table[ac]
        inv = np.empty_like(pa)
        inv[np.argsort(qa, kind='stable')] = np.sort(pa)[::-1]
        _, tie = np.unique(qa, return_inverse=True)
        inv = (np.bincount(tie, inv) / np.bincount(tie))[tie]
        inv = np.clip(inv * target / inv.mean(), 0.0, 1.0)
        p = np.zeros((8, 8))
        p[ac] = inv
    else:
        raise ValueError(f'Unknown mask kind {kind!r}; choose from {MASK_KINDS}')
    if abs(p[ac].mean() - target) > 1e-6 or p[0, 0] != 0 or p.min() < 0 or p.max() > 1:
        raise AssertionError(f'{kind}: mean AC drop {p[ac].mean():.8f} != {target:.8f}')
    return p


class BlockDCTMask(nn.Module):
    """One stochastic 8x8 coefficient mask per image, shared by its channels and blocks.

    No inverted-dropout rescaling: removing AC energy is the intervention. The mask
    draws from its own generator, so it never advances the global RNG.
    """
    def __init__(self, probability, seed=1042):
        super().__init__()
        p = torch.as_tensor(np.asarray(probability), dtype=torch.float32)
        if p.shape != (8, 8) or p[0, 0] != 0 or p.min() < 0 or p.max() > 1:
            raise ValueError('probability must be an 8x8 table in [0, 1] with DC = 0')
        n = torch.arange(8, dtype=torch.float32)
        basis = torch.cos(math.pi * (2 * n + 1) * n[:, None] / 16) * math.sqrt(2 / 8)
        basis[0] /= math.sqrt(2)
        self.register_buffer('basis', basis)
        self.register_buffer('probability', p)
        self.active = bool((p > 0).any())
        self.seed = seed
        self.generators = {}
        self.calls = 0

    def coefficients(self, x):
        b, c, h, w = x.shape
        if h % 8 or w % 8:
            raise ValueError('FAD input H/W must be divisible by 8')
        blocks = x.float().reshape(b, c, h // 8, 8, w // 8, 8).permute(0, 1, 2, 4, 3, 5)
        return self.basis @ blocks @ self.basis.T

    def reconstruct(self, coeff):
        b, c, nh, nw, _, _ = coeff.shape
        blocks = self.basis.T @ coeff @ self.basis
        return blocks.permute(0, 1, 2, 4, 3, 5).reshape(b, c, nh * 8, nw * 8)

    def keep_mask(self, n, device):
        key = str(device)
        if key not in self.generators:
            self.generators[key] = torch.Generator(device=device).manual_seed(self.seed)
        u = torch.rand((n, 1, 1, 1, 8, 8), device=device, generator=self.generators[key])
        return u >= self.probability.to(device)

    def forward(self, x):
        if not self.training or not self.active:
            return x
        if x.ndim != 4 or not x.is_floating_point():
            raise ValueError('Expected floating BCHW spatial input to fad')
        self.calls += 1
        with torch.autocast(device_type=x.device.type, enabled=False):   # DCT stays fp32
            coeff = self.coefficients(x)
            out = self.reconstruct(coeff * self.keep_mask(len(x), x.device).to(coeff.dtype))
        return out.to(x.dtype)


class MaskedFrequencyModel(nn.Module):
    """Masks the input of base.fad through a forward pre-hook; base.rgb is untouched."""
    def __init__(self, base, mask):
        super().__init__()
        if not hasattr(base, 'fad') or not hasattr(base, 'rgb'):
            raise ValueError('Use the dual-stream model with .fad and .rgb modules')
        self.base, self.mask = base, mask
        self.handle = base.fad.register_forward_pre_hook(self._mask_input)

    def _mask_input(self, module, args):
        if not (module.training and self.mask.training):   # off whenever base or wrapper is in eval
            return None
        if len(args) != 1:
            raise ValueError('Expected fad(x); adapt the hook for another signature')
        return (self.mask(args[0]),)

    def remove(self):
        self.handle.remove()

    def forward(self, x):
        return self.base(x)


# ------------------------------------------------------------------- data

def jpeg_roundtrip(image, quality, subsampling=2):
    buf = io.BytesIO()
    image.save(buf, format='JPEG', quality=int(quality), subsampling=subsampling)
    buf.seek(0)
    with Image.open(buf) as decoded:
        return decoded.convert('RGB')


def realjpeg_params(seed, epoch, index, p, q_range):
    """Deterministic per (seed, epoch, sample), so it is identical in any DataLoader worker."""
    rng = np.random.default_rng([int(seed), int(epoch), int(index)])
    apply = bool(rng.random() < p)
    return apply, int(rng.integers(q_range[0], q_range[1] + 1))


class H264Store:
    """Decoded H.264 frames (memory-mapped uint8 array) looked up by original frame path."""
    def __init__(self, array, paths):
        self.array = array
        self.row = {p: i for i, p in enumerate(paths)}

    def __getitem__(self, path):
        return np.ascontiguousarray(self.array[self.row[path]])


class FrameData(Dataset):
    def __init__(self, frame, preprocess, quality=None, realjpeg=None, seed=0, store=None, subsampling=2):
        self.paths = frame.path.tolist()
        self.labels = frame.label.astype(float).tolist()
        self.preprocess, self.quality, self.store = preprocess, quality, store
        self.realjpeg, self.seed, self.subsampling = realjpeg, seed, subsampling
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch   # workers are re-forked each epoch, so they see the new value

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        if self.store is not None:
            image = Image.fromarray(self.store[self.paths[i]])
        else:
            image = _open_rgb(self.paths[i])
        if self.quality is not None:
            image = jpeg_roundtrip(image, self.quality, self.subsampling)
        if self.realjpeg is not None:
            apply, q = realjpeg_params(self.seed, self.epoch, i, *self.realjpeg)
            if apply:
                image = jpeg_roundtrip(image, q, self.subsampling)
        return self.preprocess(image), self.labels[i], i


def _open_rgb(path):
    with Image.open(path) as im:
        return im.convert('RGB')


def loader(ds, cfg, train=False):
    return DataLoader(ds, batch_size=cfg['batch_size'] if train else cfg['eval_batch_size'],
                      shuffle=train, drop_last=train, num_workers=cfg['workers'],
                      timeout=600 if cfg['workers'] else 0,   # a stuck worker raises instead of hanging
                      pin_memory=cfg['device'].startswith('cuda'),
                      generator=torch.Generator().manual_seed(cfg['seed']))


def video_scores(frames, prob):
    d = frames[['dataset', 'video_id', 'group_id', 'label']].copy()
    d['probability'] = prob
    return d.groupby(['dataset', 'video_id', 'group_id', 'label'], as_index=False).probability.mean()


def macro_auc(v):
    return float(np.mean([roc_auc_score(g.label, g.probability) for _, g in v.groupby('dataset')]))


@torch.no_grad()
def predict(model, frames, preprocess, cfg, quality=None, store=None, log=None, what=''):
    """Video score = mean frame probability, fp16 autocast, no flip TTA (as Phase 13b Part A)."""
    model.eval()
    dev = cfg['device']
    prob = np.full(len(frames), np.nan)
    dl = loader(FrameData(frames, preprocess, quality=quality, store=store,
                          subsampling=cfg['jpeg_subsampling']), cfg)
    for k, (x, _, idx) in enumerate(dl):
        with torch.autocast(dev.split(':')[0], dtype=torch.float16, enabled=cfg['amp']):
            logit = model(x.to(dev, non_blocking=True)).float().reshape(-1)
        prob[idx.numpy()] = torch.sigmoid(logit).cpu().numpy()
        if log and k and k % cfg['log_every'] == 0:
            log(f'    {what}: batch {k + 1}/{len(dl)}')
    if not np.isfinite(prob).all():
        raise FloatingPointError(f'{what}: non-finite predictions')
    return video_scores(frames, prob)


# --------------------------------------------------------------- training

def train_step(model, x, y, criterion, opt, scaler, cfg):
    """One AMP step. Returns (loss, overflow). An fp16 overflow is not an error:
    GradScaler skips the step and lowers its scale, so the norm is only counted."""
    opt.zero_grad(set_to_none=True)
    with torch.autocast(cfg['device'].split(':')[0], enabled=cfg['amp']):
        loss = criterion(model(x).reshape(-1).float(), y)
    if not torch.isfinite(loss):
        raise FloatingPointError('non-finite training loss')
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    norm = nn.utils.clip_grad_norm_([p for p in model.parameters() if p.grad is not None], cfg['grad_clip'])
    scaler.step(opt)
    scaler.update()
    return float(loss.detach()), not bool(torch.isfinite(norm))


def check_overflow(skipped, steps, what, frac=0.05, floor=3):
    # The floor: on a short epoch (smoke, ~12 steps) one re-descent of the scale on a
    # batch with larger gradients is normal fp16 behaviour, not divergence.
    if skipped > max(frac * steps, floor):
        raise FloatingPointError(f'{what}: {skipped} of {steps} steps overflowed after GradScaler warm-up; '
                                 f'this is not warm-up. Try amp=False or a lower lr.')


# ------------------------------------------------------------------ H.264

def h264_roundtrip(frames, crf, exe, preset='medium', fps=25, threads=1, timeout=120):
    """uint8 (n, h, w, 3) RGB -> one libx264 clip -> decoded uint8 (n, h, w, 3). A hung ffmpeg
    raises TimeoutExpired instead of blocking the Pool (and the watchdog) forever."""
    frames = np.ascontiguousarray(frames, dtype=np.uint8)
    n, h, w, _ = frames.shape
    with tempfile.TemporaryDirectory(prefix='codec7_') as tmp:
        clip = os.path.join(tmp, 'clip.mp4')
        enc = [exe, '-hide_banner', '-loglevel', 'error', '-y',
               '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{w}x{h}', '-r', str(fps), '-i', 'pipe:0',
               '-c:v', 'libx264', '-preset', preset, '-crf', str(crf), '-pix_fmt', 'yuv420p',
               '-threads', str(threads), clip]
        r = subprocess.run(enc, input=frames.tobytes(), capture_output=True, timeout=timeout)
        if r.returncode:
            raise RuntimeError(f'ffmpeg encode failed: {r.stderr.decode(errors="replace")[-400:]}')
        dec = [exe, '-nostdin', '-hide_banner', '-loglevel', 'error', '-threads', str(threads), '-i', clip,
               '-f', 'rawvideo', '-pix_fmt', 'rgb24', 'pipe:1']
        r = subprocess.run(dec, capture_output=True, stdin=subprocess.DEVNULL, timeout=timeout)
        if r.returncode:
            raise RuntimeError(f'ffmpeg decode failed: {r.stderr.decode(errors="replace")[-400:]}')
    out = np.frombuffer(r.stdout, dtype=np.uint8)
    if out.size != n * h * w * 3:
        raise RuntimeError(f'H.264 round trip returned {out.size / (h * w * 3):g} frames for {n}')
    return out.reshape(n, h, w, 3).copy()


def _h264_video(job):
    """Pool worker (top level so it pickles): one video's frames in their given order."""
    paths, crf, exe, size, preset, fps = job
    frames = []
    for p in paths:
        with Image.open(p) as im:
            frames.append(np.asarray(im.convert('RGB').resize((size, size), Image.Resampling.BILINEAR),
                                     dtype=np.uint8))
    return h264_roundtrip(np.stack(frames), crf, exe, preset, fps)


def find_ffmpeg():
    """(exe, note). System ffmpeg first, then imageio-ffmpeg's binary; each must pass a
    tiny libx264 round trip. (None, reason) means the H.264 conditions are skipped."""
    candidates, notes = [], []
    if shutil.which('ffmpeg'):
        candidates.append(shutil.which('ffmpeg'))
    try:
        import imageio_ffmpeg
        candidates.append(imageio_ffmpeg.get_ffmpeg_exe())
    except Exception as e:
        notes.append(f'imageio_ffmpeg unavailable ({type(e).__name__})')
    probe = np.random.default_rng(0).integers(0, 256, (3, 32, 32, 3), dtype=np.uint8)
    for exe in candidates:
        try:
            h264_roundtrip(probe, 23, exe)
            version = subprocess.run([exe, '-version'], capture_output=True, stdin=subprocess.DEVNULL,
                                     timeout=30).stdout.decode(errors='replace').splitlines()[0]
            return exe, f'{exe} ({version})'
        except Exception as e:
            notes.append(f'{exe}: {e}')
    return None, 'no working ffmpeg with libx264: ' + ('; '.join(notes) or 'none found')


def h264_subset(test, n_per_class, seed):
    """Up to n real and n fake held-out videos, fixed by seed (all if fewer)."""
    v = test[['dataset', 'video_id', 'group_id', 'label']].drop_duplicates(['dataset', 'video_id'])
    parts = [g.sample(min(n_per_class, len(g)), random_state=seed) for _, g in v.groupby('label')]
    return pd.concat(parts).sort_values(['label', 'dataset', 'video_id']).reset_index(drop=True)


# ------------------------------------------------------------- statistics

def rank_auc(y, s):
    y = np.asarray(y).astype(bool)
    r = rankdata(s)
    n1 = int(y.sum())
    n0 = len(y) - n1
    if not n1 or not n0:
        return float('nan')
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def weighted_auc(y, s, w=None):
    """Mann-Whitney AUC (ties count 1/2) for each row of weights w; one sort serves every
    bootstrap resample, since a resample is the original videos with integer weights."""
    y = np.asarray(y).astype(bool)
    s = np.asarray(s, dtype=np.float64)
    w = np.ones((1, len(s))) if w is None else np.atleast_2d(np.asarray(w, dtype=np.float64))
    order = np.argsort(s, kind='mergesort')
    s, y, w = s[order], y[order], w[:, order]
    starts = np.flatnonzero(np.r_[True, s[1:] != s[:-1]])
    pos = np.add.reduceat(w * y, starts, axis=1)
    neg = np.add.reduceat(w * ~y, starts, axis=1)
    below = np.cumsum(neg, axis=1) - neg
    num = (pos * (below + 0.5 * neg)).sum(1)
    den = pos.sum(1) * neg.sum(1)
    out = np.full(len(w), np.nan)
    ok = den > 0
    out[ok] = num[ok] / den[ok]
    return out


def cluster_weights(clusters, n_boot, seed):
    """(n_boot, n) integer weights: clusters drawn with replacement, every video of a
    drawn cluster counted once per draw."""
    codes, uniq = pd.factorize(pd.Series(list(clusters)))
    k = len(uniq)
    if k < 2:
        raise ValueError('Need at least two independent clusters')
    rng = np.random.default_rng(seed)
    counts = np.stack([np.bincount(rng.integers(0, k, k), minlength=k) for _ in range(n_boot)])
    return counts[:, codes]


def paired_bootstrap(a, b, n_boot=1000, seed=42):
    """AUC(a) - AUC(b) on identical videos; clusters = (dataset, group_id)."""
    keys = ['dataset', 'video_id', 'group_id', 'label']
    d = a.merge(b, on=keys, validate='one_to_one', suffixes=('_a', '_b'))
    if not len(d) == len(a) == len(b):
        raise ValueError('The two arms did not score exactly the same videos')
    y = d.label.to_numpy() == 1
    clusters = d.dataset.astype(str) + '||' + d.group_id.astype(str)
    w = cluster_weights(clusters, n_boot, seed)
    auc_a, auc_b = weighted_auc(y, d.probability_a)[0], weighted_auc(y, d.probability_b)[0]
    boot = weighted_auc(y, d.probability_a, w) - weighted_auc(y, d.probability_b, w)
    boot = boot[np.isfinite(boot)]
    if len(boot) < 0.9 * n_boot:
        raise ValueError('Too few valid bootstrap samples; inspect class/group balance')
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return dict(auc_a=float(auc_a), auc_b=float(auc_b), delta=float(auc_a - auc_b), ci_low=float(lo),
                ci_high=float(hi), valid_bootstraps=int(len(boot)), videos=len(d),
                clusters=int(clusters.nunique()))


# ------------------------------------------------------- earlier outputs

def find_reuse_roots(skip=(), top='/kaggle/input', max_depth=11):
    """Folders holding <held>/<arm>/done.json from an earlier run, without listing face crops.
    A folder with images is pruned unless it has corpus subfolders: an earlier output keeps
    its plots next to <held>/."""
    skip, hits, top = {str(p) for p in skip}, set(), str(top)
    for dp, dns, fns in os.walk(top):
        p = Path(dp)
        if 'done.json' in fns and p.parent.name in CORPORA:
            hits.add(p.parent.parent)
        images = any(f.lower().endswith(IMG_EXT) for f in fns) and not set(dns) & set(CORPORA)
        if dp in skip or dp[len(top):].count(os.sep) >= max_depth or images:
            dns[:] = []
    return sorted(hits)


# ------------------------------------------------------------- experiment

class Experiment:
    """checkpoints: {'dual': {held: path}, 'rgb': {held: path}}; splits: {held: (train, val, test)}."""

    def __init__(self, cfg, splits, checkpoints, make_dual, make_rgb, preprocess, log, clock, out,
                 reuse_roots=()):
        self.cfg, self.splits, self.ckpt = cfg, splits, checkpoints
        self.builders = dict(dual=make_dual, rgb=make_rgb)
        self.preprocess, self.log, self.clock = preprocess, log, clock
        self.out = Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.reuse_roots = [Path(r) for r in reuse_roots]
        self.dev = cfg['device']
        self.cache = Path(cfg['h264_cache'])
        self.ffmpeg, self.ffmpeg_note = find_ffmpeg()
        self.conds = conditions(cfg, h264=self.ffmpeg is not None)
        self.log(f'H.264 conditions: {"ON, " if self.ffmpeg else "SKIPPED, "}{self.ffmpeg_note}')
        self.subsets, self.subset_frames = {}, {}
        for held, (_, _, test) in splits.items():
            vids = h264_subset(test, cfg['h264_videos_per_class'], cfg['h264_seed'])
            keep = set(zip(vids.dataset, vids.video_id))
            self.subsets[held] = vids
            self.subset_frames[held] = test[[k in keep for k in zip(test.dataset, test.video_id)]].reset_index(drop=True)
            (self.out / held).mkdir(exist_ok=True)
            vids.to_csv(self.out / held / 'h264_subset.csv', index=False)
        self.sha, self.stores, self.ratios, self.checks = {}, {}, {}, {}
        self.bench, self.h264_seconds, self.h264_failed = None, 0.0, {}
        if (self.out / 'preflight.json').is_file():
            self.bench = json.loads((self.out / 'preflight.json').read_text()).get('bench')

    # ---- models
    def load(self, kind, path):
        model = self.builders[kind]()
        state = torch.load(path, map_location='cpu', weights_only=True)
        if not isinstance(state, dict):
            raise TypeError(f'{path}: expected a plain state_dict')
        if state and all(k.startswith('net.') for k in state):   # Phase 11 SingleNet wrapper
            state = {k[4:]: v for k, v in state.items()}
        if kind == 'dual' and 'rgb.conv_stem.weight' not in state:
            raise ValueError(f'{path}: not a dual-stream state_dict')
        model.load_state_dict(state, strict=True)
        return model.to(self.dev)

    def mask_for(self, kind):
        return BlockDCTMask(drop_table(kind, self.cfg['max_drop']), self.cfg['seed'] + self.cfg['mask_seed_offset'])

    def checkpoint_sha(self, path):
        key = str(path)
        if key not in self.sha:
            self.sha[key] = file_sha256(path)
        return self.sha[key]

    def signature(self, held, name):
        spec = arm_spec(name)
        cfg = self.cfg
        return jsonable(dict(arm=name, definition=spec, held=held,
                             checkpoint_sha256=self.checkpoint_sha(self.ckpt[spec['model']][held]),
                             h264={k: cfg[k] for k in H264_KEYS}, **{k: cfg[k] for k in SIGNATURE_KEYS}))

    # ---- time
    def factor(self, kind, part):
        r = self.ratios.get((kind, part))
        return float(np.median(r)) if r else 1.0

    def observe(self, kind, part, actual, predicted):
        if predicted > 0 and actual > 0:
            self.ratios.setdefault((kind, part), []).append(actual / predicted)

    def projection(self, held, name, built=None, raw=False):
        """Seconds for training, evaluation and any H.264 encoding still to do; raw=True
        leaves out the correction learned from finished arms."""
        cfg, spec = self.cfg, arm_spec(name)
        kind = spec['model']
        b = (self.bench or DEFAULT_BENCH)[kind]
        tr, va, te = self.splits[held]
        sub = len(self.subset_frames[held]) if self.ffmpeg else 0
        n_eval = len(te) * (1 + len(cfg['eval_jpeg'])) + sub * len(cfg['h264_crf'])
        train = cfg['epochs'] * (len(tr) / b['train_img_s'] + len(va) / b['eval_img_s']) if spec['trained'] else 0.0
        built = set(self.stores) | set(self.h264_failed) if built is None else built
        todo = [c for c in cfg['h264_crf'] if (held, c) not in built] if self.ffmpeg else []
        h264 = len(todo) * len(self.subsets[held]) * H264_SEC_PER_VIDEO / cfg['h264_workers']
        ft, fe = (1.0, 1.0) if raw else (self.factor(kind, 'train'), self.factor(kind, 'eval'))
        return dict(train=train * ft, eval=n_eval / b['eval_img_s'] * fe, h264=h264)

    @staticmethod
    def hours_of(p):
        return 1.15 * sum(p.values()) / 3600

    # ---- reuse
    def find_done(self, held, name, sig):
        for root in [self.out] + self.reuse_roots:
            meta = root / held / name / 'done.json'
            if meta.is_file():
                try:
                    r = json.loads(meta.read_text())
                except ValueError:
                    continue
                if r.get('status') == 'complete' and r.get('signature') == sig:
                    return root, r
        return None, None

    def reuse(self, held, name, sig, d):
        root, r = self.find_done(held, name, sig)
        if r is None:
            return None
        if root != self.out:
            src = root / held / name
            for f in src.iterdir():
                if f.is_file():
                    shutil.copy2(f, d / f.name)
            r['reused_from'] = str(src)
            write_json(d / 'done.json', r)
            self.log(f'Reusing {name} | {held} held out from {src}')
        else:
            self.log(f'{name} | {held} held out: already complete in {d}')
        return r

    # ---- H.264
    def h264_store(self, held, crf):
        key = (held, crf)
        if key not in self.stores:
            frames = self.subset_frames[held]
            paths = frames.path.tolist()
            self.cache.mkdir(parents=True, exist_ok=True)
            stem = f'{self.out.name}_{held}_crf{crf}'
            arr_path, idx_path = self.cache / f'{stem}.npy', self.cache / f'{stem}.csv'
            if not (arr_path.is_file() and idx_path.is_file() and pd.read_csv(idx_path).path.tolist() == paths):
                self.build_h264(held, crf, frames, arr_path, idx_path)
            self.stores[key] = H264Store(np.load(arr_path, mmap_mode='r'), paths)
        return self.stores[key]

    def build_h264(self, held, crf, frames, arr_path, idx_path):
        cfg, t = self.cfg, time.time()
        s = cfg['h264_size']
        groups = list(frames.groupby(['dataset', 'video_id'], sort=False).indices.values())
        jobs = [(frames.path.iloc[ix].tolist(), crf, self.ffmpeg, s, cfg['h264_preset'], cfg['h264_fps'])
                for ix in groups]
        self.log(f'  H.264 CRF {crf} | {held}: encoding {len(jobs)} videos ({len(frames)} frames)')
        part = arr_path.with_name(arr_path.stem + '.partial.npy')
        arr = np.lib.format.open_memmap(part, mode='w+', dtype=np.uint8, shape=(len(frames), s, s, 3))
        try:
            # fork: the workers only read crops and run ffmpeg; they never touch CUDA.
            with multiprocessing.get_context('fork').Pool(cfg['h264_workers']) as pool:
                for k, (ix, decoded) in enumerate(zip(groups, pool.imap(_h264_video, jobs, chunksize=2))):
                    arr[ix] = decoded
                    if k % 200 == 0:
                        self.log(f'    H.264 CRF {crf} | {held}: video {k + 1}/{len(jobs)}')
            arr.flush()
        except BaseException:
            del arr
            part.unlink(missing_ok=True)
            self.h264_seconds += time.time() - t
            raise
        del arr
        os.replace(part, arr_path)
        pd.DataFrame(dict(path=frames.path)).to_csv(idx_path, index=False)
        sec = time.time() - t
        self.h264_seconds += sec
        write_json(self.out / held / f'h264_crf{crf}_build.json',
                   dict(held=held, crf=crf, videos=len(jobs), frames=len(frames), seconds=sec,
                        ffmpeg=self.ffmpeg_note, preset=cfg['h264_preset'], fps=cfg['h264_fps'], size=s,
                        pix_fmt='yuv420p', threads_per_clip=1))
        self.log(f'  H.264 CRF {crf} | {held}: done in {sec / 60:.1f} min')

    def clear_h264_cache(self):
        self.stores.clear()
        for f in self.cache.glob(f'{self.out.name}_*'):
            f.unlink()

    # ---- evaluation
    def missing(self, d):
        return [c for c in self.conds if not (d / f'pred_{c}.csv').is_file()]

    def scored(self, d):
        return [c for c in self.conds if (d / f'pred_{c}.csv').is_file()]

    def evaluate(self, model, held, name, d):
        cfg, test = self.cfg, self.splits[held][2]
        for cond in self.missing(d):
            t = time.time()
            what = f'{name} | {held} | {cond}'
            if cond == 'clean':
                pred = predict(model, test, self.preprocess, cfg, log=self.log, what=what)
            elif cond.startswith('jpeg'):
                pred = predict(model, test, self.preprocess, cfg, quality=int(cond[4:]), log=self.log, what=what)
            elif cond == 'clean_h264subset':
                clean = read_pred(d / 'pred_clean.csv')
                keep = set(zip(self.subsets[held].dataset, self.subsets[held].video_id))
                pred = clean[[k in keep for k in zip(clean.dataset, clean.video_id)]].reset_index(drop=True)
            elif cond.startswith('h264_crf'):
                crf = int(cond[8:])
                if (held, crf) in self.h264_failed:
                    continue
                try:
                    store = self.h264_store(held, crf)
                except Exception as e:
                    # Training and the other conditions stand; a later session adds this one.
                    if cfg['stop_on_error']:
                        raise
                    self.h264_failed[(held, crf)] = repr(e)
                    self.log(f'  H.264 CRF {crf} | {held}: FAILED, {cond} left out for this fold: {e!r}\n'
                             f'{traceback.format_exc()}')
                    continue
                pred = predict(model, self.subset_frames[held], self.preprocess, cfg, store=store,
                               log=self.log, what=what)
            else:
                raise ValueError(cond)
            pred.to_csv(d / f'pred_{cond}.csv', index=False)
            self.log(f'  {what}: AUC {macro_auc(pred):.4f} on {len(pred)} videos ({time.time() - t:.0f} s)')

    def consistency(self, held, name, d):
        auc = macro_auc(read_pred(d / 'pred_clean.csv'))
        ref = REFERENCE_AUC[name][held]
        if self.cfg['smoke']:
            verdict = 'not comparable (smoke frames)'
        else:
            verdict = 'PASS' if abs(auc - ref) <= 0.005 else f'WARN gap {auc - ref:+.4f}'
        self.checks[(held, name)] = dict(fold=held, arm=name, clean_auc=auc, reference_auc=ref,
                                         gap=auc - ref, verdict=verdict)
        pd.DataFrame(self.checks.values()).to_csv(self.out / 'consistency.csv', index=False)
        self.log(f'  consistency {name} | {held}: clean AUC {auc:.4f} vs Phase 13b {ref:.4f}: {verdict}')

    # ---- training
    def fit(self, held, name, d, proj):
        cfg, spec, dev = self.cfg, ARM_DEFS[name], self.dev
        train, val, _ = self.splits[held]
        seed_all(cfg['seed'], cfg['deterministic'])
        base = self.load(spec['model'], self.ckpt[spec['model']][held])
        for p in base.parameters():
            p.requires_grad_(True)
        model = MaskedFrequencyModel(base, self.mask_for(spec['mask'])).to(dev) if spec['mask'] else base
        opt = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
        scaler = torch.amp.GradScaler('cuda', enabled=cfg['amp'])
        n_fake = int((train.label == 1).sum())
        pos_weight = torch.tensor([(len(train) - n_fake) / max(1, n_fake)], device=dev)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        ds = FrameData(train, self.preprocess, seed=cfg['seed'] + cfg['realjpeg_seed_offset'],
                       realjpeg=(cfg['realjpeg_p'], tuple(cfg['realjpeg_q'])) if spec['realjpeg'] else None,
                       subsampling=cfg['jpeg_subsampling'])
        dl = loader(ds, cfg, train=True)
        rows, best, best_epoch, status = [], -float('inf'), None, 'complete'
        settled, warmup, t_all = False, 0, 0.0
        eval_h = 1.15 * (proj['eval'] + proj['h264']) / 3600
        for epoch in range(1, cfg['epochs'] + 1):
            ds.set_epoch(epoch)
            model.train()
            t, losses, skipped = time.time(), [], 0
            for step, (x, y, _) in enumerate(dl):
                loss, overflow = train_step(model, x.to(dev, non_blocking=True),
                                            y.float().to(dev, non_blocking=True), criterion, opt, scaler, cfg)
                losses.append(loss)
                if overflow and not settled:
                    warmup += 1   # GradScaler's initial descent from 2**16, skipped by design
                elif overflow:
                    skipped += 1
                else:
                    settled = True
                if step % cfg['log_every'] == 0:
                    self.log(f'  {name} | {held} | epoch {epoch} step {step + 1}/{len(dl)} '
                             f'loss {np.mean(losses[-50:]):.4f} scale {scaler.get_scale():g}')
            if warmup > WARMUP_SKIP_LIMIT:
                raise FloatingPointError(f'{name}/{held}: {warmup} overflowing steps before the first finite one')
            check_overflow(skipped, len(dl), f'{name}/{held} epoch {epoch}', cfg['max_overflow_frac'])
            train_sec = time.time() - t
            t = time.time()
            vp = predict(model, val, self.preprocess, cfg, log=self.log, what=f'{name} | {held} | source val')
            auc, val_sec = macro_auc(vp), time.time() - t
            t_all += train_sec + val_sec
            rows.append(dict(fold=held, arm=name, epoch=epoch, train_loss=float(np.mean(losses)),
                             source_val_auc=auc, steps=len(dl), overflow_steps=skipped,
                             warmup_skipped_steps=warmup, grad_scale=scaler.get_scale(),
                             train_seconds=train_sec, val_seconds=val_sec))
            pd.DataFrame(rows).to_csv(d / 'training.csv', index=False)
            self.log(f'  {name} | {held} | epoch {epoch}: loss {np.mean(losses):.4f} | source-val AUC {auc:.4f} '
                     f'| overflow {skipped} (+{warmup} warm-up) | {train_sec / 60:.1f}+{val_sec / 60:.1f} min')
            if auc > best:
                best, best_epoch = auc, epoch
                torch.save(base.state_dict(), d / 'best.pth')   # base only: the hook is not saved
                vp.to_csv(d / 'pred_val_best.csv', index=False)
            per_epoch_h = (train_sec + val_sec) / 3600
            if epoch < cfg['epochs'] and self.clock() + per_epoch_h + eval_h > cfg['time_limit_h']:
                status = 'truncated'
                self.log(f'  {name} | {held}: time limit after epoch {epoch}; scoring the best epoch so far')
                break
        if spec['mask'] and model.mask.calls == 0:
            raise RuntimeError('Mask hook never executed')
        del model, base, opt, scaler
        free()
        return dict(status=status, best_epoch=best_epoch, source_val_auc=best, epochs_run=len(rows),
                    train_seconds=t_all)

    # ---- one arm on one fold
    def run_unit(self, held, name):
        cfg, spec = self.cfg, arm_spec(name)
        kind = spec['model']
        d = self.out / held / name
        d.mkdir(parents=True, exist_ok=True)
        sig = self.signature(held, name)
        t0 = time.time()
        done = self.reuse(held, name, sig, d)
        if done is not None:
            todo = self.missing(d)
            src = d / 'best.pth' if spec['trained'] else self.ckpt[kind][held]
            if todo and Path(src).is_file():
                self.log(f'  {name} | {held}: adding conditions {todo}')
                model = self.load(kind, src)
                self.evaluate(model, held, name, d)
                del model
                done['conditions'] = self.scored(d)
                write_json(d / 'done.json', done)
            elif todo:
                self.log(f'  {name} | {held}: {todo} missing and no weights to score them')
        else:
            p, p_raw = self.projection(held, name), self.projection(held, name, raw=True)
            need = self.hours_of(p)
            if self.clock() + need > cfg['time_limit_h']:
                self.log(f'SKIP {name} | {held}: needs ~{need:.2f} h, {cfg["time_limit_h"] - self.clock():.2f} h '
                         f'left. Run it in a new session with this output attached (reuse).')
                done = dict(status='skipped_time_budget', fold=held, arm=name, projected_hours=need,
                            at_hours=self.clock(), signature=sig)
                write_json(d / 'done.json', done)
                return done
            self.log(f'=== {name} | {held} held out | ~{need:.2f} h projected ===')
            for f in d.iterdir():   # leftovers of an unfinished attempt must not count as evaluated
                if f.is_file() and (f.name.startswith('pred_') or f.name in ('training.csv', 'best.pth')):
                    f.unlink()
            info = dict(status='complete', best_epoch=None, source_val_auc=None, train_seconds=0.0)
            if spec['trained']:
                info = self.fit(held, name, d, p)
                if info['status'] == 'complete':
                    self.observe(kind, 'train', info['train_seconds'], p_raw['train'])
                model = self.load(kind, d / 'best.pth')
            else:
                model = self.load(kind, self.ckpt[kind][held])
            t, h0 = time.time(), self.h264_seconds
            self.evaluate(model, held, name, d)
            eval_sec = time.time() - t - (self.h264_seconds - h0)
            self.observe(kind, 'eval', eval_sec, p_raw['eval'])
            del model
            done = dict(info, fold=held, arm=name, seconds=time.time() - t0, eval_seconds=eval_sec,
                        projected_hours=need, finished_at_hours=self.clock(), conditions=self.scored(d),
                        missing_conditions=self.missing(d), signature=sig)
            write_json(d / 'done.json', done)
            self.log(f'  {name} | {held}: {done["status"]} in {done["seconds"] / 60:.1f} min')
        if not spec['trained'] and (d / 'pred_clean.csv').is_file():
            self.consistency(held, name, d)
        free()
        return done

    def run(self, arms, folds, originals=True):
        """Arm-major order: if time runs out, the late (lower-priority) arms are the ones
        missing, never a whole fold of the primary arms."""
        names = (list(ORIGINAL_DEFS) if originals else []) + list(arms)
        for n in names:
            arm_spec(n)
        if self.bench is None:
            self.log('No preflight benchmark: projections use DEFAULT_BENCH until arms finish')
        records = []
        for name in names:
            for held in folds:
                try:
                    r = self.run_unit(held, name)
                except Exception as e:
                    if self.cfg['stop_on_error']:
                        raise
                    self.log(f'FAILED {name} | {held}: {e!r}\n{traceback.format_exc()}')
                    r = dict(status='failed', fold=held, arm=name, error=repr(e))
                    write_json(self.out / held / name / 'done.json', r)
                    try:
                        free()
                    except Exception:
                        pass
                records.append({k: r.get(k) for k in ('fold', 'arm', 'status', 'best_epoch', 'source_val_auc',
                                                     'seconds', 'projected_hours', 'reused_from', 'error')})
                pd.DataFrame(records).to_csv(self.out / 'status.csv', index=False)
        try:
            self.clear_h264_cache()
        except Exception as e:
            self.log(f'clear_h264_cache failed: {e!r}')
        try:
            self._write_metadata(arms, folds, originals, records)
        except Exception as e:
            self.log(f'run_metadata.json not written: {e!r}')
        return pd.DataFrame(records)

    def _write_metadata(self, arms, folds, originals, records):
        write_json(self.out / 'run_metadata.json', dict(
            smoke=self.cfg['smoke'], arms=list(arms), folds=list(folds), originals=originals,
            config=self.cfg, torch=torch.__version__, numpy=np.__version__,
            gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
            ffmpeg=self.ffmpeg_note, reuse_roots=self.reuse_roots, bench=self.bench,
            checkpoints={k: {h: dict(path=str(p), sha256=self.checkpoint_sha(p)) for h, p in v.items() if h in folds}
                         for k, v in self.ckpt.items()},
            time_ratios={f'{k}/{p}': v for (k, p), v in self.ratios.items()},
            consistency=list(self.checks.values()), wall_hours=self.clock(), status=records,
            h264_failed={f'{h}/crf{c}': e for (h, c), e in self.h264_failed.items()}))

    # ---- preflight on the real checkpoints
    def preflight(self, arms, folds, originals=True, held=None):
        cfg, dev = self.cfg, self.dev
        held = held or folds[0]
        train, val, _ = self.splits[held]
        seed_all(cfg['seed'], cfg['deterministic'])
        dual = self.load('dual', self.ckpt['dual'][held]).eval()
        rgb = self.load('rgb', self.ckpt['rgb'][held]).eval()
        self.log(f'PASS: strict load of {Path(self.ckpt["dual"][held]).name} and {Path(self.ckpt["rgb"][held]).name}')
        head = train.groupby('label', group_keys=False).head(2)
        x = torch.stack([self.preprocess(_open_rgb(p)) for p in head.path]).to(dev)
        y = torch.tensor(head.label.to_numpy(), dtype=torch.float32, device=dev)
        with torch.no_grad():
            ref = dual(x).float()
            for kind in MASK_KINDS:
                w = MaskedFrequencyModel(dual, self.mask_for(kind)).to(dev).eval()
                gap = (w(x).float() - ref).abs().max().item()
                w.remove()
                assert gap <= 1e-6, f'{kind}: eval logits changed by {gap} with the mask disabled'
            assert torch.isfinite(rgb(x)).all()
        self.log('PASS: masks in eval mode leave the dual-stream logits identical')
        del dual, rgb
        free()

        m = self.load('dual', self.ckpt['dual'][held]).train()   # BN uses batch stats: same input, same output
        seen = {}
        hook = m.freq.register_forward_hook(lambda mod, i, o: seen.__setitem__('f', o.detach().clone()))
        with torch.no_grad():
            m(x)
            rgb_ref, freq_ref = getattr(m, '_pooled', None), seen['f']
            rgb_ref = None if rgb_ref is None else rgb_ref.clone()
            w = MaskedFrequencyModel(m, self.mask_for('qtable')).to(dev).train()
            w(x)
        hook.remove()
        assert w.mask.calls == 1, 'FAD hook did not run'
        assert not torch.allclose(seen['f'], freq_ref), 'mask did not change the frequency stream'
        if rgb_ref is not None:
            assert torch.allclose(m._pooled, rgb_ref, atol=1e-5), 'mask changed the RGB stream'
        self.log('PASS: training mask changes the frequency stream and leaves the RGB stream identical')
        del m, w
        free()

        for kind, mask in [('dual', None)] + [('dual', k) for k in MASK_KINDS] + [('rgb', None)]:
            base = self.load(kind, self.ckpt[kind][held])
            for p in base.parameters():
                p.requires_grad_(True)
            model = MaskedFrequencyModel(base, self.mask_for(mask)).to(dev) if mask else base
            model.train()
            with torch.autocast('cuda', enabled=cfg['amp']):
                loss = nn.functional.binary_cross_entropy_with_logits(model(x).reshape(-1).float(), y)
            loss.backward()
            grads = [p.grad for p in model.parameters() if p.grad is not None]
            assert torch.isfinite(loss) and grads and all(torch.isfinite(g).all() for g in grads), \
                f'{kind}/{mask}: non-finite loss or gradients'
            if mask:
                assert model.mask.calls > 0, 'FAD hook did not run'
            opt = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
            _, overflow = train_step(model, x, y, nn.BCEWithLogitsLoss(), opt,
                                     torch.amp.GradScaler('cuda', enabled=cfg['amp']), cfg)
            self.log(f'PASS: {kind} mask={mask}: AMP loss {loss.item():.4f}, finite gradients; '
                     f'GradScaler step {"skipped (fp16 overflow at scale 2**16, by design)" if overflow else "taken"}')
            del model, base, opt, grads
            free()

        bench = {kind: self.benchmark(kind, mask, held) for kind, mask in (('dual', 'qtable'), ('rgb', None))}
        self.bench = bench
        for kind, b in bench.items():
            self.log(f'  speed {kind}: train {b["train_img_s"]:.0f} img/s (real-JPEG data on), '
                     f'eval {b["eval_img_s"]:.0f} img/s (JPEG q50, after the first batch)')

        names = (list(ORIGINAL_DEFS) if originals else []) + list(arms)
        rows, now, built = [], self.clock(), set()
        for name in names:
            for h in folds:
                root, _ = self.find_done(h, name, self.signature(h, name))
                if root is not None:
                    rows.append(dict(arm=name, fold=h, train_h=0.0, eval_h=0.0, hours=0.0, action=f'reuse {root}'))
                    continue
                p = self.projection(h, name, built)
                hrs = self.hours_of(p)
                fits = now + hrs <= cfg['time_limit_h']
                if fits:
                    now += hrs
                    built |= {(h, c) for c in cfg['h264_crf']}
                rows.append(dict(arm=name, fold=h, train_h=1.15 * p['train'] / 3600,
                                 eval_h=1.15 * (p['eval'] + p['h264']) / 3600, hours=hrs,
                                 action='run' if fits else 'SKIP (time budget)'))
        plan = pd.DataFrame(rows)
        print(plan.round(2).to_string(index=False))
        total = self.clock() + plan.hours.sum()
        print(f'Projected finish: {total:.2f} h of this session (limit {cfg["time_limit_h"]} h; '
              f'{self.clock():.2f} h used so far).')
        advice = ''
        if total > cfg['time_limit_h']:
            drop, left = [], total
            for name in reversed([n for n in arms]):
                if left <= cfg['time_limit_h']:
                    break
                drop.append(name)
                left -= plan[plan.arm == name].hours.sum()
            advice = (f'Over the limit. Drop these arms from ARMS (lowest priority first): {drop} '
                      f'(projected {left:.2f} h), or run them in a second session with this output attached '
                      f'(finished arms are reused). Without changes, arms marked SKIP are skipped.')
            print(advice)
        else:
            print('Fits within the limit with every selected fold and arm.')
        result = dict(bench=bench, held=held, plan=plan.to_dict('records'), projected_finish_h=total,
                      advice=advice, ffmpeg=self.ffmpeg_note)
        write_json(self.out / 'preflight.json', result)
        return result

    def benchmark(self, kind, mask, held):
        """Train speed includes data loading with real-JPEG on (worst case)."""
        cfg, dev = self.cfg, self.dev
        train, val, _ = self.splits[held]
        seed_all(cfg['seed'], cfg['deterministic'])
        base = self.load(kind, self.ckpt[kind][held])
        model = MaskedFrequencyModel(base, self.mask_for(mask)).to(dev) if mask else base
        opt = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
        scaler = torch.amp.GradScaler('cuda', enabled=cfg['amp'])
        ds = FrameData(train, self.preprocess, realjpeg=(cfg['realjpeg_p'], tuple(cfg['realjpeg_q'])),
                       seed=cfg['seed'], subsampling=cfg['jpeg_subsampling'])
        ds.set_epoch(1)
        dl = loader(ds, cfg, train=True)
        n = min(len(dl), cfg['bench_batches'])
        if n < 3:
            raise ValueError('Need at least three training batches to benchmark')
        warm, t, imgs = 2, None, 0
        model.train()
        for i, (x, y, _) in enumerate(dl):
            train_step(model, x.to(dev, non_blocking=True), y.float().to(dev, non_blocking=True),
                       nn.BCEWithLogitsLoss(), opt, scaler, cfg)
            if i == warm - 1:
                torch.cuda.synchronize()
                t = time.time()
            elif i >= warm:
                imgs += len(x)
            if i + 1 >= n:
                break
        torch.cuda.synchronize()
        train_ips = imgs / (time.time() - t)
        del dl
        # Eval with JPEG q50 in the workers (conservative for clean and H.264); timed after the
        # first batch so worker start-up does not halve the figure, as it did in Phase 13.
        frames = val.head(min(len(val), cfg['eval_batch_size'] * 40))
        dl = loader(FrameData(frames, self.preprocess, quality=50, subsampling=cfg['jpeg_subsampling']), cfg)
        model.eval()
        t0, t, imgs = time.time(), None, 0
        with torch.no_grad():
            for k, (x, _, _) in enumerate(dl):
                with torch.autocast(dev.split(':')[0], dtype=torch.float16, enabled=cfg['amp']):
                    model(x.to(dev, non_blocking=True)).float().cpu()
                if k == 0:
                    t = time.time()
                else:
                    imgs += len(x)
        del dl
        eval_ips = imgs / (time.time() - t) if imgs else len(frames) / (time.time() - t0)
        del model, base, opt
        free()
        return dict(train_img_s=float(train_ips), eval_img_s=float(eval_ips))


# ---------------------------------------------------------------- summary

def collect(out):
    out = Path(out)
    rows, preds, hist = [], {}, []
    for held in [h for h in CORPORA if (out / h).is_dir()]:
        for d in sorted(p for p in (out / held).iterdir() if p.is_dir()):
            meta = d / 'done.json'
            status = json.loads(meta.read_text()).get('status') if meta.is_file() else 'incomplete'
            if status not in ('complete', 'truncated'):
                continue
            for f in sorted(d.glob('pred_*.csv')):
                cond = f.stem[len('pred_'):]
                if cond == 'val_best':
                    continue
                p = read_pred(f)
                preds[(held, d.name, cond)] = p
                rows.append(dict(fold=held, arm=d.name, condition=cond, auc=macro_auc(p), videos=len(p),
                                 real=int((p.label == 0).sum()), fake=int((p.label == 1).sum()), status=status,
                                 file=str(f.relative_to(out))))
            if (d / 'training.csv').is_file():
                hist.append(pd.read_csv(d / 'training.csv'))
    return pd.DataFrame(rows), preds, (pd.concat(hist, ignore_index=True) if hist else pd.DataFrame())


def _order(frame, cond_order):
    f = frame.copy()
    f['_f'] = f.fold.map({h: i for i, h in enumerate(CORPORA)}) if 'fold' in f else 0
    f['_a'] = f.arm.map({a: i for i, a in enumerate(ARM_ORDER)}).fillna(99) if 'arm' in f else 0
    f['_c'] = f.condition.map({c: i for i, c in enumerate(cond_order)}).fillna(99)
    f['_k'] = (f.comparison.map({f'{a} - {b}': i for i, (a, b) in enumerate(COMPARISONS)}).fillna(99)
               if 'comparison' in f else 0)
    keys = ['_f', '_a', '_c', '_k'] if 'comparison' not in f else ['_k', '_f', '_c']
    return f.sort_values(keys).drop(columns=['_f', '_a', '_c', '_k']).reset_index(drop=True)


def summarise(out, cfg, log=print):
    """Every number comes from the pred_*.csv and training.csv files under `out`."""
    out = Path(out)
    cond_order = conditions(cfg, h264=True)
    results, preds, hist = collect(out)
    if results.empty:
        log('No finished arms under ' + str(out))
        return dict(results=results)
    results = _order(results, cond_order)
    results.to_csv(out / 'results.csv', index=False)
    wide = results.pivot_table(index=['fold', 'arm'], columns='condition', values='auc')
    wide = wide[[c for c in cond_order if c in wide.columns]]
    wide = wide.reindex(sorted(wide.index, key=lambda k: (CORPORA.index(k[0]), ARM_ORDER.index(k[1])
                                                          if k[1] in ARM_ORDER else 99)))
    wide.to_csv(out / 'results_wide.csv')

    mean = (results.groupby(['arm', 'condition'])
            .agg(mean_auc=('auc', 'mean'), folds=('fold', 'size'), fold_list=('fold', lambda s: ','.join(s)),
                 truncated_folds=('status', lambda s: int((s == 'truncated').sum())))
            .reset_index())
    mean = _order(mean, cond_order)
    mean.to_csv(out / 'mean_over_folds.csv', index=False)

    # 'truncated' = the time guard stopped training early; such rows are flagged, never silently mixed in.
    status = {(r.fold, r.arm): r.status for r in results.itertuples()}
    rows = []
    for held in [h for h in CORPORA if h in set(results.fold)]:
        log(f'Paired bootstrap | {held} held out')
        for cond in cond_order:
            for a, b in COMPARISONS:
                pa, pb = preds.get((held, a, cond)), preds.get((held, b, cond))
                if pa is None or pb is None:
                    continue
                sa, sb = status[(held, a)], status[(held, b)]
                row = dict(fold=held, condition=cond, comparison=f'{a} - {b}', arm_a=a, arm_b=b,
                           endpoint=ENDPOINTS.get((a, b, cond), ''), status_a=sa, status_b=sb,
                           truncated_arm='truncated' in (sa, sb))
                if row['endpoint'] and row['truncated_arm']:
                    log(f'WARNING: endpoint {a} - {b} @ {cond} | {held}: one arm is truncated ({sa} / {sb})')
                try:
                    row.update(paired_bootstrap(pa, pb, cfg['bootstrap'], cfg['seed']))
                except ValueError as e:
                    row.update(error=str(e))
                rows.append(row)
    deltas = pd.DataFrame(rows)
    if len(deltas):
        deltas = _order(deltas, cond_order)
    if 'ci_low' in deltas:
        deltas['ci_excludes_0'] = (deltas.ci_low > 0) | (deltas.ci_high < 0)
    deltas.to_csv(out / 'paired_deltas.csv', index=False)
    endpoints = deltas[deltas.endpoint != ''] if len(deltas) else deltas
    endpoints.to_csv(out / 'endpoints.csv', index=False)
    mean_deltas = pd.DataFrame()
    if len(deltas) and 'delta' in deltas:
        mean_deltas = (deltas.dropna(subset=['delta']).groupby(['comparison', 'condition', 'endpoint'])
                       .agg(mean_delta=('delta', 'mean'), folds=('fold', 'size'),
                            folds_with_truncated_arm=('truncated_arm', 'sum'),
                            folds_ci_above_0=('ci_low', lambda s: int((s > 0).sum())),
                            folds_ci_below_0=('ci_high', lambda s: int((s < 0).sum())))
                       .reset_index())
        mean_deltas = _order(mean_deltas, cond_order)
    mean_deltas.to_csv(out / 'paired_deltas_mean.csv', index=False)
    write_json(out / 'summary_metadata.json', dict(
        bootstrap=cfg['bootstrap'], seed=cfg['seed'], clusters='(dataset, group_id)',
        delta='AUC(arm_a) - AUC(arm_b) on identical videos; 95% percentile interval',
        endpoints={f'{a} - {b} @ {c}': v for (a, b, c), v in ENDPOINTS.items()},
        h264_note='h264_* and clean_h264subset use the same fixed video subset (h264_subset.csv per fold)',
        mean_note='mean_over_folds and paired_deltas_mean are point estimates; check the folds column',
        truncated_note='status truncated = training stopped early by the time guard; see truncated_arm, '
                       'status_a/status_b and truncated_folds before reading a delta'))
    make_plots(results, hist, out, cond_order)
    return dict(results=results, wide=wide, mean=mean, deltas=deltas, endpoints=endpoints,
                mean_deltas=mean_deltas, history=hist)


def make_plots(results, hist, out, cond_order):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for held, g in results.groupby('fold'):
        conds = [c for c in cond_order if c in set(g.condition)]
        fig, ax = plt.subplots(figsize=(9, 4.8))
        for arm in [a for a in ARM_ORDER if a in set(g.arm)]:
            v = g[g.arm == arm].set_index('condition').auc.reindex(conds)
            ax.plot(conds, v.values, marker='o', label=arm, ls='--' if arm in ORIGINAL_DEFS else '-')
        ax.axhline(0.5, color='grey', lw=0.8, ls=':')
        ax.set(ylabel='video-level AUC', title=f'{held} held out: robustness (H.264 columns on the fixed subset)')
        ax.tick_params(axis='x', rotation=20)
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        fig.savefig(out / f'robustness_{held}.png', dpi=160)
        plt.close(fig)
    if hist.empty:
        return
    folds = [h for h in CORPORA if h in set(hist.fold)]
    fig, axes = plt.subplots(len(folds), 2, figsize=(10, 3.2 * len(folds)), squeeze=False)
    for r, held in enumerate(folds):
        for arm, g in hist[hist.fold == held].groupby('arm'):
            axes[r, 0].plot(g.epoch, g.train_loss, marker='o', label=arm)
            axes[r, 1].plot(g.epoch, g.source_val_auc, marker='o', label=arm)
        axes[r, 0].set(xlabel='epoch', ylabel='train BCE', title=f'{held} held out')
        axes[r, 1].set(xlabel='epoch', ylabel='source-val macro AUC', title=f'{held} held out')
    axes[0, 1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / 'training_curves.png', dpi=160)
    plt.close(fig)


def zip_out(out):
    """Everything except checkpoints (.pth stays in /kaggle/working for the version output)."""
    faulthandler.cancel_dump_traceback_later()   # the run is finished: disarm the hang watchdog
    out = Path(out)
    archive = out.with_suffix('.zip')
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.rglob('*')):
            if p.is_file() and p.suffix != '.pth':
                z.write(p, p.relative_to(out.parent))
    return archive


# ---------------------------------------------------------------- testing

def _to_tensor(image):
    return torch.from_numpy(np.asarray(image, dtype=np.float32).copy()).permute(2, 0, 1) / 255


def unit_tests(log=print):
    """CPU, synthetic data, no downloads."""
    torch.manual_seed(7)
    x = torch.randn(2, 3, 16, 24, requires_grad=True)
    m = BlockDCTMask(drop_table('qtable', 0.7), seed=1)
    c = m.coefficients(x)
    assert torch.allclose(m.reconstruct(c), x, atol=5e-6), 'DCT round trip'
    m.eval()
    assert m(x) is x, 'evaluation must disable masking'
    m.train()
    y = m(x)
    assert y.shape == x.shape and torch.isfinite(y).all() and not torch.allclose(x, y)
    assert torch.allclose(m.coefficients(y)[..., 0, 0], c[..., 0, 0], atol=1e-5), 'DC changed'
    assert m.coefficients(y).square().sum() <= c.square().sum() + 1e-3, 'energy increased'
    y.square().mean().backward()
    assert torch.isfinite(x.grad).all()
    log('PASS: DCT round trip, eval identity, DC preserved, energy, gradients')

    ac = np.ones((8, 8), bool)
    ac[0, 0] = False
    q = np.array(JPEG_LUMA, float)
    tq, tu, ti = (drop_table(k, 0.35) for k in MASK_KINDS)
    means = [t[ac].mean() for t in (tq, tu, ti)]
    assert max(means) - min(means) < 1e-6 and all(t[0, 0] == 0 for t in (tq, tu, ti))
    assert tq[7, 7] > tq[0, 1] and ti[0, 1] > ti[7, 7] and np.ptp(tu[ac]) == 0
    o = np.argsort(q[ac], kind='stable')
    assert (np.diff(tq[ac][o]) >= 0).all() and (np.diff(ti[ac][o]) <= 1e-12).all(), 'inverted order'
    assert np.allclose(np.sort(ti[ac]).sum(), np.sort(tq[ac]).sum())
    for t in (tq, tu, ti):
        keep = BlockDCTMask(t, 3).keep_mask(20000, torch.device('cpu'))[:, 0, 0, 0].float().mean(0).numpy()
        assert np.abs((1 - keep) - t).max() < 0.02, 'empirical drop rate differs from the table'
    try:
        drop_table('flat')
    except ValueError:
        pass
    else:
        raise AssertionError('unknown mask kind accepted')
    log(f'PASS: qmask/randmask/invmask mean AC drop {means[0]:.6f} each, orderings, empirical rates')

    a, b = BlockDCTMask(tq, 8).train(), BlockDCTMask(tq, 8).train()
    assert torch.equal(a(x.detach()), b(x.detach())), 'mask RNG not reproducible'
    state = torch.random.get_rng_state()
    a(x.detach())
    assert torch.equal(state, torch.random.get_rng_state()), 'mask moved the global RNG'

    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.rgb, self.fad = nn.Identity(), nn.Identity()

        def forward(self, z):
            return self.rgb(z), self.fad(z)

    w = MaskedFrequencyModel(Toy(), BlockDCTMask(tq, 1))
    rgb, freq = w(x.detach())
    assert torch.equal(rgb, x.detach()) and not torch.equal(freq, x.detach()), 'branch isolation'
    w.eval()
    rgb, freq = w(x.detach())
    assert torch.equal(rgb, freq)
    w.train()
    w.base.eval()   # validation scored through the bare base must not be masked either
    rgb, freq = w(x.detach())
    assert torch.equal(rgb, freq), 'hook active while base is in eval mode'
    w.train()
    w.remove()
    w.train()
    rgb, freq = w(x.detach())
    assert torch.equal(rgb, freq), 'hook not removed'
    log('PASS: independent mask RNG, branch isolation through the fad hook')

    with tempfile.TemporaryDirectory() as root:
        rng = np.random.default_rng(0)
        paths = []
        (Path(root) / 'crops').mkdir()
        for i in range(3):
            p = Path(root) / 'crops' / f'{i}.png'
            Image.fromarray(rng.integers(0, 256, (48, 40, 3), dtype=np.uint8)).save(p)
            paths.append(str(p))
        frame = pd.DataFrame(dict(path=paths, label=[0, 1, 0]))
        ds = FrameData(frame, _to_tensor, realjpeg=(1.0, (20, 95)), seed=5)
        ds.set_epoch(1)
        first = ds[0][0]
        ds2 = FrameData(frame, _to_tensor, realjpeg=(1.0, (20, 95)), seed=5)
        ds2.set_epoch(1)
        assert torch.equal(first, ds[0][0]) and torch.equal(first, ds2[0][0]), 'real-JPEG not deterministic'
        params = {realjpeg_params(5, e, 0, 0.5, (20, 95)) for e in range(10)}
        assert len(params) > 1, 'real-JPEG draws do not change across epochs'
        ds.set_epoch(2)
        assert any(not torch.equal(ds[i][0], ds2[i][0]) for i in range(3)), 'real-JPEG identical across epochs'
        hits = np.mean([realjpeg_params(5, 1, i, 0.5, (20, 95))[0] for i in range(4000)])
        qs = [realjpeg_params(5, 1, i, 1.0, (20, 95))[1] for i in range(4000)]
        assert abs(hits - 0.5) < 0.03 and min(qs) == 20 and max(qs) == 95
        clean = FrameData(frame, _to_tensor)[1][0]
        assert clean.shape == (3, 48, 40) and not torch.equal(clean, FrameData(frame, _to_tensor, quality=20)[1][0])

        tree = Path(root) / 'out' / 'phase14b' / 'dfdc' / 'control'
        tree.mkdir(parents=True)
        (tree / 'done.json').write_text('{}')
        shutil.copy(paths[0], tree.parent.parent / 'robustness_dfdc.png')   # Cell 8 plots sit next to <held>/
        hidden = Path(root) / 'crops' / 'sub' / 'dfdc' / 'control'          # below an image folder: never walked
        hidden.mkdir(parents=True)
        (hidden / 'done.json').write_text('{}')
        assert find_reuse_roots(top=root) == [Path(root) / 'out' / 'phase14b'], 'reuse discovery'
    log('PASS: real-JPEG augmentation deterministic per (seed, epoch, index), varies by epoch; reuse discovery')

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'   # the run's own GradScaler is the CUDA one
    toy = nn.Linear(4, 1).to(dev)
    opt = torch.optim.SGD(toy.parameters(), lr=0.1)
    scaler = torch.amp.GradScaler(dev, init_scale=8.0)
    tcfg = dict(device=dev, amp=False, grad_clip=1.0)
    xb, yb = torch.randn(6, 4, device=dev), torch.tensor([0., 1., 0., 1., 1., 0.], device=dev)
    hook = toy.weight.register_hook(lambda g: g * float('inf'))
    before = toy.weight.detach().clone()
    _, overflow = train_step(toy, xb, yb, nn.BCEWithLogitsLoss(), opt, scaler, tcfg)
    assert overflow and torch.equal(toy.weight.detach(), before) and scaler.get_scale() < 8.0, \
        'an inf gradient must skip the step, not raise'
    hook.remove()
    _, overflow = train_step(toy, xb, yb, nn.BCEWithLogitsLoss(), opt, scaler, tcfg)
    assert not overflow and not torch.equal(toy.weight.detach(), before)
    check_overflow(3, 12, 'test')
    check_overflow(28, 562, 'test')
    for bad in ((4, 12), (29, 562)):
        try:
            check_overflow(*bad, 'test')
        except FloatingPointError:
            pass
        else:
            raise AssertionError(f'{bad[0]} of {bad[1]} overflowing steps accepted')
    log('PASS: GradScaler overflow path skips the step; more than max(5%, 3) overflows raises')

    rng = np.random.default_rng(1)
    for _ in range(5):
        yy = rng.integers(0, 2, 300)
        ss = np.round(rng.normal(size=300) + yy * 0.7, 1)   # rounded: many ties
        sk = roc_auc_score(yy, ss)
        assert abs(rank_auc(yy, ss) - sk) < 1e-12 and abs(weighted_auc(yy, ss)[0] - sk) < 1e-12
        cnt = rng.integers(0, 3, 300)
        rep = np.repeat(np.arange(300), cnt)
        assert abs(weighted_auc(yy, ss, cnt)[0] - roc_auc_score(yy[rep], ss[rep])) < 1e-12
    v = pd.DataFrame(dict(dataset='dfdc', video_id=[f'v{i}' for i in range(60)],
                          group_id=[f'g{i // 2}' for i in range(60)], label=rng.integers(0, 2, 60)))
    v['probability'] = rng.uniform(size=60)
    r = paired_bootstrap(v, v, 200, 42)
    assert r['delta'] == 0 and r['ci_low'] == 0 and r['ci_high'] == 0, 'identical arms must give exactly 0'
    v2 = v.assign(probability=np.clip(v.probability + 0.3 * v.label, 0, 1))
    r2 = paired_bootstrap(v2, v, 200, 42)
    assert r2['delta'] > 0 and r2['clusters'] == 30 and r2['ci_low'] <= r2['delta'] <= r2['ci_high']
    sub = h264_subset(pd.concat([v] * 2, ignore_index=True), 5, 42)
    assert sub.equals(h264_subset(v, 5, 42)) and (sub.label.value_counts() == 5).all()
    log('PASS: rank AUC = sklearn (with ties and integer weights), paired bootstrap identical arms = 0, '
        'H.264 subset fixed by seed')

    exe, note = find_ffmpeg()
    if exe is None:
        log(f'SKIP: H.264 round trip test ({note}); the H.264 conditions will be skipped')
    else:
        g = np.linspace(0, 255, 224)
        base = np.stack([np.add.outer(g, g) / 2] * 3, -1)
        frames = np.stack([np.clip(base + rng.normal(0, 20, base.shape) + 5 * k, 0, 255)
                           for k in range(5)]).astype(np.uint8)
        err = {}
        for crf in (23, 40):
            out = h264_roundtrip(frames, crf, exe)
            assert out.shape == frames.shape and out.dtype == np.uint8, 'H.264 round trip shape'
            err[crf] = np.abs(out.astype(int) - frames).mean()
        assert err[40] > err[23], 'CRF 40 should lose more than CRF 23'
        log(f'PASS: H.264 round trip, 5 frames 224x224x3 uint8; mean abs error CRF23 {err[23]:.1f}, '
            f'CRF40 {err[40]:.1f} ({note})')
    return True
