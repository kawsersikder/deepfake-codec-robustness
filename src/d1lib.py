"""D1: train-time codec augmentations for EVA-02 + LoRA (single-view arms).

Every augmentation acts on the PIL face crop before the backbone's training
transform; the evaluation transform is never touched. Masks reuse codec7's drop
tables and its block-DCT math (one 8x8 keep mask per image, shared by every
block and channel, DC never dropped, no rescaling), applied here to the crop at
native resolution in pixel space, because EVA-02 has no frequency stream.

Strength-matched arms (the *_lowE / *_highE arms) keep a mask's ordering but rescale its
drop probabilities so that the expected squared error it adds per pixel equals that of
another original table, measured on the fold's own training crops (strength_tables).

Randomness comes from Python's `random` module unless a generator is passed:
torch reseeds it in every DataLoader worker (base seed + worker id, new base
seed each epoch), so workers never draw identical masks. Self-contained apart
from codec7 (drop_table, h264_roundtrip, jpeg_roundtrip, paired_bootstrap).
Nothing runs on import.
"""
import contextlib
import math
import random
import subprocess

import numpy as np
import pandas as pd
from PIL import Image

import codec7 as C7


MASK_KINDS = C7.MASK_KINDS   # ('qtable', 'uniform', 'inverted')
# Each arm is an ordered list of operations on the crop; the arms differ only in this.
ARM_OPS = {
    'plain':             (),
    'realh264':          ('h264',),
    'qmask':             ('qtable',),
    'randmask':          ('uniform',),
    'realjpeg':          ('jpeg',),
    'randmask_realh264': ('uniform', 'h264'),
    'qmask_realh264':    ('qtable', 'h264'),
    'invmask':           ('inverted',),
    'noaug':             ('identity',),   # retrained without augmentation: the same-session control for plain
    # strength-matched: lowE = the strength of qmask's table, highE = the strength of randmask's table
    'randmask_lowE':     ('uniform_at_qtable',),
    'invmask_lowE':      ('inverted_at_qtable',),
    'qmask_highE':       ('qtable_at_uniform',),
    'invmask_highE':     ('inverted_at_uniform',),
}
ARM_ORDER = list(ARM_OPS)
MATCHED = {   # matched op -> (ordering it keeps, original table whose strength it takes)
    'uniform_at_qtable': ('uniform', 'qtable'),
    'inverted_at_qtable': ('inverted', 'qtable'),
    'qtable_at_uniform': ('qtable', 'uniform'),
    'inverted_at_uniform': ('inverted', 'uniform'),
}
OP_KEYS = {   # config keys each operation depends on (enter the arm's signature)
    'qtable': ('max_drop',), 'uniform': ('max_drop',), 'inverted': ('max_drop',),
    'h264': ('aug_h264_p', 'aug_h264_crf', 'aug_h264_size', 'aug_h264_preset', 'aug_h264_fps'),
    'jpeg': ('aug_jpeg_p', 'aug_jpeg_q', 'jpeg_subsampling'),
    'identity': (),
    **{op: ('max_drop', 'energy_frames', 'energy_seed') for op in MATCHED},
}


def trained(arm):
    return bool(ARM_OPS[arm])


def arm_params(arm, cfg):
    keys = sorted({k for op in ARM_OPS[arm] for k in OP_KEYS[op]})
    return {k: cfg[k] for k in keys}


# ------------------------------------------------------------------ masks

def dct_basis():
    """Orthonormal 8-point DCT-II matrix, identical to codec7.BlockDCTMask.basis."""
    n = np.arange(8, dtype=np.float64)
    basis = np.cos(math.pi * (2 * n + 1) * n[:, None] / 16) * math.sqrt(2 / 8)
    basis[0] /= math.sqrt(2)
    return basis


BASIS = dct_basis()


def draw_keep(table, rng):
    """One 8x8 keep mask: coefficient (u, v) is dropped with probability table[u, v]."""
    return rng.random((8, 8)) >= table


def mask_array(x, keep):
    """float (h, w, c) -> float (h, w, c). Edge-pad to a multiple of 8, zero the dropped
    coefficients of every 8x8 block of every channel, crop back. Not clipped."""
    h, w, c = x.shape
    x = np.pad(x, ((0, -h % 8), (0, -w % 8), (0, 0)), mode='edge')
    H, W = x.shape[:2]
    blocks = x.reshape(H // 8, 8, W // 8, 8, c).transpose(0, 2, 4, 1, 3)   # (bh, bw, c, 8, 8)
    coeff = BASIS @ blocks @ BASIS.T
    out = BASIS.T @ (coeff * keep) @ BASIS
    return out.transpose(0, 3, 1, 4, 2).reshape(H, W, c)[:h, :w]


def mask_image(image, table, rng):
    x = np.asarray(image.convert('RGB'), dtype=np.float64)
    out = mask_array(x, draw_keep(table, rng))
    return Image.fromarray(np.clip(np.rint(out), 0, 255).astype(np.uint8))


# ------------------------------------------------- strength-matched masks
# Strength = the expected squared error a mask adds per pixel. The block DCT is orthonormal, so
# dropping coefficient (u, v) of a block removes exactly its squared value, and the expected
# removed energy is linear in the table: sum over (u, v) of p[u, v] * e[u, v], with e the mean
# per-pixel energy of that coefficient over the training crops (native size, as masks are applied).

def coefficient_energy(images):
    """8x8 mean per-pixel energy of each DCT coefficient over PIL crops at native size, edge-padded
    to a multiple of 8 as mask_array pads them. Every crop counts once, as in training."""
    total, n = np.zeros((8, 8)), 0
    for image in images:
        x = np.asarray(image.convert('RGB'), dtype=np.float64)
        h, w, c = x.shape
        x = np.pad(x, ((0, -h % 8), (0, -w % 8), (0, 0)), mode='edge')
        H, W = x.shape[:2]
        coeff = BASIS @ x.reshape(H // 8, 8, W // 8, 8, c).transpose(0, 2, 4, 1, 3) @ BASIS.T
        total += (coeff ** 2).sum((0, 1, 2)) / (H * W * c)
        n += 1
    if not n:
        raise ValueError('No images for the energy calibration')
    return total / n


def removed_energy(table, energy):
    """Expected squared error per pixel that a drop table adds (its strength)."""
    return float((np.asarray(table, dtype=np.float64) * energy).sum())


AC_INDEX = [(u, v) for u in range(8) for v in range(8) if (u, v) != (0, 0)]


def resized_gram(images, size=224):
    """63x63 mean per-pixel Gram matrix of the AC components after a bicubic resize to size x size (or to
    (width, height)), with PIL as the training transform resizes PIL crops. Component k of a crop is what
    dropping coefficient k from every block removes. The expected squared error after the resize is then
    exact for any table: p @ G @ p + sum_k (p_k - p_k^2) G_kk (removed_energy_resized), as drops are independent."""
    wh = (size, size) if np.isscalar(size) else tuple(size)
    total, n = np.zeros((63, 63)), 0
    for image in images:
        x = np.asarray(image.convert('RGB'), dtype=np.float64)
        h, w, c = x.shape
        x = np.pad(x, ((0, -h % 8), (0, -w % 8), (0, 0)), mode='edge')
        H, W = x.shape[:2]
        coeff = BASIS @ x.reshape(H // 8, 8, W // 8, 8, c).transpose(0, 2, 4, 1, 3) @ BASIS.T
        rows = []
        for u, v in AC_INDEX:
            one = np.zeros((8, 8))
            one[u, v] = 1.0
            comp = (BASIS.T @ (coeff * one) @ BASIS).transpose(0, 3, 1, 4, 2).reshape(H, W, c)[:h, :w]
            rows.append(np.concatenate([np.asarray(Image.fromarray(comp[:, :, ch].astype(np.float32), 'F')
                                                   .resize(wh, Image.Resampling.BICUBIC)).ravel()
                                        for ch in range(c)]))
        a = np.asarray(rows, dtype=np.float64)
        total += a @ a.T / a.shape[1]
        n += 1
    if not n:
        raise ValueError('No images for the Gram matrix')
    return total / n


def removed_energy_resized(table, gram):
    """Expected squared error per pixel a drop table adds, measured after the resize (resized_gram)."""
    p = np.array([table[u][v] for u, v in AC_INDEX], dtype=np.float64)
    return float(p @ gram @ p + ((p - p ** 2) * np.diag(gram)).sum())


def scaled_table(kind, scale, max_drop=0.35):
    """The original `kind` table times `scale`, clipped to [0, 1]; scale 1 is the original."""
    return np.clip(C7.drop_table(kind, max_drop) * scale, 0.0, 1.0)


def matched_table(kind, target, energy, max_drop=0.35):
    """(table, scale): the scaled `kind` table whose removed energy equals `target`. The removed
    energy rises monotonically with the scale, from 0 to all AC energy, so bisection finds it."""
    base = C7.drop_table(kind, max_drop)
    top = 1.0 / base[base > 0].min()   # at this scale every AC coefficient is dropped with p = 1
    if not 0 < target < removed_energy(scaled_table(kind, top, max_drop), energy):
        raise ValueError(f'{kind}: target {target} is outside what the ordering can reach')
    lo, hi = 0.0, top
    for _ in range(100):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if removed_energy(scaled_table(kind, mid, max_drop), energy) < target else (lo, mid)
    table = scaled_table(kind, hi, max_drop)
    if abs(removed_energy(table, energy) - target) > 1e-6 * target:
        raise AssertionError(f'{kind}: bisection did not reach the target strength')
    return table, hi


def strength_tables(energy, max_drop=0.35):
    """Every matched op's table for one fold, with the facts to report: scale, mean AC drop
    probability, removed energy per pixel and the PSNR that energy corresponds to."""
    ac = np.ones((8, 8), bool)
    ac[0, 0] = False

    def facts(table, scale=1.0):
        e, p = removed_energy(table, energy), table[ac]
        sure = p >= 1.0 - 1e-12
        q = np.clip(p, 1e-12, 1 - 1e-12)
        return dict(table=np.asarray(table), scale=float(scale), mean_ac_drop=float(p.mean()),
                    removed_energy=e, psnr_db=float(10 * np.log10(255.0 ** 2 / e)),
                    share_of_ac_energy=e / float(energy[ac].sum()),
                    expected_dropped=float(p.sum()),                       # AC coefficients dropped per image
                    untouched_share=float(np.prod(1 - p)),                 # images the mask leaves unchanged
                    mask_entropy_bits=float(np.where((p > 0) & ~sure, -(q * np.log2(q) + (1 - q) * np.log2(1 - q)), 0).sum()),
                    always_dropped=int(sure.sum()),                        # coefficients removed from every image
                    deterministic_share=float(energy[ac][sure].sum() / e)) # removed energy that is not random

    original = {kind: facts(C7.drop_table(kind, max_drop)) for kind in MASK_KINDS}
    ops = {}
    for op, (kind, like) in MATCHED.items():
        table, scale = matched_table(kind, original[like]['removed_energy'], energy, max_drop)
        ops[op] = dict(facts(table, scale), ordering=kind, strength_of=like)
    return dict(energy=energy, original=original, ops=ops)


# ------------------------------------------------------------ real codecs

def h264_image(image, crf, exe, size=224, preset='veryfast', fps=25):
    """Resize to size x size (PIL bilinear, as Gate 0's clips), one-frame libx264 clip, decode."""
    a = np.asarray(image.convert('RGB').resize((size, size), Image.Resampling.BILINEAR), dtype=np.uint8)
    try:
        out = C7.h264_roundtrip(a[None], crf, exe, preset, fps, threads=1)
    except (RuntimeError, subprocess.TimeoutExpired):   # one retry: a transient ffmpeg failure would kill a fold
        out = C7.h264_roundtrip(a[None], crf, exe, preset, fps, threads=1)
    return Image.fromarray(out[0])


class CodecAug:
    """Applies `ops` in order to a PIL crop. Mask ops always apply (p = 1); 'h264' and
    'jpeg' apply with their own probability and draw their quality per call; 'identity' does nothing.
    Strength-matched ops take their fold's table from `tables` ({op: 8x8 table}, strength_tables)."""

    def __init__(self, ops, max_drop=0.35, exe=None, h264_p=0.5, h264_crf=(23, 40), h264_size=224,
                 h264_preset='veryfast', h264_fps=25, jpeg_p=0.5, jpeg_q=(20, 95), jpeg_subsampling=2,
                 tables=None):
        unknown = [op for op in ops if op not in MASK_KINDS + tuple(MATCHED) + ('h264', 'jpeg', 'identity')]
        if unknown or not ops:
            raise ValueError(f'Unknown or empty ops {ops}')
        if 'h264' in ops and exe is None:
            raise ValueError('The h264 op needs a working ffmpeg (codec7.find_ffmpeg)')
        missing = [op for op in ops if op in MATCHED and op not in (tables or {})]
        if missing:
            raise ValueError(f'Strength-matched ops {missing} need their fold table (strength_tables)')
        self.ops, self.exe = tuple(ops), exe
        self.tables = {op: C7.drop_table(op, max_drop) for op in ops if op in MASK_KINDS}
        self.tables.update({op: np.asarray(tables[op], dtype=np.float64) for op in ops if op in MATCHED})
        for op, t in self.tables.items():
            if t.shape != (8, 8) or t[0, 0] != 0 or t.min() < 0 or t.max() > 1:
                raise ValueError(f'{op}: a drop table must be 8x8 in [0, 1] with DC = 0')
        self.h264 = dict(p=h264_p, crf=tuple(h264_crf), size=h264_size, preset=h264_preset, fps=h264_fps)
        self.jpeg = dict(p=jpeg_p, q=tuple(jpeg_q), subsampling=jpeg_subsampling)

    def __call__(self, image, rng=None):
        # Default: the module RNG, which torch reseeds per worker. A Random object stored on
        # self would be copied into every forked worker with the same state.
        rng = random if rng is None else rng
        for op in self.ops:
            if op in self.tables:
                image = mask_image(image, self.tables[op], np.random.default_rng(rng.getrandbits(64)))
            elif op == 'h264':
                if rng.random() < self.h264['p']:
                    image = h264_image(image, rng.randint(*self.h264['crf']), self.exe, self.h264['size'],
                                       self.h264['preset'], self.h264['fps'])
            elif op == 'jpeg' and rng.random() < self.jpeg['p']:
                image = C7.jpeg_roundtrip(image, rng.randint(*self.jpeg['q']), self.jpeg['subsampling'])
        return image

    def __repr__(self):
        return f'CodecAug({self.ops})'


def make_aug(arm, cfg, exe=None, tables=None):
    """None for 'plain'; otherwise the arm's CodecAug built from the config. tables: the fold's
    {op: table} for strength-matched ops (ignored by the other arms)."""
    ops = ARM_OPS[arm]
    if not ops:
        return None
    return CodecAug(ops, max_drop=cfg['max_drop'], exe=exe, h264_p=cfg['aug_h264_p'],
                    h264_crf=cfg['aug_h264_crf'], h264_size=cfg['aug_h264_size'],
                    h264_preset=cfg['aug_h264_preset'], h264_fps=cfg['aug_h264_fps'],
                    jpeg_p=cfg['aug_jpeg_p'], jpeg_q=cfg['aug_jpeg_q'], jpeg_subsampling=cfg['jpeg_subsampling'],
                    tables=tables)


# -------------------------------------------------------- transform wrap

class AugTransform:
    """aug(PIL) then base_tf; aug None passes the crop straight to base_tf. Top level, so it pickles."""

    def __init__(self, base_tf, aug):
        self.base_tf, self.aug = base_tf, aug

    def __call__(self, image):
        return self.base_tf(image if self.aug is None else self.aug(image))

    def __repr__(self):
        return f'AugTransform({self.aug!r}, {self.base_tf!r})'


@contextlib.contextmanager
def augmented_transforms(namespace, aug):
    """Rebind namespace['backbone_transforms'] so that it returns (AugTransform(tf_train, aug),
    tf_eval, norm); the original is restored on exit, also after an exception. Pass the
    namespace the library functions resolve it in (run_lora_fold.__globals__)."""
    original = namespace['backbone_transforms']
    if getattr(original, 'd1_original', None) is not None:
        raise RuntimeError('backbone_transforms is already wrapped')

    def wrapped(model, size):
        wrapped.calls += 1
        tf_train, tf_eval, norm = original(model, size)
        return AugTransform(tf_train, aug), tf_eval, norm

    wrapped.calls, wrapped.d1_original = 0, original
    namespace['backbone_transforms'] = wrapped
    try:
        yield wrapped
    finally:
        namespace['backbone_transforms'] = original


# ---------------------------------------------------------------- testing

def _synthetic(h, w, seed):
    rng = np.random.default_rng(seed)
    g = np.add.outer(np.linspace(0, 200, h), np.linspace(0, 55, w))
    x = np.stack([g, g[::-1], g[:, ::-1]], -1) + rng.normal(0, 25, (h, w, 3))
    return Image.fromarray(np.clip(x, 0, 255).astype(np.uint8))


def unit_tests(exe=None, log=print):
    """CPU, synthetic images, no downloads. exe: ffmpeg from codec7.find_ffmpeg() or None."""
    ac = np.ones((8, 8), bool)
    ac[0, 0] = False
    tables = {k: C7.drop_table(k, 0.35) for k in MASK_KINDS}
    means = [t[ac].mean() for t in tables.values()]
    assert max(means) - min(means) < 1e-9 and all(t[0, 0] == 0 for t in tables.values()), 'tables'
    assert np.allclose(BASIS @ BASIS.T, np.eye(8)), 'DCT basis not orthonormal'
    rng = np.random.default_rng(0)
    for kind, t in tables.items():
        keep = np.stack([draw_keep(t, rng) for _ in range(20000)])
        assert keep[:, 0, 0].all(), f'{kind}: DC dropped'
        assert np.abs((1 - keep.mean(0)) - t).max() < 0.02, f'{kind}: empirical drop rate differs from the table'
    log(f'PASS: qmask/randmask/invmask tables share mean AC drop {means[0]:.6f}, DC kept, empirical rates match')

    x = np.asarray(_synthetic(64, 48, 1), dtype=np.float64)
    assert np.allclose(mask_array(x, np.ones((8, 8), bool)), x, atol=1e-9), 'keep-all mask is not the identity'
    for kind, t in tables.items():
        y = mask_array(x, draw_keep(t, np.random.default_rng(3)))
        bx = x.reshape(8, 8, 6, 8, 3).mean((1, 3))
        by = y.reshape(8, 8, 6, 8, 3).mean((1, 3))
        assert np.allclose(bx, by, atol=1e-9), f'{kind}: block DC (mean) changed'
        assert ((y - by.repeat(8, 0).repeat(8, 1)) ** 2).sum() <= ((x - bx.repeat(8, 0).repeat(8, 1)) ** 2).sum() + 1e-6, \
            f'{kind}: AC energy increased'
    for h, w in [(128, 128), (131, 97), (160, 160), (7, 13)]:
        img = _synthetic(h, w, h + w)
        same = mask_image(img, np.zeros((8, 8)), np.random.default_rng(0))
        assert np.array_equal(np.asarray(same), np.asarray(img)), f'{h}x{w}: zero table changed the crop'
        for kind, t in tables.items():
            out = mask_image(img, t, np.random.default_rng(5))
            a = np.asarray(out)
            assert out.size == img.size and out.mode == 'RGB' and a.dtype == np.uint8, f'{kind} {h}x{w}: size/type'
            assert a.min() >= 0 and a.max() <= 255
            if min(h, w) >= 8:
                assert not np.array_equal(a, np.asarray(img)), f'{kind} {h}x{w}: crop unchanged'
    log('PASS: masks keep the DC of every block, never add AC energy, keep size and [0, 255], pad/crop non-multiples of 8')

    img = _synthetic(131, 97, 7)
    aug = CodecAug(('qtable',))
    outs = [np.asarray(aug(img)) for _ in range(6)]   # successive samples in one worker
    assert all(not np.array_equal(outs[0], o) for o in outs[1:]), 'identical masks across samples'
    random.seed(1001)
    a = np.asarray(aug(img))
    random.seed(1002)   # what torch does in worker 2: base seed + worker id
    b = np.asarray(aug(img))
    random.seed(1001)
    assert not np.array_equal(a, b) and np.array_equal(a, np.asarray(aug(img))), 'per-worker reseeding'
    r1, r2 = random.Random(9), random.Random(9)
    assert np.array_equal(np.asarray(aug(img, r1)), np.asarray(aug(img, r2))), 'explicit RNG not reproducible'
    log('PASS: masks vary across samples and across workers (module RNG reseeded per worker), reproducible by seed')

    jp = CodecAug(('jpeg',), jpeg_p=1.0, jpeg_q=(20, 20))
    out = jp(img)
    assert out.size == img.size and not np.array_equal(np.asarray(out), np.asarray(img)), 'real JPEG'
    assert np.array_equal(np.asarray(out), np.asarray(C7.jpeg_roundtrip(img, 20))), 'real JPEG differs from codec7'
    never = CodecAug(('jpeg',), jpeg_p=0.0)
    assert never(img) is img, 'p = 0 must pass through'
    r = random.Random(4)
    half = CodecAug(('jpeg',), jpeg_p=0.5, jpeg_q=(20, 95))
    hits = np.mean([half(img, r) is not img for _ in range(400)])
    assert 0.4 < hits < 0.6, f'real-JPEG rate {hits}'
    log(f'PASS: real JPEG at native size (equals codec7.jpeg_roundtrip), applied at rate {hits:.2f} for p = 0.5')

    if exe is None:
        log('SKIP: realh264 augmentation test (no ffmpeg with libx264)')
    else:
        err = {}
        for crf in (23, 40):
            out = CodecAug(('h264',), exe=exe, h264_p=1.0, h264_crf=(crf, crf))(img)
            ref = np.asarray(img.resize((224, 224), Image.Resampling.BILINEAR), dtype=np.int16)
            assert out.size == (224, 224) and out.mode == 'RGB', 'realh264 size'
            err[crf] = np.abs(np.asarray(out, dtype=np.int16) - ref).mean()
        assert err[40] > err[23] > 0, f'CRF 40 must change the crop more than CRF 23: {err}'
        both = CodecAug(('qtable', 'h264'), exe=exe, h264_p=1.0)(img)
        assert both.size == (224, 224)
        log(f'PASS: realh264 one-frame clip 224x224, mean abs change CRF23 {err[23]:.1f}, CRF40 {err[40]:.1f}; '
            f'mask then H.264 composes')

    marker = object()
    assert AugTransform(lambda im: im, None)(marker) is marker, 'AugTransform(None) must pass through'
    seen = []
    t = AugTransform(lambda im: seen.append(im) or 'out', lambda im: ('aug', im))
    assert t('x') == 'out' and seen == [('aug', 'x')], 'AugTransform order'

    def fake_bt(model, size):
        return 'train', 'eval', {'mean': 0}

    ns = {'backbone_transforms': fake_bt}
    try:
        with augmented_transforms(ns, 'AUG') as w:
            tr, ev, _ = ns['backbone_transforms'](None, 224)
            assert isinstance(tr, AugTransform) and tr.aug == 'AUG' and ev == 'eval' and w.calls == 1
            raise KeyError('simulated failure inside training')
    except KeyError:
        pass
    assert ns['backbone_transforms'] is fake_bt, 'backbone_transforms not restored after an exception'
    with augmented_transforms(ns, None):
        try:
            with augmented_transforms(ns, None):
                raise AssertionError('nested wrap accepted')
        except RuntimeError:
            pass
    assert ns['backbone_transforms'] is fake_bt
    log('PASS: AugTransform passes through with aug None; rebinding wraps tf_train only and is restored after an exception')

    assert make_aug('plain', {}) is None and [a for a in ARM_ORDER if not trained(a)] == ['plain']
    assert trained('noaug') and CodecAug(('identity',))(img) is img and arm_params('noaug', {}) == {}
    assert ARM_OPS['qmask_realh264'] == ('qtable', 'h264') and ARM_OPS['randmask_realh264'] == ('uniform', 'h264')

    imgs = [_synthetic(h, w, s) for s, (h, w) in enumerate([(96, 96), (131, 97), (160, 144), (64, 80)])]
    st = strength_tables(coefficient_energy(imgs), 0.35)
    for kind in MASK_KINDS:
        assert np.allclose(scaled_table(kind, 1.0), C7.drop_table(kind, 0.35)), f'{kind}: scale 1 is not the original'
    for op, info in st['ops'].items():
        kind, like = MATCHED[op]
        target, t = st['original'][like]['removed_energy'], info['table']
        assert abs(info['removed_energy'] - target) <= 1e-6 * target, f'{op}: strength not matched'
        assert t[0, 0] == 0 and t.min() >= 0 and t.max() <= 1, f'{op}: invalid table'
        order = np.argsort(C7.drop_table(kind, 0.35)[ac], kind='stable')
        assert np.all(np.diff(t[ac][order]) >= -1e-12), f'{op}: the ordering changed'
    up, down = st['ops']['qtable_at_uniform'], st['ops']['uniform_at_qtable']
    assert up['scale'] > 1 > down['scale'], 'qmask must be strengthened and randmask weakened'
    xs = [np.asarray(im, dtype=np.float64) for im in imgs]
    rng = np.random.default_rng(11)
    for op in ('qtable_at_uniform', 'uniform_at_qtable'):   # what mask_array removes matches the expectation
        t, want = st['ops'][op]['table'], st['ops'][op]['removed_energy']
        got = np.mean([np.mean((mask_array(x, draw_keep(t, rng)) - x) ** 2) for x in xs for _ in range(150)])
        assert abs(got - want) / want < 0.1, f'{op}: removed {got:.2f} per pixel, expected {want:.2f}'
    try:
        CodecAug(('qtable_at_uniform',))
        raise AssertionError('a strength-matched op without its table was accepted')
    except ValueError:
        pass
    cfg = dict(max_drop=0.35, aug_h264_p=0.5, aug_h264_crf=(23, 40), aug_h264_size=224, aug_h264_preset='veryfast',
               aug_h264_fps=25, aug_jpeg_p=0.5, aug_jpeg_q=(20, 95), jpeg_subsampling=2, energy_frames=600, energy_seed=42)
    tabs = {op: v['table'] for op, v in st['ops'].items()}
    aug = make_aug('qmask_highE', cfg, tables=tabs)
    assert np.array_equal(aug.tables['qtable_at_uniform'], tabs['qtable_at_uniform'])
    assert np.asarray(aug(img)).shape == np.asarray(img).shape
    assert arm_params('qmask_highE', cfg) == dict(energy_frames=600, energy_seed=42, max_drop=0.35)
    assert arm_params('qmask', cfg) == dict(max_drop=0.35), 'old arm signatures must not change (reuse)'
    sq = [_synthetic(64, 48, 21), _synthetic(48, 40, 22)]
    base = resized_gram([_synthetic(64, 48, 21)], size=(48, 64))   # (width, height) unchanged: the resize is the identity
    e = coefficient_energy([_synthetic(64, 48, 21)])
    assert np.allclose(np.diag(base), [e[u, v] for u, v in AC_INDEX], rtol=1e-4), 'Gram diagonal != coefficient energy'
    assert np.abs(base - np.diag(np.diag(base))).max() < 1e-3 * np.diag(base).max(), 'AC components not orthogonal'
    t = C7.drop_table('uniform', 0.35)
    assert abs(removed_energy_resized(t, base) - removed_energy(t, e)) < 1e-3 * removed_energy(t, e)
    g224 = resized_gram(sq, size=224)
    want = removed_energy_resized(t, g224)
    rng = np.random.default_rng(13)
    got = np.mean([np.mean((np.asarray(Image.fromarray(mask_array(np.asarray(im, dtype=np.float64), draw_keep(t, rng))
                                                       .astype(np.float32)[:, :, ch], 'F').resize((224, 224), Image.Resampling.BICUBIC))
                            - np.asarray(Image.fromarray(np.asarray(im, dtype=np.float32)[:, :, ch], 'F')
                                         .resize((224, 224), Image.Resampling.BICUBIC))) ** 2)
                    for im in sq for _ in range(150) for ch in range(3)])
    assert abs(got - want) / want < 0.08, f'resized Gram: expected {want:.2f}, Monte Carlo {got:.2f}'
    log(f"PASS: strength-matched tables hit their target strength, keep their ordering, and mask_array removes "
        f"the expected energy (qmask x{up['scale']:.2f} up, randmask x{down['scale']:.2f} down on synthetic crops); "
        f"the exact strength after the 224 px resize matches Monte Carlo ({want:.1f} vs {got:.1f})")

    v = np.random.default_rng(2)
    pred = pd.DataFrame(dict(dataset='dfdc', video_id=[f'v{i}' for i in range(60)],
                             group_id=[f'v{i}' for i in range(60)], label=v.integers(0, 2, 60)))
    pred['probability'] = v.uniform(size=60)
    r = C7.paired_bootstrap(pred, pred.copy(), 200, 42)
    assert r['delta'] == 0 and r['ci_low'] == 0 and r['ci_high'] == 0, 'identical arms must give exactly 0'
    log('PASS: paired bootstrap of identical arms = 0 [0, 0]')
    return True
