"""experiments/target_eval.py

Per-target scoring shared by `eval_summary.py` (the methods x targets table) and
`run_dynamics.py` (metrics along training):

    te = TargetEval.load(target)          # "skewnormal" | "sur" | "iohmm" | "lda"
    te.log_prob(x)                        # batched log pi~ (numpy in, numpy out)
    te.assign(x)                          # mode label per draw; te.n_modes = off-mode label
    te.mode_weights(x)                    # (n_modes + 1,) empirical weights, last = off-mode
    te.sw2(x, key)                        # sliced W2 to the reference (folded, see below)

Modes and reference, per target:
  - skewnormal: the 2 true components; nearest component under the standardized
    distance (x - mu_k) / sigma_k (no off-mode bin); true weights; reference = exact
    draws from the target.
  - sur: no ground-truth modes, so the 3 local maxima of log pi found by gradient
    ascent from the NUTS reference (`data/sur/sur_ref_modes.npz`). A draw's mode is its
    basin of attraction: `SUR_ASCENT_STEPS` Adam ascent steps on log pi, then the nearest
    maximum. Reference weights: the basin fractions of the NUTS reference.
  - iohmm: the 6 label-permutation modes (the `mode_centers` saved by `run_iohmm.py`),
    nearest center under a per-coordinate standardization, with an in-basin radius
    (`iohmm_reference_basins`); uniform true weights.
  - lda: the 2 topic-swap modes of the NUTS reference (`lda_reference_basins`); uniform
    true weights.

SW2 (sliced 2-Wasserstein, `n_proj` random directions, `n_sw` draws of each side). For
the symmetric targets (iohmm, lda) every draw, method and reference alike, is first
folded into the basin of mode 0 by the exact symmetry: the unfolded SW2 is dominated by
the mass split, which the mode TV already scores, so SW2 measures the within-mode shape.
skewnormal and sur are not folded.
"""
from __future__ import annotations

import itertools
import os
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import common
from src.utils.eval import _sw2

SUR_ASCENT_STEPS = 500
SUR_ASCENT_LR = 2e-2


def _chunked(fn, x, size=2048):
    return np.concatenate([np.asarray(fn(jnp.asarray(x[i:i + size], dtype=jnp.float64)))
                           for i in range(0, x.shape[0], size)]) if x.shape[0] else np.zeros((0,))


def is_metrics(logp, logq):
    """(ESS / n, Var(log w)) of the self-normalized importance weights w = pi / q."""
    lw = np.asarray(logp, np.float64) - np.asarray(logq, np.float64)
    lw = lw[np.isfinite(lw)]
    if not lw.size:
        return float("nan"), float("nan")
    w = np.exp(lw - lw.max())
    return float(w.sum() ** 2 / np.sum(w * w) / lw.size), float(np.var(lw))


@dataclass
class TargetEval:
    name: str
    d: int
    log_prob_jax: Callable          # jitted (n, d) -> (n,)
    reference: np.ndarray           # (N, d) reference draws (unfolded)
    n_modes: int
    true_weights: Optional[np.ndarray]     # (n_modes,); sur: the reference's basin weights
    symmetric: bool
    _assign: Callable               # x -> labels in [0, n_modes] (n_modes = off-mode)
    _fold: Optional[Callable] = None
    centers: Optional[np.ndarray] = None

    @classmethod
    def load(cls, name: str, results_dir: Optional[str] = None) -> "TargetEval":
        """`results_dir`: the benchmark runs of `name` (iohmm reads its mode centers there)."""
        return {"skewnormal": _skewnormal, "sur": _sur, "iohmm": _iohmm, "lda": _lda}[name](results_dir)

    def log_prob(self, x) -> np.ndarray:
        return _chunked(self.log_prob_jax, np.asarray(x, np.float64))

    def assign(self, x) -> np.ndarray:
        return np.asarray(self._assign(np.asarray(x, np.float64)))

    def mode_weights(self, x) -> np.ndarray:
        lab = self.assign(x)
        return np.bincount(lab, minlength=self.n_modes + 1).astype(np.float64) / max(lab.size, 1)

    def mode_tv(self, x) -> float:
        """0.5 * (sum_k |phat_k - w_k| + p_off): off-mode draws count fully as error."""
        ph = self.mode_weights(x)
        return 0.5 * float(np.sum(np.abs(ph[:-1] - self.true_weights)) + ph[-1])

    def fold(self, x) -> np.ndarray:
        x = np.asarray(x, np.float64)
        return self._fold(x) if self._fold is not None else x

    def sw2(self, x, key, n_sw: int = 2048, n_proj: int = 512) -> float:
        """SW2 of (up to) n_sw draws of x vs n_sw reference draws (a fresh random subset
        per key), both folded for the symmetric targets."""
        x = np.asarray(x, np.float64)
        x = x[np.all(np.isfinite(x), -1)]
        k_ref, k_x, k_proj = jax.random.split(key, 3)
        ri = np.asarray(jax.random.choice(k_ref, self.reference.shape[0], (n_sw,), replace=False))
        m = min(n_sw, x.shape[0])
        xi = np.asarray(jax.random.choice(k_x, x.shape[0], (m,), replace=False))
        return _sw2(jnp.asarray(self.fold(x[xi])), jnp.asarray(self.fold(self.reference[ri][:m])),
                    n_proj=n_proj, key=k_proj)


def _std_assign(centers, scale, radius=None):
    """Nearest center under per-coordinate standardization; draws beyond `radius`
    (and non-finite draws) get the off-mode label."""
    def f(x):
        z = (x[:, None, :] - centers[None]) / scale[None, None, :]
        d2 = np.sum(z * z, -1)
        lab = np.argmin(d2, -1)
        if radius is not None:
            dist = np.sqrt(np.take_along_axis(d2, lab[:, None], -1)[:, 0])
            lab = np.where(dist <= radius, lab, centers.shape[0])
        return np.where(np.all(np.isfinite(x), -1), lab, centers.shape[0])
    return f


def _nearest(x, centers, scale):
    """Nearest center index and its standardized L2 distance, per sample."""
    z = (x[:, None, :] - centers[None, :, :]) / scale[None, None, :]
    dist2 = np.sum(z * z, axis=-1)
    lab = np.argmin(dist2, axis=-1)
    return lab, np.sqrt(np.take_along_axis(dist2, lab[:, None], axis=-1)[:, 0])


# ---------------- skewnormal ----------------
def _skewnormal(results_dir=None):
    from run_skewnormal import build_target
    t = build_target()
    mus, sig, w = (np.asarray(a) for a in (t.mus, t.sigmas, t.weights))
    ref = np.asarray(t.sample(jax.random.PRNGKey(12345), 50000)[0])

    def assign(x):
        zz = (x[:, None, :] - mus[None]) / sig[None, :, None]
        return np.argmin(np.sum(zz * zz, -1), -1)
    return TargetEval("skewnormal", mus.shape[1], jax.jit(t.log_prob), ref, mus.shape[0], w, False,
                      assign, None, mus)


# ---------------- sur ----------------
_ASCEND_CACHE: dict = {}


def sur_ascend(log_prob_single, x, steps=SUR_ASCENT_STEPS, lr=SUR_ASCENT_LR, chunk=4096):
    """`steps` Adam steps of gradient ascent on log pi from every row of x (compiled once
    per log_prob_single; rows are processed in zero-padded chunks of a fixed size)."""
    import optax
    ck = (id(log_prob_single), steps, lr, chunk)
    if ck not in _ASCEND_CACHE:
        g = jax.vmap(jax.grad(log_prob_single))
        opt = optax.adam(lr)

        @jax.jit
        def run(x):
            def body(c, _):
                x, st = c
                u, st = opt.update(-g(x), st)
                return (optax.apply_updates(x, u), st), None
            (x, _), _ = jax.lax.scan(body, (x, opt.init(x)), None, length=steps)
            return x
        _ASCEND_CACHE[ck] = (run, log_prob_single)      # keep the function alive (id key)
    run = _ASCEND_CACHE[ck][0]
    out = []
    for i in range(0, x.shape[0], chunk):
        xc = x[i:i + chunk]
        pad = chunk - xc.shape[0]
        xp = np.concatenate([xc, np.repeat(xc[-1:], pad, 0)]) if pad else xc
        out.append(np.asarray(run(jnp.asarray(xp)))[:xc.shape[0]])
    return np.concatenate(out) if out else x


def _sur(results_dir=None):
    from run_sur import build_target
    t = build_target()
    lp_single = lambda x: t.log_prob(x[None])[0]
    ref = np.load(common.SUR_REFERENCE_PATH)["oracle_samples"]
    m = np.load(common.SUR_MODES_PATH)
    centers = m["centers"]

    def assign(x):
        ok = np.all(np.isfinite(x), -1)
        lab = np.full(x.shape[0], centers.shape[0])        # non-finite: off-mode
        xa = sur_ascend(lp_single, x[ok])
        lab[ok] = np.argmin(np.sum((xa[:, None] - centers[None]) ** 2, -1), -1)
        return lab
    return TargetEval("sur", ref.shape[1], jax.jit(t.log_prob), ref, centers.shape[0], m["ref_weights"],
                      False, assign, None, centers)


# ---------------- iohmm ----------------
def iohmm_reference_basins(centers, ref, n_iter: int = 3):
    """(scale, radius): `scale` is the per-coordinate within-mode std of the NUTS reference,
    pooled over all 6 modes (the pool is symmetrized across them), iterated a few times
    since the assignment itself depends on the scale (starting from unstandardized L2);
    `radius` is the largest standardized nearest-center distance over the pool, the
    in-basin cutoff."""
    scale = np.ones(ref.shape[1])
    for _ in range(n_iter):
        lab, _ = _nearest(ref, centers, scale)
        scale = (ref - centers[lab]).std(axis=0)
    _, dist = _nearest(ref, centers, scale)
    return scale, float(dist.max())


def _iohmm(results_dir=None):
    t = common.build_iohmm()
    rdir = results_dir or common.benchmark_dir("iohmm")
    centers = np.asarray(np.load(os.path.join(common.seed_dirs(rdir)[0], "samples.npz"))["mode_centers"])
    ref = np.load(common.IOHMM_REFERENCE_PATH)["draws"]
    scale, radius = iohmm_reference_basins(centers, ref)
    assign = _std_assign(centers, scale, radius)
    P = jnp.stack([jnp.array(p) for p in itertools.permutations(range(t.K))])
    permute = jax.jit(jax.vmap(lambda x, p: t.permute_components(x, p), in_axes=(None, 0)))

    def fold(x):
        """The label permutation of each draw closest (standardized) to mode 0's center."""
        out = np.empty_like(x)
        for i in range(0, x.shape[0], 2048):
            xp = np.asarray(permute(jnp.asarray(x[i:i + 2048]), P))            # (6, n, d)
            z = (xp - centers[0]) / scale
            best = np.argmin(np.sum(z * z, -1), 0)
            out[i:i + 2048] = xp[best, np.arange(xp.shape[1])]
        return out
    return TargetEval("iohmm", centers.shape[1], jax.jit(t.log_prob), ref, centers.shape[0],
                      np.full(centers.shape[0], 1.0 / centers.shape[0]), True, assign, fold, centers)


# ---------------- lda ----------------
def build_lda_with_reference():
    """The LDA target with its NUTS reference (Stan/ILR coordinates) mapped into the
    target's inference coordinates, as `target._reference`."""
    target = common.build_lda()
    ref_stan = common.lda_reference_stan()
    target._reference = jnp.asarray(np.concatenate([
        np.asarray(target.stan_to_inference(jnp.asarray(ref_stan[i:i + 2000], dtype=jnp.float64)))
        for i in range(0, len(ref_stan), 2000)
    ], axis=0), dtype=jnp.float64)
    return target


def lda_reference_basins(target):
    """(centers (2, d), scale (d,), radius): the two topic-label mode centers (reference
    within-basin mean and its `topic_swap` image), the pooled per-coordinate within-basin
    std, and the in-basin cutoff (largest standardized nearest-center distance over the
    reference). The pool is chain-major with chains 0-1 in one basin and 2-3 in the other."""
    ref = np.asarray(target._reference)
    half = ref.shape[0] // 2
    c0 = ref[:half].mean(0)
    centers = np.stack([c0, np.asarray(target.topic_swap(jnp.asarray(c0)))])
    scale = np.concatenate([ref[:half] - centers[0], ref[half:] - centers[1]]).std(0)
    _, dist = _nearest(ref, centers, scale)
    return centers, scale, float(dist.max())


def lda_fold(target, x, centers, scale):
    """Map every sample into the basin of `centers[0]` via `topic_swap`."""
    lab, _ = _nearest(np.asarray(x), centers, scale)
    swapped = np.asarray(target.topic_swap(jnp.asarray(x)))
    return np.where(lab[:, None] == 1, swapped, np.asarray(x))


def _lda(results_dir=None):
    t = build_lda_with_reference()
    centers, scale, radius = lda_reference_basins(t)
    ref = np.asarray(t._reference)
    return TargetEval("lda", centers.shape[1], jax.jit(t.log_prob), ref, 2, np.full(2, 0.5), True,
                      _std_assign(centers, scale, radius), lambda x: lda_fold(t, x, centers, scale), centers)
