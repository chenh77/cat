# utils/eval.py
# ------------------------------------------------------------
# Dimension-agnostic, batched metrics for high-d targets (JAX-only).
# Public API matches the original idea: evaluate_metrics(...)
#
# Notes / conventions:
# - `target.log_prob(x)` should accept x with shape (N,d) and return (N,).
# - If `target.score(x)` exists, it should return (N,d). Otherwise we compute
#   score via autodiff of log_prob (batched).
# - If `target.sample(key, n)` exists, we use it for KL(p||q) and SW2.
# - `q_log_prob` if provided must accept x (N,d) and return (N,).
#
# KSD is O(m^2) for m=ksd_samples (default 2048). For m up to a couple thousand
# this is fine on GPU/TPU; on CPU you may want to reduce ksd_samples.
# ------------------------------------------------------------

from __future__ import annotations

from typing import Callable, Optional, Dict, Any

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jr


__all__ = ["evaluate_metrics", "ess_from_logw"]


Array = jax.Array
LogProbFn = Callable[[Array], Array]  # (N,d)->(N,)


# ---------------- importance-sampling diagnostics ----------------

def _normalize_logw(logw: Array) -> Array:
    logw = logw - jnp.max(logw)
    w = jnp.exp(logw)
    return w / jnp.sum(w)


def ess_from_logw(logw: Array) -> Array:
    """Effective sample size 1/sum(w^2) from unnormalized log importance
    weights (self-normalized)."""
    w = _normalize_logw(logw)
    return 1.0 / jnp.sum(w * w)


# ---------------- batching utilities ----------------

def _batched_apply(fn: LogProbFn, x: Array, batch: int = 8192) -> Array:
    """
    Apply fn to x in batches along axis 0; returns (N,).
    """
    x = jnp.asarray(x)
    n = int(x.shape[0])
    outs = []
    for i in range(0, n, int(batch)):
        outs.append(jnp.asarray(fn(x[i:i + batch])).reshape(-1))
    return jnp.concatenate(outs, axis=0)


def _subsample_rows(key: Array, x: Array, m: Optional[int]) -> tuple[Array, Array]:
    """
    Subsample up to m rows from x without replacement.
    Returns (x_sub, key_next).
    """
    if m is None or m <= 0:
        return x, key
    n = int(x.shape[0])
    if m >= n:
        return x, key
    key, ksub = jr.split(key, 2)
    idx = jr.choice(ksub, n, shape=(int(m),), replace=False)
    return x[idx], key


def _pairwise_sqdist(x: Array, y: Array) -> Array:
    """
    Pairwise squared distances between rows: returns (Nx, Ny).
    Uses the standard trick: ||x||^2 + ||y||^2 - 2 x y^T.
    """
    x = jnp.asarray(x)
    y = jnp.asarray(y)
    x2 = jnp.sum(x * x, axis=1, keepdims=True)           # (Nx,1)
    y2 = jnp.sum(y * y, axis=1, keepdims=True).T         # (1,Ny)
    d2 = jnp.maximum(0.0, x2 + y2 - 2.0 * (x @ y.T))
    return d2


# ---------------- core helpers ----------------

def _get_score(target: Any, x: Array) -> Array:
    """
    Return score ∇_x log p(x); use target.score if present, else autodiff.
    Expects x: (N,d), returns (N,d).
    """
    if hasattr(target, "score"):
        return jnp.asarray(target.score(x), dtype=x.dtype)

    # autodiff: score(x_i) = grad logp(x_i)
    # We assume target.log_prob is batched; use vmap over single-sample.
    def logp_one(x1):
        return jnp.asarray(target.log_prob(x1[None, :])).reshape(())[0] if False else jnp.asarray(target.log_prob(x1[None, :])).reshape(())

    # The above line is a bit awkward: easiest is define:
    def logp_single(z):
        return jnp.asarray(target.log_prob(z[None, :]))[0]

    return jax.vmap(jax.grad(logp_single))(x)


def _maybe_sample_p(key: Array, target: Any, n: int) -> tuple[Optional[Array], Array]:
    """
    Return (samples_from_p_or_None, key_next).
    Handles targets whose .sample returns either x or (x, key).
    """
    if hasattr(target, "sample"):
        try:
            key, ks = jr.split(key, 2)
            result = target.sample(ks, int(n))
            # Support both `x` and `(x, key)` return conventions
            x = result[0] if isinstance(result, tuple) else result
            return jnp.asarray(x), key
        except Exception:
            return None, key
    return None, key


# ---------------- metrics ----------------

def _kl_from_terms(terms: Array, trim_frac: float = 0.01) -> Dict[str, Any]:
    """
    Summarize a KL Monte-Carlo estimator's per-sample log-ratio terms.

    A plain `mean(terms)` is reported alongside diagnostics because a single
    non-finite or extreme-magnitude term (e.g. a sample landing where q
    underflows to ~0 density) can make that mean an arbitrary, even negative,
    number even though the true KL is >= 0 -- silently trusting it hides
    exactly that failure mode. `mean_finite`/`median_finite` drop non-finite
    terms so the estimator degrades to "noisy but sane" instead of "silently
    wrong" when they occur; `n_nonfinite` and `max_abs_finite` say whether
    that happened and how extreme the surviving terms are.

    `mean_trimmed` goes further: even among *finite* terms, a handful of
    large-magnitude ones (e.g. a barely-finite underflowed density, not
    literally inf/nan) can still dominate the mean. It drops the top/bottom
    `trim_frac` of finite terms by value before averaging, so it is robust to
    that case too -- `isfinite` filtering alone is not.

    `suspect_negative` flags when `mean_trimmed` is negative by much more
    than its own Monte-Carlo noise (`se_finite`) would explain. Since true
    KL >= 0, a mean this reliably negative is not sampling noise: it means
    the underlying q density (or its log-det/normalization) is
    inconsistent, and the reported number should not be trusted at face
    value -- see e.g. a diverged transport whose density is mis-scaled
    rather than merely noisy.
    """
    terms = jnp.ravel(terms)
    n = int(terms.shape[0])
    finite_mask = jnp.isfinite(terms)
    n_finite = int(jnp.sum(finite_mask))
    mean_raw = float(jnp.mean(terms))

    if n_finite == 0:
        return {
            "mean": mean_raw,
            "mean_finite": None,
            "mean_trimmed": None,
            "median_finite": None,
            "se_finite": None,
            "n_nonfinite": n,
            "max_abs_finite": None,
            "suspect_negative": None,
        }

    finite_terms = jnp.where(finite_mask, terms, 0.0)
    sum_finite = jnp.sum(finite_terms)
    mean_finite = float(sum_finite / n_finite)
    # note: median/std over only the finite entries, via boolean-selected values
    finite_vals = terms[finite_mask]
    median_finite = float(jnp.median(finite_vals))
    se_finite = float(jnp.std(finite_vals) / jnp.sqrt(float(n_finite))) if n_finite > 1 else 0.0
    max_abs_finite = float(jnp.max(jnp.abs(finite_vals)))

    sorted_vals = jnp.sort(finite_vals)
    k_trim = int(trim_frac * n_finite)
    if n_finite - 2 * k_trim > 0:
        trimmed_vals = sorted_vals[k_trim: n_finite - k_trim]
    else:
        trimmed_vals = sorted_vals
    mean_trimmed = float(jnp.mean(trimmed_vals))

    # "Significantly negative" = more than a few SEs below 0, i.e. not
    # explainable by this estimator's own Monte-Carlo noise.
    suspect_negative = bool(mean_trimmed < -3.0 * se_finite - 1e-8)

    return {
        "mean": mean_raw,
        "mean_finite": mean_finite,
        "mean_trimmed": mean_trimmed,
        "median_finite": median_finite,
        "se_finite": se_finite,
        "n_nonfinite": n - n_finite,
        "max_abs_finite": max_abs_finite,
        "suspect_negative": suspect_negative,
    }


def _kl_reverse_q_p(x_q: Array, target: Any, q_log_prob: Optional[LogProbFn]) -> Optional[Dict[str, Any]]:
    """KL(q||p) = E_q[log q - log p]."""
    if q_log_prob is None:
        return None
    lq = _batched_apply(q_log_prob, x_q)
    lp = _batched_apply(target.log_prob, x_q)
    return _kl_from_terms(lq - lp)


def _kl_forward_p_q(x_p: Optional[Array], target: Any, q_log_prob: Optional[LogProbFn]) -> Optional[Dict[str, Any]]:
    """KL(p||q) = E_p[log p - log q]."""
    if x_p is None or q_log_prob is None:
        return None
    lp = _batched_apply(target.log_prob, x_p)
    lq = _batched_apply(q_log_prob, x_p)
    return _kl_from_terms(lp - lq)


def _ksd_rbf_unbiased(x: Array, target: Any, median_heuristic: bool = True) -> float:
    """
    Kernel Stein Discrepancy (RBF), unbiased off-diagonal estimator.

    KSD^2 = 1/(n(n-1)) sum_{i≠j} u(x_i, x_j).
    """
    x = jnp.asarray(x)
    n, d = x.shape
    s = _get_score(target, x)  # (n,d)

    D2 = _pairwise_sqdist(x, x)  # (n,n)

    if median_heuristic:
        med2 = jnp.median(jnp.ravel(D2))
        h2 = jnp.where(med2 > 0, med2 / jnp.log(float(n) + 1.0), jnp.asarray(1e-3, dtype=x.dtype))
    else:
        h2 = jnp.asarray(1.0, dtype=x.dtype)

    K = jnp.exp(-D2 / (2.0 * h2))  # (n,n)

    # s_i^T s_j * k
    sKs = (s @ s.T) * K

    # (x_i · s_j), (s_i · x_j)
    xsT = x @ s.T
    sxT = s @ x.T
    diag_xs = jnp.sum(x * s, axis=1)  # (n,)

    # Mixed terms: s_i^T ∇_y k + (∇_x k)^T s_j
    # matches your torch expression:
    term_mixed = (K / h2) * ((sxT - diag_xs[:, None]) - (xsT - diag_xs[None, :]))

    # Laplacian term: Δ_x k = (d/h2 - ||x-y||^2/h2^2) k
    trace_term = K * ((float(d) / h2) - (D2 / (h2 * h2)))

    U = sKs + term_mixed + trace_term

    # unbiased off-diagonal average
    sum_off = jnp.sum(U) - jnp.sum(jnp.diag(U))
    ksd2 = sum_off / (float(n) * float(n - 1))
    ksd2 = jnp.maximum(ksd2, 0.0)
    return float(jnp.sqrt(ksd2))


def _sw2(x: Array, y: Array, n_proj: int = 128, key: Optional[Array] = None) -> float:
    """
    Sliced 2-Wasserstein via random Gaussian directions (normalized), quantile matching.
    """
    x = jnp.asarray(x)
    y = jnp.asarray(y)
    d = int(x.shape[1])

    if key is None:
        key = jr.PRNGKey(0)
    key, kg = jr.split(key, 2)

    g = jr.normal(kg, (int(n_proj), d), dtype=x.dtype)
    g = g / (jnp.linalg.norm(g, axis=1, keepdims=True) + 1e-12)

    xg = jnp.sort(x @ g.T, axis=0)
    yg = jnp.sort(y @ g.T, axis=0)

    L = min(int(xg.shape[0]), int(yg.shape[0]))
    w2 = jnp.mean((xg[:L] - yg[:L]) ** 2, axis=0)  # per projection
    return float(jnp.sqrt(jnp.mean(w2)))


def _mode_weight_error(samples_q: Array, target: Any) -> Dict[str, Any]:
    """
    If target exposes mus (K,d), sigmas (K,) and weights (K,),
    assign each sample to nearest component (standardized L2) and compare
    the resulting empirical mode weights phat against the target's weights.

    Returns {} if target doesn't have the attributes, else:
      - ModeTV: 0.5 * sum_k |phat_k - weight_k|, the total-variation
        distance between the empirical and true mode-weight vectors
      - ModeWeights: phat, the (K,) empirical mode-weight vector itself
      - ModeWeightMin / ModeWeightMax: min/max over phat -- flags starved
        (near-zero) or over-represented modes that TV alone can hide.
    """
    if not (hasattr(target, "mus") and hasattr(target, "sigmas") and hasattr(target, "weights")):
        return {}

    mus = jnp.asarray(target.mus, dtype=samples_q.dtype)          # (K,d)
    sigmas = jnp.asarray(target.sigmas, dtype=samples_q.dtype)    # (K,)
    weights = jnp.asarray(target.weights, dtype=samples_q.dtype)  # (K,)

    x = jnp.asarray(samples_q)
    z = (x[:, None, :] - mus[None, :, :]) / sigmas[None, :, None]  # (N,K,d)
    dist2 = jnp.sum(z * z, axis=-1)                                # (N,K)
    a = jnp.argmin(dist2, axis=-1)                                 # (N,)

    counts = jnp.bincount(a, length=mus.shape[0]).astype(weights.dtype)
    phat = counts / jnp.sum(counts)
    tv = 0.5 * jnp.sum(jnp.abs(phat - weights))
    return {
        "ModeTV": float(tv),
        "ModeWeights": [float(v) for v in phat],
        "ModeWeightMin": float(jnp.min(phat)),
        "ModeWeightMax": float(jnp.max(phat)),
    }


# ---------------- public API ----------------

def evaluate_metrics(
    *,
    key: Array,
    target: Any,
    samples_q: Array,                 # (N,d) samples from q
    q_log_prob: Optional[LogProbFn] = None,
    n_p_samples: int = 2048,
    sw_projections: int = 128,
    ksd_samples: int = 2048,
    sw_samples: int = 4096,
    median_heuristic: bool = True,
    assume_zero_mean_if_missing: bool = True,
) -> tuple[Dict[str, Any], Array]:
    """
    Compute:
      - KL(q||p) if q_log_prob provided
      - KL(p||q) if target.sample exists and q_log_prob provided
      - KSD (RBF; subsampled)
      - SW2 (subsampled; requires target.sample)
      - MeanDelta and CovDelta (if target exposes true_mean/true_cov; else MeanDelta vs 0 if enabled)
      - Optional ModeTV/ModeWeights/ModeWeightMin/ModeWeightMax if target
        exposes mus/sigmas/weights

    Returns: (metrics_dict, key_next)
    """
    x_q = jnp.asarray(samples_q)

    # KL(q||p)
    kl_q_p = _kl_reverse_q_p(x_q, target, q_log_prob)

    # Draw from p if possible
    x_p, key = _maybe_sample_p(key, target, int(n_p_samples))

    # KL(p||q)
    kl_p_q = _kl_forward_p_q(x_p, target, q_log_prob) if x_p is not None else None

    # KSD (subsample)
    try:
        x_ksd, key = _subsample_rows(key, x_q, min(int(ksd_samples), int(x_q.shape[0])))
        ksd = _ksd_rbf_unbiased(x_ksd, target, median_heuristic=median_heuristic)
    except Exception as e:
        print(f"[warn] KSD failed: {e}")
        ksd = None

    # SW2 (subsample) if we have p-samples
    if x_p is not None:
        x_sw_q, key = _subsample_rows(key, x_q, min(int(sw_samples), int(x_q.shape[0])))
        x_sw_p, key = _subsample_rows(key, x_p, min(int(sw_samples), int(x_p.shape[0])))
        key, ksw = jr.split(key, 2)
        sw2 = _sw2(x_sw_q, x_sw_p, n_proj=int(sw_projections), key=ksw)
    else:
        sw2 = None

    # Mean & covariance diagnostics
    mean_q = jnp.mean(x_q, axis=0)
    xc = x_q - mean_q[None, :]
    den = max(1, int(x_q.shape[0]) - 1)
    cov_q = (xc.T @ xc) / float(den)

    mean_ref = None
    cov_ref = None

    if hasattr(target, "true_mean"):
        try:
            mean_ref = jnp.asarray(target.true_mean(), dtype=x_q.dtype)
        except Exception:
            mean_ref = None

    if hasattr(target, "true_cov"):
        try:
            cov_ref = jnp.asarray(target.true_cov(), dtype=x_q.dtype)
        except Exception:
            cov_ref = None

    if mean_ref is None and assume_zero_mean_if_missing:
        mean_ref = jnp.zeros_like(mean_q)

    mean_delta = float(jnp.linalg.norm(mean_q - mean_ref)) if mean_ref is not None else None
    cov_delta = float(jnp.linalg.norm(cov_q - cov_ref, ord="fro")) if cov_ref is not None else None

    def _flatten_kl(name: str, kl: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        # "<name>" keeps the naive full-sample mean (comparable to the old
        # scalar API); "_robust"/"_trimmed"/"_se"/"_nbad" expose whether that
        # mean is trustworthy or being driven by a handful of non-finite/
        # extreme per-sample terms (see `_kl_from_terms`). "_suspect" flags
        # a mean_trimmed that is negative well beyond its own MC noise --
        # since true KL >= 0, that means the estimate is unreliable (a
        # mis-scaled q), not just noisy, and should not be read at face value.
        if kl is None:
            return {name: None}
        return {
            name: kl["mean"],
            f"{name}_robust": kl["mean_finite"],
            f"{name}_trimmed": kl["mean_trimmed"],
            f"{name}_se": kl["se_finite"],
            f"{name}_nbad": kl["n_nonfinite"],
            f"{name}_suspect": kl["suspect_negative"],
        }

    out: Dict[str, Any] = {
        **_flatten_kl("KL(q||p)", kl_q_p),
        **_flatten_kl("KL(p||q)", kl_p_q),
        "KSD": ksd,
        "SW2": sw2,
        "MeanDelta": mean_delta,
        "CovDelta": cov_delta,
    }

    # Optional mixture coverage metric
    try:
        x_mode, key = _subsample_rows(key, x_q, min(20000, int(x_q.shape[0])))
        out.update(_mode_weight_error(x_mode, target))
    except Exception as e:
        print(f"[warn] mode-weight metrics failed: {e}")

    return out, key