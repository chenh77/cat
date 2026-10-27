# targets/constraints.py
"""Stan-compatible constraining transforms, hand-ported to plain JAX.

Used by `iohmm.py` and `lda.py` to reproduce the posteriordb/Stan targets'
unconstrained-space parameterization exactly, so `x` (the unconstrained vector
every method in this repo samples) means the same thing as it would to
`bridgestan`/Stan, without a per-call host round-trip to Stan. Each port was
verified against bridgestan.

Every `*_constrain` function returns `(x, log_abs_jacobian_det)` for a
single unconstrained vector `y`; log-Jacobian conventions match Stan's
`jacobian=True` `log_density` exactly (verified against bridgestan to
machine precision -- see each target module's validation cell/report).

Formulas transcribed directly from `stan-dev/math`'s
`stan/math/prim/constraint/{simplex,positive_ordered,lub}_constrain.hpp`
(current Stan, >= ~2.33): the simplex transform in particular changed from
older Stan's plain "stick-breaking logistic" to an isometric-log-ratio (ILR)
construction (`softmax(sum_to_zero_constrain(y))`) -- ported here as the
literal `sum_to_zero_constrain` recursion (implemented as a precomputed
linear map, since that transform is exactly linear) composed with a stable
softmax, not the older, no-longer-matching stick-breaking formula.
"""
from __future__ import annotations

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

Array = jnp.ndarray


# ---------------------------------------------------------------------
# simplex: y in R^{K-1} -> x in simplex(K), via ILR-style sum-to-zero + softmax
# ---------------------------------------------------------------------

def _sum_to_zero_constrain_np(y: np.ndarray) -> np.ndarray:
    """Direct transcription of `sum_to_zero_constrain.hpp`'s loop (used only
    to build the fixed linear map `A_K` below at target-construction time,
    not inside any jitted/traced path)."""
    N = y.shape[0]
    z = np.zeros(N + 1, dtype=np.float64)
    sum_w = 0.0
    for i in range(N, 0, -1):
        n = float(i)
        w = y[i - 1] / np.sqrt(n * (n + 1.0))
        sum_w += w
        z[i - 1] += sum_w
        z[i] -= w * n
    return z


def sum_to_zero_matrix(K: int) -> np.ndarray:
    """(K, K-1) matrix A such that `A @ y == sum_to_zero_constrain(y)` for
    y in R^{K-1} (exact: `sum_to_zero_constrain` is a linear map)."""
    N = K - 1
    A = np.zeros((K, N), dtype=np.float64)
    for j in range(N):
        e = np.zeros(N, dtype=np.float64)
        e[j] = 1.0
        A[:, j] = _sum_to_zero_constrain_np(e)
    return A


def make_simplex_constrain(K: int):
    """Returns `simplex_constrain(y) -> (x, logjac)` for fixed `K`, with the
    linear map precomputed once (host numpy) and closed over as a JAX
    constant -- see module docstring for the `softmax(sum_to_zero(y))`
    formula this implements, transcribed from `simplex_constrain.hpp`."""
    A = jnp.asarray(sum_to_zero_matrix(K))  # (K, K-1)
    logK = float(np.log(K))

    def simplex_constrain(y: Array):
        w = A @ y  # (K,), sums to (numerically) zero
        m = jnp.max(w)
        lse = m + jnp.log(jnp.sum(jnp.exp(w - m)))
        x = jnp.exp(w - lse)
        logjac = -K * lse + 0.5 * logK
        return x, logjac

    return simplex_constrain


# ---------------------------------------------------------------------
# positive_ordered: y in R^K -> x in R^K, 0 < x_1 < x_2 < ... < x_K
# ---------------------------------------------------------------------

def positive_ordered_constrain(y: Array):
    x = jnp.cumsum(jnp.exp(y))
    logjac = jnp.sum(y)
    return x, logjac


def positive_ordered_constrain_stable(y: Array):
    """Same map as `positive_ordered_constrain`, but also returns
    `log(x)` computed via a numerically stable cumulative log-sum-exp
    (`jax.lax.associative_scan(jnp.logaddexp, y)`) instead of `jnp.log(x)`.

    `x` itself (`cumsum(exp(y))`) can still overflow to `inf` for extreme
    `y` (unavoidable -- some callers genuinely need `x` in linear space,
    e.g. an `Exponential(rate=x)` log-density's `-rate*y_obs` term), but
    `log(x)` computed this way stays finite for any finite `y` (it only
    grows as fast as `max(y[:k+1])`, never exponentially), so an emission
    log-density built as `log_x - x*y_obs` degrades to the mathematically
    correct `-inf` (not `nan`) when `x` overflows, instead of the naive
    `jnp.log(x) - x*y_obs` hitting `inf - inf = nan`.

    Clipping `y` before `exp` instead would avoid the overflow but create an
    exactly flat, zero-gradient plateau beyond the clip, into which L-BFGS mode
    search can wander and stop at a spurious mode. This version has no flat
    region: it is the exact, unclipped map, computed more stably.
    """
    x = jnp.cumsum(jnp.exp(y))
    logjac = jnp.sum(y)
    log_x = jax.lax.associative_scan(jnp.logaddexp, y)
    return x, logjac, log_x


# ---------------------------------------------------------------------
# ordered: y in R^K -> x in R^K, x_1 < x_2 < ... < x_K (no positivity)
# ---------------------------------------------------------------------

def ordered_constrain(y: Array):
    """Transcribed from `ordered_constrain.hpp`: `x[0] = y[0]` (free real,
    no transform/Jacobian contribution), `x[k] = x[k-1] + exp(y[k])` for
    `k >= 1`. Same cumulative structure as `positive_ordered_constrain`
    minus the initial `exp` on `x[0]`, so unlike that transform `x` here is
    unbounded below (only the *gaps* between consecutive entries are forced
    positive)."""
    x = jnp.concatenate([y[:1], y[0] + jnp.cumsum(jnp.exp(y[1:]))])
    logjac = jnp.sum(y[1:])
    return x, logjac


def ordered_constrain_stable(y: Array):
    """Same map as `ordered_constrain`, but also returns `log(x[1:] -
    x[:-1])` (the log of each positive gap) via `jax.lax.associative_scan`
    cumulative log-sum-exp over `y[1:]`, mirroring
    `positive_ordered_constrain_stable`'s rationale: useful to any caller
    building a log-density term that needs those gaps' logs without first
    computing `x` and re-subtracting (which would re-derive the gap via
    cancellation instead of carrying it through directly)."""
    x = jnp.concatenate([y[:1], y[0] + jnp.cumsum(jnp.exp(y[1:]))])
    logjac = jnp.sum(y[1:])
    log_gaps = jax.lax.associative_scan(jnp.logaddexp, y[1:])
    return x, logjac, log_gaps


# ---------------------------------------------------------------------
# lower/upper-bounded scalar: y in R -> x in (lb, ub)
# ---------------------------------------------------------------------

def lub_constrain(y: Array, lb: float, ub: float):
    logjac = jnp.log(ub - lb) + jax.nn.log_sigmoid(y) + jax.nn.log_sigmoid(-y)
    x = lb + (ub - lb) * jax.nn.sigmoid(y)
    return x, logjac


# ---------------------------------------------------------------------
# positive scalar (lower=0 only): y in R -> x in (0, inf)
# ---------------------------------------------------------------------

def positive_constrain(y: Array):
    return jnp.exp(y), y
