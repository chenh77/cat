# density.py
"""Distributions and densities built on a discovered mode set `I =
{(mu_j, Sigma_j, a_j)}` (ModeExplore, `explore.py`):

  - `StandardNormal`  — the base/exploration distribution gamma.
  - `ModeGaussian`    — a single mode's Laplace anchor b_j = N(mu_j, Sigma_j).
  - `Mode`, `modes_to_arrays` — re-exported from `explore` / host-side
    list[Mode] -> stacked-array conversion.
  - `_log_mvnorm_batch_modes` — batched log N(x | mu_j, Sigma_j) over all
    modes at once.
  - `log_q_hat_batch`/`basin_assign` — the mode-adaptive Laplace anchor
    b(x) = sum_j a_j b_j(x) and its beta-free hard basin assignment
    A(x) = argmax_j w_j phi_j(x).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple
import math

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jr

from ..utils.numeric import as2d_jax, sym_jax, chol_jitter_jax
from .explore import Mode

Array = jax.Array


# ============================================================
# Base distributions
# ============================================================
@dataclass
class StandardNormal:
    d: int

    def sample(self, key: Array, n: int, dtype=jnp.float64) -> Array:
        return jr.normal(key, (int(n), int(self.d)), dtype=dtype)

    def log_prob(self, z: Array) -> Array:
        z = as2d_jax(z)
        d = z.shape[-1]
        return -0.5 * (
            jnp.sum(z * z, axis=-1)
            + jnp.asarray(d, dtype=z.dtype) * jnp.log(jnp.asarray(2.0 * math.pi, dtype=z.dtype))
        )


@dataclass
class ModeGaussian:
    mu: Array
    Sigma: Array
    jitter0: float = 1e-6
    max_tries: int = 10

    def __post_init__(self):
        self._L = chol_jitter_jax(sym_jax(self.Sigma), jitter0=self.jitter0, max_tries=self.max_tries)

    def chol(self) -> Array:
        return self._L

    def log_prob(self, x: Array) -> Array:
        x = as2d_jax(x)
        d = x.shape[-1]
        L = self.chol()
        xc = x - self.mu[None, :]
        y = jax.scipy.linalg.solve_triangular(L, xc.T, lower=True).T
        quad = jnp.sum(y * y, axis=-1)
        ld = 2.0 * jnp.sum(jnp.log(jnp.diagonal(L)))
        return -0.5 * (
            quad + ld + jnp.asarray(d, dtype=x.dtype) * jnp.log(jnp.asarray(2.0 * math.pi, dtype=x.dtype))
        )

    def sample(self, key: Array, n: int, dtype=jnp.float64) -> Array:
        L = self.chol().astype(dtype)
        eps = jr.normal(key, (int(n), self.mu.shape[0]), dtype=dtype)
        return self.mu[None, :].astype(dtype) + eps @ L.T


# ============================================================
# Helpers: convert list[Mode] -> stacked arrays (host-side)
# ============================================================
def modes_to_arrays(
    modes: Sequence[Mode],
    dtype=jnp.float64,
    *,
    anchor_means: bool = False,
) -> Tuple[Array, Array, Array]:
    """
    Returns:
      mu:    (m,d), stationary points by default; Gaussian anchor means when
             ``anchor_means=True``
      Sigma: (m,d,d)
      logw:  (m,)
    Note: this is intended as a non-jitted convenience wrapper.
    """
    if len(modes) == 0:
        raise ValueError("modes must be non-empty.")

    mu = jnp.stack(
        [
            jnp.asarray(
                m.anchor_mu
                if anchor_means and getattr(m, "anchor_mu", None) is not None
                else m.mu,
                dtype=dtype,
            )
            for m in modes
        ],
        axis=0,
    )
    Sigma = jnp.stack([jnp.asarray(m.Sigma, dtype=dtype) for m in modes], axis=0)
    w = jnp.asarray([float(m.w) for m in modes], dtype=dtype)
    if jnp.any(w <= 0):
        raise ValueError("All mode weights must be positive.")
    logw = jnp.log(w)
    return mu, Sigma, logw


# ============================================================
# JAX MVN utilities (batched over modes)
# ============================================================
def _log_mvnorm_batch_modes(
    x: Array,      # (n,d)
    mu: Array,     # (m,d)
    Sigma: Array,  # (m,d,d)
    jitter0: float = 1e-10,
    max_tries: int = 12,
) -> Array:
    x = as2d_jax(x)
    mu = jnp.asarray(mu)
    Sigma = sym_jax(jnp.asarray(Sigma))

    n, d = x.shape
    m = mu.shape[0]

    if mu.shape != (m, d):
        raise ValueError(f"mu must have shape (m,d)=({m},{d}), got {mu.shape}")
    if Sigma.shape != (m, d, d):
        raise ValueError(f"Sigma must have shape (m,d,d)=({m},{d},{d}), got {Sigma.shape}")

    L = chol_jitter_jax(Sigma, jitter0=float(jitter0), max_tries=int(max_tries))  # (m,d,d)

    diff_mnd = (x[:, None, :] - mu[None, :, :]).transpose(1, 0, 2)  # (m,n,d)

    def solve_mode(Lj, diff_n_d):
        return jax.scipy.linalg.solve_triangular(Lj, diff_n_d.T, lower=True).T  # (n,d)

    y_mnd = jax.vmap(solve_mode, in_axes=(0, 0), out_axes=0)(L, diff_mnd)  # (m,n,d)
    y = y_mnd.transpose(1, 0, 2)  # (n,m,d)

    quad = jnp.sum(y * y, axis=-1)  # (n,m)
    ld = 2.0 * jnp.sum(jnp.log(jnp.diagonal(L, axis1=-2, axis2=-1)), axis=-1)  # (m,)

    const = float(d) * math.log(2.0 * math.pi)
    return -0.5 * (const + ld[None, :] + quad)


# ============================================================
# Mode-adaptive Laplace anchor: `log_q_hat_batch` is the mixture
# b(x) = sum_j a_j b_j(x), and `basin_assign` a beta-free hard basin
# assignment A(x) = argmax_j w_j phi_j(x).
# ============================================================
def log_q_hat_batch(
    x: Array,
    mu: Array,
    Sigma: Array,
    logw: Array,
    *,
    anchor: str = "mixture",
    jitter0: float = 1e-10,
    max_tries: int = 12,
) -> Array:
    """log q_hat(x) for the mode-adaptive Laplace anchor, batched over x: (n,d) -> (n,).

    anchor="mixture": logsumexp_j (logw_j + log phi_j(x))         (b(x) = sum_j a_j b_j(x))
    anchor="max":     max_j     (logw_j + log phi_j(x))
    """
    x = as2d_jax(x)
    logphi = _log_mvnorm_batch_modes(x, mu, Sigma, jitter0=jitter0, max_tries=max_tries)  # (n,m)
    scores = logphi + jnp.asarray(logw)[None, :]  # (n,m)

    if anchor == "mixture":
        return jax.scipy.special.logsumexp(scores, axis=-1)
    if anchor == "max":
        return jnp.max(scores, axis=-1)
    raise ValueError(f"Unknown anchor={anchor!r}. Use 'mixture' or 'max'.")


def basin_assign(
    x: Array,
    mu: Array,
    Sigma: Array,
    logw: Array,
    *,
    jitter0: float = 1e-10,
    max_tries: int = 12,
) -> Array:
    """Hard basin assignment A(x) = argmax_j w_j phi_j(x), independent of beta."""
    x = as2d_jax(x)
    logphi = _log_mvnorm_batch_modes(x, mu, Sigma, jitter0=jitter0, max_tries=max_tries)  # (n,m)
    scores = logphi + jnp.asarray(logw)[None, :]
    return jnp.argmax(scores, axis=-1)

