# targets/common.py
"""Shared helpers used across the target-distribution modules.

Every target below implements the same lightweight interface:
  - `log_prob(x)`: (N,d) or (d,) -> (N,) unnormalized log-density
  - `dim`: int, dimension of the sample space
  - `sample(key, n)`: (key, n) -> (x, next_key), (N,d) i.i.d. draws
  - `score(x)` (optional): (N,d) -> (N,), grad_x log_prob; falls back to
    autodiff via `_score_from_logprob` when not analytically available.
"""
from __future__ import annotations

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import vmap
from jax.scipy.linalg import solve_triangular

Array = jnp.ndarray


def _as_2d(x: Array) -> Array:
    x = jnp.asarray(x)
    return x[None, :] if x.ndim == 1 else x


def _log_phi_std(z: Array) -> Array:
    # log N(0,1) density elementwise
    return -0.5 * (jnp.log(2.0 * jnp.pi) + z * z)


def _log_mvnorm_chol(x: Array, mean: Array, cov: Array) -> Array:
    """
    x: (N,d), mean: (d,), cov: (d,d) SPD
    returns: (N,)
    """
    x = _as_2d(x)
    mean = jnp.asarray(mean)
    cov = jnp.asarray(cov)
    d = mean.shape[0]

    L = jnp.linalg.cholesky(cov)  # (d,d), lower
    diff = x - mean[None, :]      # (N,d)

    # solve L y = diff^T  -> y shape (d,N)
    y = solve_triangular(L, diff.T, lower=True)    # (d,N)
    quad = jnp.sum(y * y, axis=0)                  # (N,)
    logdet = 2.0 * jnp.sum(jnp.log(jnp.diag(L)))
    return -0.5 * (d * jnp.log(2.0 * jnp.pi) + logdet + quad)


def _score_from_logprob(log_prob_fn):
    # returns score(x) = grad_x log_prob(x) with batching
    def _lp1(x1: Array) -> Array:
        # x1: (d,)
        return log_prob_fn(x1)[0]  # scalar
    g1 = jax.grad(_lp1)           # (d,) -> (d,)
    return vmap(g1)               # (N,d) -> (N,d)
