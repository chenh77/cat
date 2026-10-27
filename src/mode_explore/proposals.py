# proposals.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Protocol, Tuple, Dict, Type

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jr
from jax.scipy.linalg import solve_triangular

Array = jnp.ndarray
LogPi = Callable[[Array], Array]       # log pi(x) (beta=1), x: (d,) -> scalar ()
GradLogPi = Callable[[Array], Array]   # grad log pi(x), x: (d,) -> (d,)


# ============================================================
# Helpers
# ============================================================

def as1d_jax(x: Array) -> Array:
    x = jnp.asarray(x)
    if x.ndim != 1:
        raise ValueError(f"Expected 1D array, got shape {x.shape}")
    return x

def logpi_beta(logpi: LogPi, x: Array, beta: float) -> Array:
    """log pi_beta(x) where pi_beta ∝ pi^beta."""
    x = as1d_jax(x)
    return jnp.asarray(beta) * jnp.asarray(logpi(x))

def grad_beta(grad_logpi: GradLogPi, x: Array, beta: float) -> Array:
    """∇ log pi_beta(x) = beta * ∇ log pi(x)."""
    x = as1d_jax(x)
    return jnp.asarray(beta) * jnp.asarray(grad_logpi(x))


class Kernel(Protocol):
    def step(
        self,
        x: Array,
        logpi: LogPi,
        beta: float,
        key: jr.PRNGKey,
        grad_logpi: Optional[GradLogPi] = None,
    ) -> Tuple[Array, Array, jr.PRNGKey]:
        """Return (new_x, accepted_flag, new_key)."""


# ============================================================
# RWMH
# ============================================================

@dataclass
class RWMH:
    """Random-walk MH with proposal x' = x + step_size * N(0, I)."""
    step_size: float = 0.5

    def step(
        self,
        x: Array,
        logpi: LogPi,
        beta: float,
        key: jr.PRNGKey,
        grad_logpi: Optional[GradLogPi] = None,
    ) -> Tuple[Array, Array, jr.PRNGKey]:
        x = as1d_jax(x)
        key, k_prop, k_u = jr.split(key, 3)

        prop = x + float(self.step_size) * jr.normal(k_prop, shape=x.shape, dtype=x.dtype)

        loga = logpi_beta(logpi, prop, beta) - logpi_beta(logpi, x, beta)
        accept = jnp.log(jr.uniform(k_u, shape=(), dtype=x.dtype)) < loga
        new_x = jnp.where(accept, prop, x)
        return new_x, accept, key


# ============================================================
# MALA
# ============================================================

@dataclass
class MALA:
    """
    MALA:
      x' = x + (s^2/2) * ∇ log pi_beta(x) + s * N(0, I)
    with MH correction.
    """
    step_size: float = 0.2

    def step(
        self,
        x: Array,
        logpi: LogPi,
        beta: float,
        key: jr.PRNGKey,
        grad_logpi: Optional[GradLogPi] = None,
    ) -> Tuple[Array, Array, jr.PRNGKey]:
        if grad_logpi is None:
            raise ValueError("MALA requires grad_logpi.")

        x = as1d_jax(x)
        s = float(self.step_size)
        s2 = s * s

        key, k_prop, k_u = jr.split(key, 3)

        g = grad_beta(grad_logpi, x, beta)
        mean_fwd = x + 0.5 * s2 * g
        prop = mean_fwd + s * jr.normal(k_prop, shape=x.shape, dtype=x.dtype)

        lp_x = logpi_beta(logpi, x, beta)
        lp_p = logpi_beta(logpi, prop, beta)

        g_p = grad_beta(grad_logpi, prop, beta)
        mean_bwd = prop + 0.5 * s2 * g_p

        # log q(y | m) for N(m, s^2 I) up to constant
        def log_q(y: Array, m: Array) -> Array:
            diff = y - m
            return -0.5 * jnp.dot(diff, diff) / s2

        loga = (lp_p + log_q(x, mean_bwd)) - (lp_x + log_q(prop, mean_fwd))
        accept = jnp.log(jr.uniform(k_u, shape=(), dtype=x.dtype)) < loga
        new_x = jnp.where(accept, prop, x)
        return new_x, accept, key


# ============================================================
# Independence MH
# ============================================================

@dataclass
class IndependenceMH:
    """Independence MH with proposal N(mean, cov)."""
    mean: Array
    cov: Array
    jitter: float = 1e-10  # single-shot diagonal jitter (simple + works well in practice)

    def __post_init__(self) -> None:
        self.mean = as1d_jax(self.mean)
        self.cov = jnp.asarray(self.cov, dtype=self.mean.dtype)

        d = self.mean.size
        if self.cov.shape != (d, d):
            raise ValueError(f"cov must be {(d, d)}, got {self.cov.shape}")

        # one-shot jittered cholesky; if you truly need adaptive retries, we can add lax.while_loop later
        I = jnp.eye(d, dtype=self.mean.dtype)
        self.L = jnp.linalg.cholesky(0.5 * (self.cov + self.cov.T) + self.jitter * I)  # lower-tri

    def _log_q(self, x: Array) -> Array:
        # log N(x | mean, cov) up to constant = -0.5 * ||L^{-1}(x-mean)||^2
        diff = as1d_jax(x) - self.mean
        y = solve_triangular(self.L, diff, lower=True)  # (d,)
        return -0.5 * jnp.dot(y, y)

    def step(
        self,
        x: Array,
        logpi: LogPi,
        beta: float,
        key: jr.PRNGKey,
        grad_logpi: Optional[GradLogPi] = None,
    ) -> Tuple[Array, Array, jr.PRNGKey]:
        x = as1d_jax(x)
        key, k_prop, k_u = jr.split(key, 3)

        z = jr.normal(k_prop, shape=x.shape, dtype=x.dtype)
        prop = self.mean + self.L @ z

        loga = (logpi_beta(logpi, prop, beta) + self._log_q(x)) - (logpi_beta(logpi, x, beta) + self._log_q(prop))
        accept = jnp.log(jr.uniform(k_u, shape=(), dtype=x.dtype)) < loga
        new_x = jnp.where(accept, prop, x)
        return new_x, accept, key


# ============================================================
# ULA (unadjusted)
# ============================================================

@dataclass
class ULA:
    """
    Unadjusted Langevin Algorithm:

      x' = x + (s^2/2) * ∇ log pi_beta(x) + s * N(0, I)
    """
    step_size: float = 0.2

    def step(
        self,
        x: Array,
        logpi: LogPi,  # unused (kept for interface compatibility)
        beta: float,
        key: jr.PRNGKey,
        grad_logpi: Optional[GradLogPi] = None,
    ) -> Tuple[Array, Array, jr.PRNGKey]:
        if grad_logpi is None:
            raise ValueError("ULA requires grad_logpi.")

        x = as1d_jax(x)
        s = float(self.step_size)
        s2 = s * s

        key, k_prop = jr.split(key, 2)
        g = grad_beta(grad_logpi, x, beta)
        prop = x + 0.5 * s2 * g + s * jr.normal(k_prop, shape=x.shape, dtype=x.dtype)
        accept = jnp.array(True)
        return prop, accept, key


KERNEL_REGISTRY: Dict[str, Type[Kernel]] = {
    "rwmh": RWMH,
    "mala": MALA,
    "indep": IndependenceMH,
    "ula": ULA,
}