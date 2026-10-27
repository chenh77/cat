# targets/sur.py
from __future__ import annotations

from dataclasses import dataclass

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from .common import Array, _as_2d


@dataclass
class SURModel:
    """
    Seemingly Unrelated Regression (SUR) model — Zellner (1962).

    Model:
        y_m = X_m @ beta_m + eps_m,   m = 1,...,M
        (eps_1,...,eps_M) ~ N(0, Sigma ⊗ I_N)

    Target: profile posterior with flat prior on beta and Sigma profiled out.

        log p(beta | Y) ∝ -(N/2) log det(Sigma_hat(beta))
        Sigma_hat_{ml}(beta) = (1/N) * e_m^T e_l,  e_m = y_m - X_m @ beta_m

    Parameter: theta = concat(beta_1,...,beta_M) in R^{sum_m K_m}

    Reference: Grunfeld (1958) investment data / Zellner (1962) / ALPS paper sec 7.2.
    """
    Y: Array      # (N, M)  — response matrix, each column is one equation
    X_list: list  # list of M arrays, each (N, K_m) — design matrices

    def __post_init__(self):
        self.Y = jnp.asarray(self.Y, dtype=jnp.float64)
        if self.Y.ndim != 2:
            raise ValueError(f"Y must be (N, M), got {self.Y.shape}")
        self.N, self.M = int(self.Y.shape[0]), int(self.Y.shape[1])
        self.X_list = [jnp.asarray(X, dtype=jnp.float64) for X in self.X_list]
        if len(self.X_list) != self.M:
            raise ValueError(f"len(X_list)={len(self.X_list)} != M={self.M}")
        self.K_list = [int(X.shape[1]) for X in self.X_list]
        self.K_total = int(sum(self.K_list))
        # Cumulative split boundaries (Python ints — trace-time constants)
        ends, acc = [], 0
        for k in self.K_list:
            acc += k
            ends.append(acc)
        self._ends = ends

    @property
    def dim(self) -> int:
        return self.K_total

    def log_prob(self, theta: Array) -> Array:
        """
        theta : (N_batch, K_total) or (K_total,)
        returns: (N_batch,) profile log-posterior values
        """
        theta = _as_2d(theta)        # (N_batch, K_total)

        # Build residual matrix: list of (N_batch, N_obs), one per equation
        resid_list = []
        start = 0
        for m, end in enumerate(self._ends):
            beta_m   = theta[:, start:end]              # (N_batch, K_m)
            fitted_m = beta_m @ self.X_list[m].T        # (N_batch, N_obs)
            resid_m  = self.Y[:, m][None, :] - fitted_m # (N_batch, N_obs)
            resid_list.append(resid_m)
            start = end

        E = jnp.stack(resid_list, axis=-1)              # (N_batch, N_obs, M)

        # Sigma_hat = E^T E / N  →  (N_batch, M, M)
        Sigma_hat = jnp.einsum("bni,bnj->bij", E, E) / self.N
        # Tiny jitter for numerical stability
        Sigma_hat = Sigma_hat + 1e-10 * jnp.eye(self.M, dtype=jnp.float64)

        _, logdet = jnp.linalg.slogdet(Sigma_hat)       # (N_batch,)

        # Profile log-posterior  (flat prior on beta, Sigma profiled out)
        lp = (
            -(self.N / 2.0) * logdet
            - (self.N * self.M / 2.0) * jnp.log(2.0 * jnp.pi)
            - self.N * self.M / 2.0
        )
        return lp
