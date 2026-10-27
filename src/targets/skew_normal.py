# targets/skew_normal.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jr
from jax.scipy.special import log_ndtr, logsumexp

from .common import Array, _as_2d, _log_phi_std, _score_from_logprob


def build_two_mode(*, d: int = 5, sep: float = 10.0, alpha: float = 4.0, w2: float = 0.2,
                   dtype=jnp.float64) -> Tuple[Array, Array, Array, Array]:
    """The two-mode skew-normal mixture of the experiments: mode 1 is N(-c 1, I), mode 2 a
    product of iid skew-normals at +c 1 with skewness `alpha` (tail pointing away from
    mode 1), c = (sep / 2) / sqrt(d), weights (1 - w2, w2). The skewed mode's peak
    curvature understates its mass, so the Laplace weights are far from the true ones.
    Returns `(mus, sigmas, weights, alphas)` for `SkewNormalMixture`."""
    c = 0.5 * sep / np.sqrt(d)
    mus = jnp.stack([jnp.full((d,), -c, dtype), jnp.full((d,), c, dtype)])
    return (mus, jnp.ones((2,), dtype), jnp.asarray([1.0 - w2, w2], dtype),
            jnp.asarray([0.0, alpha], dtype))


@dataclass
class SkewNormalMixture:
    """Skew-normal mixture with component-specific skewness alpha_k."""
    mus: Array                 # (K,d)
    sigmas: Array              # (K,)
    weights: Optional[Array] = None
    alpha: Union[float, Array] = 10.0   # scalar or (K,)

    def __post_init__(self):
        mus = jnp.asarray(self.mus, dtype=jnp.float64)
        if mus.ndim != 2:
            raise ValueError(f"mus must be (K,d), got {mus.shape}")
        self.mus = mus
        K, d = mus.shape
        self.K = int(K)
        self.d = int(d)

        sigmas = jnp.asarray(self.sigmas, dtype=jnp.float64).reshape(-1)
        if sigmas.shape[0] != self.K:
            raise ValueError(f"sigmas must have length K={self.K}, got {sigmas.shape[0]}")
        if jnp.any(sigmas <= 0):
            raise ValueError("sigmas must be strictly positive")
        self.sigmas = sigmas

        if self.weights is None:
            self.weights = jnp.ones((self.K,), dtype=jnp.float64) / self.K
        else:
            w = jnp.asarray(self.weights, dtype=jnp.float64).reshape(-1)
            if w.shape[0] != self.K:
                raise ValueError(f"weights must have length K={self.K}, got {w.shape[0]}")
            if jnp.any(w <= 0):
                raise ValueError("weights must be strictly positive")
            self.weights = w / jnp.sum(w)

        a = jnp.asarray(self.alpha, dtype=jnp.float64).reshape(-1)
        if a.shape[0] == 1:
            a = jnp.repeat(a, self.K)
        if a.shape[0] != self.K:
            raise ValueError(f"alpha must be scalar or length K={self.K}, got {a.shape[0]}")
        self.alphas = a

    @property
    def dim(self) -> int:
        return self.d

    def log_prob(self, x: Array) -> Array:
        """
        x: (N,d) or (d,) -> (N,)
        pi(x) ∝ Σ_k w_k Π_j (2/σ_k) φ(z) Φ(alpha_k z)
        with z=(x-mu_k)/sigma_k
        """
        x = _as_2d(x)
        if x.shape[1] != self.d:
            raise ValueError(f"x must have last dim {self.d}, got {x.shape}")

        # Broadcast to (N,K,d)
        x_ = x[:, None, :]
        mu = self.mus[None, :, :]
        sigma = self.sigmas[None, :, None]
        alpha = self.alphas[None, :, None]

        z = (x_ - mu) / sigma
        log_phi = _log_phi_std(z)
        log_Phi = log_ndtr(alpha * z)  # stable log Φ

        # sum over d, then include -d log sigma
        comp_lp = jnp.sum(log_phi + log_Phi, axis=-1) - self.d * jnp.log(sigma.squeeze(-1))  # (N,K)
        wlog = jnp.log(self.weights)[None, :]  # (1,K)
        return logsumexp(comp_lp + wlog, axis=-1)

    def score(self, x: Array) -> Array:
        return _score_from_logprob(self.log_prob)(_as_2d(x))

    def sample(self, key: jr.PRNGKey, n: int) -> Tuple[Array, jr.PRNGKey]:
        """
        Per-component skew-normal sampling (Azzalini form):
        If alpha_k given, delta = alpha / sqrt(1+alpha^2)
        y = delta*|z0| + sqrt(1-delta^2)*z1
        x = mu + sigma*y
        """
        n = int(n)
        key, k_idx, k0, k1 = jr.split(key, 4)

        idx = jr.categorical(k_idx, jnp.log(self.weights), shape=(n,))  # (n,)
        mu = self.mus[idx, :]                       # (n,d)
        sigma = self.sigmas[idx][:, None]           # (n,1)

        alpha_k = self.alphas[idx][:, None]         # (n,1)
        delta = alpha_k / jnp.sqrt(1.0 + alpha_k * alpha_k)
        s = jnp.sqrt(1.0 - delta * delta)

        z0 = jr.normal(k0, (n, self.d), dtype=jnp.float64)
        z1 = jr.normal(k1, (n, self.d), dtype=jnp.float64)
        y = delta * jnp.abs(z0) + s * z1
        x = mu + sigma * y
        return x, key

    def true_mean(self) -> Array:
        """
        E[X] = Σ_k w_k (mu_k + sigma_k * delta_k * sqrt(2/pi))
        where delta_k = alpha_k / sqrt(1 + alpha_k^2).
        """
        delta = self.alphas / jnp.sqrt(1.0 + self.alphas ** 2)          # (K,)
        skew_offset = self.sigmas * delta * jnp.sqrt(2.0 / jnp.pi)      # (K,)
        component_means = self.mus + skew_offset[:, None]                # (K,d)
        return jnp.sum(self.weights[:, None] * component_means, axis=0)  # (d,)

    def true_cov(self) -> Array:
        """
        Cov[X] = Σ_k w_k (C_k + (mu_k* - mu*)(mu_k* - mu*)^T)
        where mu_k* = component mean, mu* = mixture mean,
        C_k = diag(sigma_k^2 * (1 - 2*delta_k^2/pi)) (skew-normal marginal variance, diagonal).
        """
        delta = self.alphas / jnp.sqrt(1.0 + self.alphas ** 2)           # (K,)
        comp_var = self.sigmas ** 2 * (1.0 - 2.0 * delta ** 2 / jnp.pi) # (K,)

        skew_offset = self.sigmas * delta * jnp.sqrt(2.0 / jnp.pi)      # (K,)
        comp_means = self.mus + skew_offset[:, None]                     # (K,d)
        mix_mean = jnp.sum(self.weights[:, None] * comp_means, axis=0)  # (d,)

        diff = comp_means - mix_mean[None, :]                            # (K,d)
        # Σ_k w_k * (diag(comp_var_k) + diff_k diff_k^T)
        cov = jnp.sum(
            self.weights[:, None, None] * (
                jnp.eye(self.d, dtype=jnp.float64)[None, :, :] * comp_var[:, None, None]
                + diff[:, :, None] * diff[:, None, :]
            ),
            axis=0,
        )
        return cov
