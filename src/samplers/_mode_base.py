# _mode_base.py
"""Laplace-mixture base/reference distribution built from a ModeExplore
inventory, shared by the `modes=` option of `aft.py`, `apt.py`, `fab.py`,
`gat.py` and `temperflow.py`, and by `alps.py`'s leap proposal
(`inflate = leap_inflate`).

Given modes {(w_j, mu_j, Sigma_j)} (the inventory CAT transports and
ALPS leaps between), this is

    b(x) = sum_j w_j N(x; mu_j, c^2 Sigma_j),     c = `inflate`,

plus its tempered version, the Laplace approximation of pi^beta at the same
modes,

    b_beta(x) = sum_j w_j(beta) N(x; mu_j, c^2 Sigma_j / beta),
    w_j(beta) ∝ pi~(mu_j)^beta |Sigma_j / beta|^{1/2}.

`mode_explore` stores Laplace weights w_j ∝ pi~(mu_j) |Sigma_j|^{1/2}
(`recompute_mode_weights`), so log pi~(mu_j) = log w_j - 0.5 logdet Sigma_j
up to a shared constant, and

    log w_j(beta) = beta (log w_j - 0.5 logdet Sigma_j) + 0.5 logdet Sigma_j + const

(the |1/beta|^{d/2} factor is shared by every mode and cancels). beta=1
recovers b exactly. Only TemperFlow, whose path is pi^beta itself with no
reference term, uses beta != 1; the bridge-based baselines use the fixed b.

Methods take `beta` as a traced value, so one jitted function serves every
rung. `log_prob` accepts a single point (d,) or a batch (..., d).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jr
from jax.scipy.special import logsumexp

from ..mode_explore.density import modes_to_arrays
from ..mode_explore.explore import Mode
from ..utils.numeric import chol_jitter_jax, sym_jax

Array = jax.Array


# eq=False keeps identity hashing, so instances can be closed over by
# functions passed as static arguments to jit.
@dataclass(frozen=True, eq=False)
class LaplaceMixtureBase:
    mu: Array            # (m, d)
    L: Array             # (m, d, d) Cholesky factor of c^2 Sigma_j
    log_w: Array         # (m,) normalized log weights at beta=1
    half_logdet: Array   # (m,) 0.5 logdet Sigma_j (un-inflated)

    @classmethod
    def from_modes(
        cls, modes: Sequence[Mode], *, inflate: float = 1.0, dtype=jnp.float64,
    ) -> "LaplaceMixtureBase":
        if inflate <= 0.0:
            raise ValueError("inflate must be positive.")
        mu, Sigma, logw = modes_to_arrays(modes, dtype=dtype)
        L0 = chol_jitter_jax(sym_jax(Sigma))
        half_logdet = jnp.sum(jnp.log(jnp.diagonal(L0, axis1=-2, axis2=-1)), axis=-1)
        return cls(
            mu=mu, L=float(inflate) * L0, log_w=logw - logsumexp(logw),
            half_logdet=half_logdet,
        )

    @property
    def n_modes(self) -> int:
        return int(self.mu.shape[0])

    @property
    def dim(self) -> int:
        return int(self.mu.shape[1])

    def log_weights(self, beta=1.0) -> Array:
        lw = beta * (self.log_w - self.half_logdet) + self.half_logdet
        return lw - logsumexp(lw)

    def log_prob(self, x: Array, beta=1.0) -> Array:
        x = jnp.asarray(x)
        d = self.dim
        batch_shape = x.shape[:-1]
        xf = x.reshape(-1, d)
        beta = jnp.asarray(beta, dtype=xf.dtype)

        def per_mode(mu_j, L_j):
            y = jax.scipy.linalg.solve_triangular(L_j, (xf - mu_j).T, lower=True).T
            logdet_L = jnp.sum(jnp.log(jnp.diag(L_j)))
            return -0.5 * beta * jnp.sum(y * y, axis=-1) - logdet_L

        comp = jax.vmap(per_mode)(self.mu, self.L)   # (m, n)
        const = 0.5 * d * (jnp.log(beta) - jnp.log(2.0 * jnp.pi))
        lp = logsumexp(comp + self.log_weights(beta)[:, None], axis=0) + const
        return lp.reshape(batch_shape)

    def sample(self, key: Array, n: int, beta=1.0, *, return_components: bool = False):
        """(n, d) draws from b_beta; with `return_components`, also the (n,)
        component labels (ALPS tracks them as each chain's mode label)."""
        k_comp, k_eps = jr.split(key)
        n = int(n)
        dtype = self.mu.dtype
        comps = jr.categorical(k_comp, self.log_weights(beta), shape=(n,))
        eps = jr.normal(k_eps, (n, self.dim), dtype=dtype)
        scale = jnp.asarray(beta, dtype=dtype) ** -0.5

        # Accumulate mode by mode: gathering L[comps] would materialize an
        # (n, d, d) array, prohibitive at the d ~ 10^3 targets here.
        def body(z, inputs):
            j, mu_j, L_j = inputs
            z_j = mu_j + scale * (eps @ L_j.T)
            return jnp.where((comps == j)[:, None], z_j, z), None

        z, _ = jax.lax.scan(
            body, jnp.zeros((n, self.dim), dtype=dtype),
            (jnp.arange(self.n_modes), self.mu, self.L),
        )
        return (z, comps) if return_components else z


def build_mode_base(
    modes: Optional[Sequence[Mode]], inflate: float = 1.0, dtype=jnp.float64,
) -> Optional[LaplaceMixtureBase]:
    if modes is None:
        return None
    if len(modes) == 0:
        raise ValueError("modes must be non-empty when given.")
    return LaplaceMixtureBase.from_modes(modes, inflate=inflate, dtype=dtype)


class DistraxLikeMixture:
    """Duck-typed stand-in for a `distrax.Distribution`, exposing only the
    two calls fab-jax's flow makes on its base (`sample(seed=,
    sample_shape=)`, `log_prob(value=)`)."""

    def __init__(self, base: LaplaceMixtureBase):
        self.base = base

    def sample(self, *, seed, sample_shape=()):
        shape = (sample_shape,) if isinstance(sample_shape, int) else tuple(sample_shape)
        n = 1
        for s in shape:
            n *= int(s)
        return self.base.sample(seed, n).reshape(shape + (self.base.dim,))

    def log_prob(self, value):
        return self.base.log_prob(value)
