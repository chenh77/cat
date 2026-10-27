# targets/iohmm.py
"""Native-JAX port of posteriordb's `iohmm_reg_simulated-iohmm_reg`
(`models/stan/iohmm_reg.stan` + `data/data/iohmm_reg_simulated.json`,
originally `github.com/luisdamiano/stancon18`): an input-output HMM
regression over a simulated series (`T=500`, `K=3` states, `M=4`-dim
exogenous input `u_t`),

    pi1 ~ implicit uniform over simplex(K)   (initial-state distribution)
    w[k], b[k] ~ N(0, 5) elementwise, k=1..K  (M-vectors, unconstrained)
    sigma[k] ~ N(0, 3) [restricted to sigma[k] > 0 by the parameter's own
                         <lower=0> constraint -- Stan does *not* renormalize
                         this into a proper half-normal density, it just
                         evaluates normal_lpdf at the positive constrained
                         value plus the constraining transform's Jacobian,
                         same as any other `<lower=0>` parameter with a
                         two-sided prior written on it]
    y[t] | z_t ~ N(u[t]' b[z_t], sigma[z_t])

marginalized over latent states via the forward algorithm. Unlike a time-*homogeneous*
HMM transition matrix, this model's "transition" is really just an
input-dependent softmax regression, `unA[t][j] = u[t]' w[j]`,
`A[t] = softmax(unA[t])` -- notionally `Pr(z_t = j | u_t)` with no explicit
dependence on the *previous* state `z_{t-1}` at all.

`log_prob_single` below does *not* use a `jax.lax.scan` recursion: the reproduced
index quirk (see `permute_components`'s docstring and the derivation inline in
`log_prob_single`) collapses the forward recursion into a closed-form
reduction over all `T` timesteps at once, avoiding a `T`-long sequential
dependency chain.

**Reproduced quirk (not a modeling choice of this port -- verified against
bridgestan): the shipped Stan model's
own forward-algorithm loop indexes `logA[t]` by `i` (the *previous* state
being summed over), not by `j` (the *current* state being accumulated
into):**

    accumulator[i] = logalpha[t-1, i] + logA[t][i] + logoblik[t][j]
    logalpha[t, j] = log_sum_exp_i(accumulator[i])

Since `logA[t][i]` here does not depend on `j`, it (along with
`logalpha[t-1,:]`) factors completely out of the `log_sum_exp` over `i`,
collapsing to a single scalar shared by *every* `j` at that timestep:

    logalpha[t, j] = log_sum_exp_i(logalpha[t-1, i] + logA[t][i]) + logoblik[t, j]

i.e. `w` (via `logA`) only ever influences which *previous*-state term
dominates the running scalar, never which *current* state is favored
(that comes entirely from `logoblik`, the observation likelihood) -- almost
certainly an unintended index swap in the original stancon18 example (present at
every timestep `t>=2`), but bridgestan/Stan compute exactly this, so it is
reproduced here rather than the presumably-intended `logA[t][j]`.

Unconstrained layout: `[pi1_free(K-1) | w_free(K*M) | b_free(K*M) |
sigma_free(K)]`, `dim = (K-1) + 2*K*M + K = 29` for `K=3, M=4`. Within the
`w_free`/`b_free` blocks, `array[K] vector[M]`'s flattening is
row-major-by-state (state `k`'s `M` components, then state `k+1`'s, ...),
i.e. plain `reshape(K, M)`. bridgestan's `param_unc_names` labels (`w.2.1` right after
`w.1.1`) suggest the opposite order; `param_constrain` confirms the per-state blocks.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import vmap
from jax.scipy.special import logsumexp

from .common import Array, _as_2d
from .constraints import make_simplex_constrain, positive_constrain, sum_to_zero_matrix

_LOG_2PI = float(np.log(2.0 * np.pi))


def _normal_lpdf_scalar(y, mu, sigma):
    return -0.5 * (_LOG_2PI + 2.0 * jnp.log(sigma) + ((y - mu) / sigma) ** 2)


def _normal_lpdf_logscale(y, mu, log_sigma):
    """Computes
    `_normal_lpdf_scalar(y, mu, exp(log_sigma))` directly from `log_sigma`,
    avoiding the `log(exp(log_sigma))` round-trip that can underflow to
    `-inf + inf = nan` for very negative `log_sigma` (this target's `sigma`
    has a two-sided-but-positivity-constrained prior, so this failure mode is
    reachable during `mode_explore`'s LBFGS)."""
    return -0.5 * _LOG_2PI - log_sigma - 0.5 * jnp.exp(-2.0 * log_sigma) * (y - mu) ** 2


@dataclass
class IohmmReg:
    """posteriordb `iohmm_reg_simulated-iohmm_reg`. `u`: (T,M) exogenous
    input, `y`: (T,) observed output, `K`: number of hidden states."""
    u: Array
    y: Array
    K: int

    def __post_init__(self):
        self.u = jnp.asarray(np.asarray(self.u), dtype=jnp.float64)
        self.y = jnp.asarray(np.asarray(self.y), dtype=jnp.float64)
        self.T, self.M = self.u.shape
        assert self.y.shape[0] == self.T
        self._simplex = make_simplex_constrain(self.K)
        K, M = self.K, self.M
        # [pi1_free(K-1) | w_free(K*M) | b_free(K*M) | sigma_free(K)]
        self.dim = (K - 1) + K * M + K * M + K

        # precomputed for `permute_components` -- see that method's docstring.
        A = sum_to_zero_matrix(K)  # (K, K-1)
        self._A = jnp.asarray(A)
        self._A_pinv = jnp.asarray(np.linalg.pinv(A))  # (K-1, K)

    def _unpack_single(self, x: Array):
        K, M = self.K, self.M
        i = 0
        pi1_free = x[i: i + (K - 1)]; i += (K - 1)
        w_free = x[i: i + K * M]; i += K * M
        b_free = x[i: i + K * M]; i += K * M
        sigma_free = x[i: i + K]; i += K

        pi1, lj_pi1 = self._simplex(pi1_free)
        w = w_free.reshape(K, M)  # w[k] = state k's regressor, identity transform
        b = b_free.reshape(K, M)
        sigma, lj_sigma_elem = positive_constrain(sigma_free)

        logjac = lj_pi1 + jnp.sum(lj_sigma_elem)
        return pi1, w, b, sigma, sigma_free, logjac

    def log_prob_single(self, x: Array) -> Array:
        pi1, w, b, sigma, log_sigma, logjac = self._unpack_single(x)

        unA = self.u @ w.T  # (T,K); unA[t,j] = u[t] . w[j]
        logA = jax.nn.log_softmax(unA, axis=-1)  # (T,K); row 0 unused (Stan: t=1 is a filler)

        mu_y = self.u @ b.T  # (T,K); mu_y[t,j] = u[t] . b[j]
        logoblik = _normal_lpdf_logscale(self.y[:, None], mu_y, log_sigma[None, :])  # (T,K)

        # The forward-algorithm recursion (module docstring) reproduces the
        # shipped model's index quirk: logA_t (indexed by the *previous*
        # state, summed over) does not depend on the *current* state j, so
        # logalpha_t[j] collapses to c_t + logoblik_t[j] for every t >= 1,
        # with c_t = logsumexp_i(logalpha_{t-1}[i] + logA_t[i]). Substituting
        # the collapsed form of logalpha_{t-1} into that recursion for
        # t >= 2 shows c_t = c_{t-1} + logsumexp_i(logoblik_{t-1}[i] +
        # logA_t[i]) -- i.e. each increment depends only on *local*,
        # precomputable per-timestep data, not on the running recursion
        # state. So c_{T-1} = c_1 + sum of those increments: a plain
        # reduction over all T timesteps at once, with no sequential
        # `lax.scan` needed (checked against the scan formulation on random
        # inputs: log_prob and its gradient agree to float64 noise).
        c1 = logsumexp(jnp.log(pi1) + logoblik[0] + logA[1])
        if self.T > 2:
            increments = logsumexp(logoblik[1:-1] + logA[2:], axis=-1)  # (T-2,)
            c_last = c1 + jnp.sum(increments)
        else:
            c_last = c1
        log_lik = c_last + logsumexp(logoblik[-1])

        log_prior = (
            jnp.sum(_normal_lpdf_scalar(w, 0.0, 5.0))
            + jnp.sum(_normal_lpdf_scalar(b, 0.0, 5.0))
            + jnp.sum(_normal_lpdf_scalar(sigma, 0.0, 3.0))
        )

        return logjac + log_prior + log_lik

    def log_prob(self, x: Array) -> Array:
        x2 = _as_2d(jnp.asarray(x, dtype=jnp.float64))
        return vmap(self.log_prob_single)(x2)

    def permute_components(self, x: Array, perm: Array) -> Array:
        """Exact `log_prob`-preserving action of `S_K` on the unconstrained
        parameterization: relabels state `k` as state `perm[k]` consistently
        across `pi1`, `w`, `b`, `sigma` -- unlike HMMs whose state labels are
        pinned by an `ordered[K]` constraint, this model has *no* identifiability constraint on `w`,
        `b`, or `sigma` at all, so (verified directly: permuting `w[k],
        b[k], sigma[k]` together for every `k`, and `pi1[k]` to match,
        leaves every `logalpha[t,:]` exactly permuted the same way by
        induction on `t` -- the forward algorithm's `c_t` term is a
        `log_sum_exp` over the *previous* state, invariant to how the
        *current* state axis is labeled) this target is genuinely `K!`-fold
        symmetric. `w_free`/`b_free` (reshaped `(K,M)`, identity transform)
        and `sigma_free` permute trivially along their leading `K` axis;
        `pi1_free`'s simplex map needs the change of basis
        `pinv(A) @ P_perm @ A` of the ILR simplex transform."""
        K, M = self.K, self.M
        x = jnp.asarray(x)
        perm = jnp.asarray(perm)
        i = 0
        pi1_free = x[..., i: i + (K - 1)]; i += (K - 1)
        w_free = x[..., i: i + K * M]; i += K * M
        b_free = x[..., i: i + K * M]; i += K * M
        sigma_free = x[..., i: i + K]; i += K

        P = jnp.eye(K)[perm]                # (K, K): P @ v == v[perm]
        M_theta = self._A_pinv @ P @ self._A  # (K-1, K-1)
        pi1_free2 = jnp.einsum("ij,...j->...i", M_theta, pi1_free)

        w2 = jnp.take(w_free.reshape(*x.shape[:-1], K, M), perm, axis=-2).reshape(*x.shape[:-1], K * M)
        b2 = jnp.take(b_free.reshape(*x.shape[:-1], K, M), perm, axis=-2).reshape(*x.shape[:-1], K * M)
        sigma_free2 = jnp.take(sigma_free, perm, axis=-1)

        return jnp.concatenate([pi1_free2, w2, b2, sigma_free2], axis=-1)

    # -- reference pool for evaluate_metrics's KL(p||q)/SW2, set post-construction
    _reference: "Array | None" = None

    def sample(self, key, n: int):
        if self._reference is None:
            raise RuntimeError(f"{type(self).__name__} has no reference pool set; "
                                "assign ._reference before calling .sample().")
        import jax.random as _jr
        m = self._reference.shape[0]
        idx = _jr.choice(key, m, shape=(int(n),), replace=bool(n > m))
        return self._reference[idx], key

    def score(self, x: Array) -> Array:
        x2 = _as_2d(jnp.asarray(x, dtype=jnp.float64))
        return vmap(jax.grad(self.log_prob_single))(x2)
