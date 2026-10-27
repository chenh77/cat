# targets/lda.py
"""Native-JAX port of posteriordb's `three_men1-ldaK2`
(`models/stan/ldaK2.stan` + `data/data/three_men1.json`): Latent Dirichlet
Allocation, `K=2` topics, token indicators marginalized out,

    theta[m] ~ Dirichlet(1,1)               for each of M documents
    phi[k]   ~ Dirichlet(1,...,1)  (V-dim)  for each of K=2 topics
    w[n] ~ sum_k theta[doc[n], k] * phi[k, w[n]]     (marginalized over topic)

Supports both Stan's unconstrained ILR parameterization
(`parameterization="stan"`, the default) and a prior-Gaussian parameterization
(`parameterization="prior_gaussian"`). The latter maps independent standard
normals for each topic directly through the exact Dirichlet(1) Gaussian-CDF
stick break. Both induce the same priors and posterior on constrained `theta`
and `phi`.

The experiments use `parameterization="prior_gaussian"`.

**Parameter layout:** the free coordinates of `array[K] simplex[V] phi` are laid out
topic-block-contiguous (`phi[1]`'s `V-1` free dims, then `phi[2]`'s), i.e.
`reshape(K, V - 1)` with no transpose, matching `theta`'s layout. bridgestan's
`param_unc_names` (`phi.1.1, phi.2.1, phi.1.2, ...`) reads as topic-index-fastest, but
its two integers are `<free-coordinate-index>.<topic-index>`; getting this backwards
still gives a finite (wrong) log density. The port is checked against bridgestan's
`log_density` and `log_density_gradient`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import gammaln, ndtri as scipy_ndtri
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import vmap
from jax.scipy.special import logsumexp

from .common import Array, _as_2d
from .constraints import make_simplex_constrain, sum_to_zero_matrix


def _gaussian_stick_log_constrain(z: Array) -> Array:
    """Log simplex coordinates for the N(0,I) -> Dirichlet(1) stick map."""
    z = jnp.asarray(z)
    n_free = z.shape[-1]
    remaining_counts = jnp.arange(n_free, 0, -1, dtype=z.dtype)
    log_one_minus_break = jax.scipy.special.log_ndtr(-z) / remaining_counts
    safe_log_one_minus = jnp.minimum(log_one_minus_break, -jnp.finfo(z.dtype).tiny)
    log_break_direct = jnp.log(-jnp.expm1(safe_log_one_minus))
    # For very negative z, log_ndtr(-z) rounds to -0 before log_ndtr(z)
    # underflows. There the exact break is u/count up to negligible O(u^2).
    log_break_tail = jax.scipy.special.log_ndtr(z) - jnp.log(remaining_counts)
    log_break = jnp.where(z < -30.0, log_break_tail, log_break_direct)
    log_remaining_before = jnp.concatenate(
        [jnp.zeros_like(log_one_minus_break[..., :1]),
         jnp.cumsum(log_one_minus_break[..., :-1], axis=-1)],
        axis=-1,
    )
    log_first = log_remaining_before + log_break
    log_last = jnp.sum(log_one_minus_break, axis=-1, keepdims=True)
    return jnp.concatenate([log_first, log_last], axis=-1)


def _gaussian_stick_constrain(z: Array) -> Array:
    """N(0,I) -> Dirichlet(1) with a dimension-preserving stick break."""
    return jnp.exp(_gaussian_stick_log_constrain(z))


def _gaussian_stick_unconstrain(p: Array) -> Array:
    """Inverse of `_gaussian_stick_constrain` for interior simplex points."""
    p = jnp.asarray(p)
    n_free = p.shape[-1] - 1
    remaining_counts = jnp.arange(n_free, 0, -1, dtype=p.dtype)
    remaining_before = jnp.flip(jnp.cumsum(jnp.flip(p, axis=-1), axis=-1), axis=-1)[..., :-1]
    tiny = jnp.finfo(p.dtype).eps
    breaks = jnp.clip(p[..., :-1] / remaining_before, 0.0, 1.0 - tiny)
    u = -jnp.expm1(remaining_counts * jnp.log1p(-breaks))
    u = jnp.clip(u, tiny, 1.0 - tiny)
    return jax.scipy.special.ndtri(u)


def fit_posterior_quantile_knots(
    samples: np.ndarray,
    *,
    n_knots: int = 65,
    tail_prob: float = 1e-3,
    sign_flip_mask: np.ndarray | None = None,
):
    """Fit a coordinatewise normal-score map to independent pilot draws.

    Returns ``(x_knots, z_knots)`` with shapes ``(n_knots, d)`` and
    ``(n_knots,)``, for a monotone rational-quadratic
    spline from normal scores ``z`` to fitted pilot quantiles ``x``, with
    C1 linear tails keeping the map bijective on all of R.

    ``sign_flip_mask`` identifies coordinates negated by the target's exact
    label symmetry.  Their fitted quantiles are projected onto exact odd
    symmetry, making the resulting transform commute with that group action
    rather than relying on finite-pilot Monte Carlo symmetry.
    """
    samples = np.asarray(samples, dtype=np.float64)
    if samples.ndim != 2 or samples.shape[0] < 4:
        raise ValueError("samples must have shape (n,d) with n >= 4.")
    if n_knots < 5:
        raise ValueError("n_knots must be >= 5.")
    if not (0.0 < tail_prob < 0.5):
        raise ValueError("tail_prob must be in (0, 0.5).")
    probs = np.linspace(float(tail_prob), 1.0 - float(tail_prob), int(n_knots))
    x_knots = np.quantile(samples, probs, axis=0)
    if sign_flip_mask is not None:
        mask = np.asarray(sign_flip_mask, dtype=bool)
        if mask.shape != (samples.shape[1],):
            raise ValueError(
                f"sign_flip_mask must have shape {(samples.shape[1],)}, got {mask.shape}."
            )
        x_knots[:, mask] = 0.5 * (
            x_knots[:, mask] - np.flip(x_knots[:, mask], axis=0)
        )
    slopes = np.diff(x_knots, axis=0)
    if np.any(~np.isfinite(x_knots)) or np.any(slopes <= 0.0):
        raise ValueError(
            "Pilot quantiles must be finite and strictly increasing in every coordinate."
        )
    return x_knots, scipy_ndtri(probs)


@dataclass
class LdaK2:
    """posteriordb `three_men1-ldaK2`. `w`: (N,) 1-indexed word ids in
    [1,V], `doc`: (N,) 1-indexed doc ids in [1,M], `V`: vocab size,
    `M`: number of documents, `K`: number of topics (2)."""
    w: Array
    doc: Array
    V: int
    M: int
    K: int = 2
    alpha: float = 1.0  # theta's Dirichlet concentration (flat: all-ones)
    beta: float = 1.0   # phi's Dirichlet concentration (flat: all-ones)
    parameterization: str = "stan"

    def __post_init__(self):
        self.w0 = np.asarray(self.w, dtype=np.int64) - 1     # 0-indexed, (N,)
        self.doc0 = np.asarray(self.doc, dtype=np.int64) - 1  # 0-indexed, (N,)
        self.N = int(self.w0.shape[0])
        self._theta_simplex = make_simplex_constrain(self.K)
        self._phi_simplex = make_simplex_constrain(self.V)
        if self.parameterization not in ("stan", "prior_gaussian"):
            raise ValueError("parameterization must be 'stan' or 'prior_gaussian'.")
        if self.parameterization == "prior_gaussian" and (
            self.K != 2 or self.alpha != 1.0 or self.beta != 1.0
        ):
            raise ValueError(
                "prior_gaussian currently requires K=2 and flat Dirichlet(1) priors."
            )
        # Stan layout: theta block, then topic-block-contiguous phi blocks.
        self.theta_len = self.M * (self.K - 1)
        self.phi_len = (self.V - 1) * self.K
        self.dim = self.theta_len + self.phi_len
        self._A_theta = jnp.asarray(sum_to_zero_matrix(self.K))
        self._A_phi = jnp.asarray(sum_to_zero_matrix(self.V))

        # Dirichlet(alpha,...,alpha) log-normalizer constants (parameter-
        # independent, but included for a full-precision value match against
        # bridgestan's propto=False).
        self._log_dirichlet_const_theta = float(
            gammaln(self.K * self.alpha) - self.K * gammaln(self.alpha)
        )
        self._log_dirichlet_const_phi = float(
            gammaln(self.V * self.beta) - self.V * gammaln(self.beta)
        )

        self.w0_j = jnp.asarray(self.w0)
        self.doc0_j = jnp.asarray(self.doc0)
        # The marginalized token likelihood depends on an observation only
        # through (document, word).  Aggregate repeated pairs exactly once:
        # three_men1 has 4,999 tokens but only 503 unique pairs, so this cuts
        # the dominant batched log-sum-exp tensor by almost 10x without
        # changing the target or introducing stochastic data subsampling.
        doc_word = np.stack([self.doc0, self.w0], axis=1)
        unique_doc_word, counts = np.unique(
            doc_word, axis=0, return_counts=True
        )
        self.doc_count0_j = jnp.asarray(unique_doc_word[:, 0])
        self.w_count0_j = jnp.asarray(unique_doc_word[:, 1])
        self.doc_word_counts_j = jnp.asarray(counts)
        self.N_unique_doc_word = int(unique_doc_word.shape[0])

    def _unpack_stan_single(self, x: Array):
        theta_free = x[: self.theta_len].reshape(self.M, self.K - 1)
        # topic-block-contiguous: phi[1]'s (V-1) free dims, then phi[2]'s, ...
        # -- see class docstring for why this is NOT reshape(V-1, K).T.
        phi_free = x[self.theta_len:].reshape(self.K, self.V - 1)

        theta, logjac_theta = vmap(self._theta_simplex)(theta_free)  # (M,K), (M,)
        phi_T, logjac_phi = vmap(self._phi_simplex)(phi_free)        # (K,V), (K,)

        logjac = jnp.sum(logjac_theta) + jnp.sum(logjac_phi)
        return theta, phi_T, logjac

    def _unpack_prior_gaussian_single(self, z: Array):
        theta_z = z[: self.theta_len].reshape(self.M)
        # Topic-block-contiguous raw stick coordinates: phi[1]'s standard
        # normals followed by phi[2]'s.  No topic mean/contrast rotation is
        # applied; topic swapping therefore exchanges these two blocks.
        phi_z = z[self.theta_len:].reshape(self.K, self.V - 1)
        theta = jnp.stack(
            [jax.scipy.special.ndtr(theta_z), jax.scipy.special.ndtr(-theta_z)],
            axis=-1,
        )
        phi = _gaussian_stick_constrain(phi_z)
        return theta, phi, jnp.asarray(0.0, dtype=z.dtype)

    def _prior_gaussian_log_simplexes_single(self, z: Array):
        """Stable log(theta), log(phi), avoiding probability-space underflow."""
        theta_z = z[: self.theta_len].reshape(self.M)
        phi_z = z[self.theta_len:].reshape(self.K, self.V - 1)
        log_theta = jnp.stack(
            [jax.scipy.special.log_ndtr(theta_z),
             jax.scipy.special.log_ndtr(-theta_z)],
            axis=-1,
        )
        return log_theta, _gaussian_stick_log_constrain(phi_z)

    def _unpack_single(self, x: Array):
        if self.parameterization == "prior_gaussian":
            return self._unpack_prior_gaussian_single(x)
        return self._unpack_stan_single(x)

    def stan_to_prior_gaussian(self, x: Array) -> Array:
        """Convert Stan/ILR coordinates to prior-Gaussian coordinates."""
        x = jnp.asarray(x, dtype=jnp.float64)
        original_shape = x.shape
        x2 = x.reshape((-1, self.dim))
        theta, phi, _ = vmap(self._unpack_stan_single)(x2)
        tiny = jnp.finfo(x.dtype).eps
        theta_z = jax.scipy.special.ndtri(jnp.clip(theta[..., 0], tiny, 1.0 - tiny))
        phi_z = _gaussian_stick_unconstrain(phi)
        z = jnp.concatenate(
            [theta_z, phi_z.reshape((-1, self.phi_len))], axis=-1
        )
        return z.reshape(original_shape)

    def prior_gaussian_to_stan(self, z: Array) -> Array:
        """Convert prior-Gaussian coordinates to the original Stan/ILR space."""
        z = jnp.asarray(z, dtype=jnp.float64)
        original_shape = z.shape
        z2 = z.reshape((-1, self.dim))
        log_theta, log_phi = vmap(self._prior_gaussian_log_simplexes_single)(z2)
        theta_free = jnp.einsum("nmk,ki->nmi", log_theta, self._A_theta)
        phi_free = jnp.einsum("nkv,vi->nki", log_phi, self._A_phi)
        x = jnp.concatenate(
            [theta_free.reshape((-1, self.theta_len)),
             phi_free.reshape((-1, self.phi_len))],
            axis=-1,
        )
        return x.reshape(original_shape)

    def stan_to_inference(self, x: Array) -> Array:
        """Convert Stan coordinates to this instance's inference coordinates."""
        if self.parameterization == "stan":
            return jnp.asarray(x, dtype=jnp.float64)
        return self.stan_to_prior_gaussian(x)

    def inference_to_stan(self, x: Array) -> Array:
        """Convert this instance's inference coordinates to Stan coordinates."""
        if self.parameterization == "stan":
            return jnp.asarray(x, dtype=jnp.float64)
        return self.prior_gaussian_to_stan(x)

    def log_prob_single(self, x: Array) -> Array:
        if self.parameterization == "prior_gaussian":
            log_theta, log_phi = self._prior_gaussian_log_simplexes_single(x)
            log_prior = -0.5 * jnp.sum(x * x) - 0.5 * self.dim * np.log(2.0 * np.pi)
            gamma = (
                log_theta[self.doc_count0_j]
                + log_phi[:, self.w_count0_j].T
            )
        else:
            theta, phi, logjac = self._unpack_stan_single(x)
            log_prior_theta = self.M * self._log_dirichlet_const_theta + (
                (self.alpha - 1.0) * jnp.sum(jnp.log(theta))
            )
            log_prior_phi = self.K * self._log_dirichlet_const_phi + (
                (self.beta - 1.0) * jnp.sum(jnp.log(phi))
            )
            log_prior = logjac + log_prior_theta + log_prior_phi
            theta_doc = theta[self.doc_count0_j]
            phi_w = phi[:, self.w_count0_j].T
            gamma = jnp.log(theta_doc) + jnp.log(phi_w)
        log_lik = jnp.sum(
            self.doc_word_counts_j * logsumexp(gamma, axis=-1)
        )

        return log_prior + log_lik

    def log_prob(self, x: Array) -> Array:
        x2 = _as_2d(jnp.asarray(x, dtype=jnp.float64))
        return vmap(self.log_prob_single)(x2)

    def topic_swap(self, x: Array) -> Array:
        """Exact `log_prob`-preserving involution swapping "topic 1" and
        "topic 2". In Stan coordinates this negates each document's theta
        coordinate and swaps the two phi blocks. The prior-Gaussian coordinates
        use the same topic-block swap. Both forms give exactly-zero
        `log_prob_single` discrepancy (not just close) over sampled points --
        the NUTS reference was canonicalized to one topic-label basin with
        this map before computing split-Rhat and pooling, since raw chains
        land in either basin ~50/50. Only valid for `K == 2` -- the
        theta-negation identity relies on K=2 and does not hold generally for
        `K > 2`."""
        assert self.K == 2, "topic_swap's theta-negation identity only holds for K == 2"
        x = jnp.asarray(x)
        theta_free = x[..., : self.theta_len]
        phi_free = x[..., self.theta_len:].reshape(x.shape[:-1] + (self.K, self.V - 1))
        phi_swapped = phi_free[..., ::-1, :].reshape(x.shape[:-1] + (self.phi_len,))
        return jnp.concatenate([-theta_free, phi_swapped], axis=-1)

    # -- reference pool (long-NUTS draws mapped into this module's unconstrained
    # space), set after construction; `sample` draws from it.
    _reference: "Array | None" = None

    def sample(self, key, n: int):
        if self._reference is None:
            raise RuntimeError(f"{type(self).__name__} has no reference pool set; "
                                "assign .{_reference} before calling .sample().")
        import jax.random as _jr
        m = self._reference.shape[0]
        idx = _jr.choice(key, m, shape=(int(n),), replace=bool(n > m))
        return self._reference[idx], key

    def score(self, x: Array) -> Array:
        x2 = _as_2d(jnp.asarray(x, dtype=jnp.float64))
        return vmap(jax.grad(self.log_prob_single))(x2)
