# chain.py
"""Flow-chain composition and mixture-of-transported-base-distributions
plumbing, generic over any Flax module exposing `.f`/`.finv` methods with
RealNVPBounded's (x)->(y,logdet) signature (see `nets.py`).

A "chain" here is a list of per-step flow param dicts applied in sequence via
`jax.lax.scan`, so the JIT trace graph stays O(1) regardless of chain length
(e.g. a mode's incremental flows S_k^j = F_k^j o ... o F_1^j).

`component_log_q`/`mixture_log_q`/`sample_mixture` push a `base` distribution
(anything with `.mu`, `.sample(key, n)`, `.log_prob(x)` -- e.g.
`src.mode_explore.density.ModeGaussian`) through such a chain and combine several
components into a weighted mixture. With `whiten=True`, the base must also
provide `.chol()` and the chain acts on `L^{-1}(x-mu)` rather than merely
`x-mu`. This module only relies on that duck-typed interface, not on any
concrete distribution class.

Callers may also supply a mode-specific posterior chart: monotone marginal
normal-score splines followed by dense pilot-copula whitening. The helpers
below carry both chart Jacobians, so an identity flow remains exactly the
identity map and densities stay in the original physical coordinates.
"""
from __future__ import annotations

from typing import Any, List, NamedTuple, Optional, Sequence, Tuple

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jr

from .nets import RealNVPBounded
from ..utils.numeric import as2d_jax

Array = jax.Array


def _rqs_knot_derivatives(x_knots: Array, u_knots: Array) -> Array:
    """Positive C1 knot derivatives dx/du for a fixed monotone RQ spline."""
    secants = (x_knots[1:] - x_knots[:-1]) / (u_knots[1:] - u_knots[:-1])[:, None]
    interior = 2.0 * secants[:-1] * secants[1:] / (secants[:-1] + secants[1:])
    return jnp.concatenate([secants[:1], interior, secants[-1:]], axis=0)


def _rqs_forward_bin(theta, x0, x1, du, d0, d1):
    """Rational-quadratic bin value and derivative dx/du."""
    width = x1 - x0
    delta = width / du
    omt = 1.0 - theta
    theta_omt = theta * omt
    denom = delta + (d1 + d0 - 2.0 * delta) * theta_omt
    numer = width * (delta * theta * theta + d0 * theta_omt)
    x = x0 + numer / denom
    deriv_numer = delta * delta * (
        d1 * theta * theta
        + 2.0 * delta * theta_omt
        + d0 * omt * omt
    )
    return x, deriv_numer / (denom * denom)


def quantile_coordinate_inverse(
    x: Array, x_knots: Array, u_knots: Array, coordinate_chol: Optional[Array] = None
):
    """Map physical coordinates ``x`` to pilot-normal-score coordinates ``u``.

    Returns ``(u, log|du/dx|)``.  Linear extrapolation outside the fitted
    knots keeps the component chart globally bijective.
    """
    x = as2d_jax(x)
    derivatives = _rqs_knot_derivatives(x_knots, u_knots)
    idx = jax.vmap(
        lambda knots_j, values_j: jnp.searchsorted(knots_j, values_j, side="right") - 1,
        in_axes=(1, 1), out_axes=1,
    )(x_knots, x)
    idx = jnp.clip(idx, 0, x_knots.shape[0] - 2)
    dims = jnp.arange(x.shape[1])[None, :]
    x0, x1 = x_knots[idx, dims], x_knots[idx + 1, dims]
    u0, u1 = u_knots[idx], u_knots[idx + 1]
    d0, d1 = derivatives[idx, dims], derivatives[idx + 1, dims]
    du = u1 - u0

    # Newton is only used inside a bin. Eight vectorized iterations reach
    # machine precision for the fixed, monotone spline and remain
    # differentiable, unlike bisection through data-dependent branches.
    theta = jnp.clip((x - x0) / (x1 - x0), 0.0, 1.0)
    for _ in range(8):
        x_hat, dx_du = _rqs_forward_bin(theta, x0, x1, du, d0, d1)
        theta = jnp.clip(theta - (x_hat - x) / (dx_du * du), 0.0, 1.0)
    _, dx_du = _rqs_forward_bin(theta, x0, x1, du, d0, d1)
    u = u0 + theta * du

    below = x < x_knots[0][None, :]
    above = x > x_knots[-1][None, :]
    d_left, d_right = derivatives[0][None, :], derivatives[-1][None, :]
    u_below = u_knots[0] + (x - x_knots[0][None, :]) / d_left
    u_above = u_knots[-1] + (x - x_knots[-1][None, :]) / d_right
    u = jnp.where(below, u_below, jnp.where(above, u_above, u))
    dx_du = jnp.where(below, d_left, jnp.where(above, d_right, dx_du))
    logjac = -jnp.sum(jnp.log(dx_du), axis=-1)
    if coordinate_chol is not None:
        u = jax.scipy.linalg.solve_triangular(coordinate_chol, u.T, lower=True).T
        logjac = logjac - jnp.sum(jnp.log(jnp.diag(coordinate_chol)))
    return u, logjac


def quantile_coordinate_forward(
    u: Array, x_knots: Array, u_knots: Array, coordinate_chol: Optional[Array] = None
):
    """Map pilot-normal-score coordinates ``u`` back to physical ``x``.

    Returns ``(x, log|dx/du|)``.
    """
    u = as2d_jax(u)
    logdet_chol = jnp.asarray(0.0, dtype=u.dtype)
    if coordinate_chol is not None:
        u = u @ coordinate_chol.T
        logdet_chol = jnp.sum(jnp.log(jnp.diag(coordinate_chol)))
    derivatives = _rqs_knot_derivatives(x_knots, u_knots)
    idx = jnp.searchsorted(u_knots, u, side="right") - 1
    idx = jnp.clip(idx, 0, u_knots.shape[0] - 2)
    dims = jnp.arange(u.shape[1])[None, :]
    x0, x1 = x_knots[idx, dims], x_knots[idx + 1, dims]
    u0, u1 = u_knots[idx], u_knots[idx + 1]
    d0, d1 = derivatives[idx, dims], derivatives[idx + 1, dims]
    du = u1 - u0
    theta = jnp.clip((u - u0) / du, 0.0, 1.0)
    x, dx_du = _rqs_forward_bin(theta, x0, x1, du, d0, d1)

    below = u < u_knots[0]
    above = u > u_knots[-1]
    d_left, d_right = derivatives[0][None, :], derivatives[-1][None, :]
    x_below = x_knots[0][None, :] + d_left * (u - u_knots[0])
    x_above = x_knots[-1][None, :] + d_right * (u - u_knots[-1])
    x = jnp.where(below, x_below, jnp.where(above, x_above, x))
    dx_du = jnp.where(below, d_left, jnp.where(above, d_right, dx_du))
    return x, jnp.sum(jnp.log(dx_du), axis=-1) + logdet_chol


def _safe_log(x: Array, eps: float = 1e-300) -> Array:
    return jnp.log(jnp.maximum(x, eps))


class PaddedChain(NamedTuple):
    """Fixed-capacity parameters with a traced active length.

    Unused slots are skipped, including their Jacobians. A static scan keeps
    forward and inverse evaluation reverse-mode differentiable.
    """

    params: Any
    length: Array


def stack_chain(raw_chain: List[Any], *, capacity: Optional[int] = None,
                template: Optional[Any] = None) -> Optional[Any]:
    """Stack a list of per-step param dicts into a single pytree with a leading
    axis of size K (number of steps). Returns None for empty chains.

    With ``capacity``, return a PaddedChain instead. ``template`` supplies
    parameter shapes/dtypes for an empty chain; it may contain ShapeDtypeStructs.
    Existing callers without a capacity retain the unpadded representation.
    """
    if capacity is not None:
        if capacity < 1 or len(raw_chain) > capacity:
            raise ValueError("capacity must be positive and at least the chain length")
        if template is None:
            if not raw_chain:
                raise ValueError("an empty padded chain requires a parameter template")
            template = raw_chain[0]
        padding = jax.tree.map(lambda a: jnp.zeros(a.shape, a.dtype), template)
        steps = raw_chain + [padding] * (capacity - len(raw_chain))
        params = jax.tree.map(lambda *a: jnp.stack(a, axis=0), *steps)
        return PaddedChain(params, jnp.asarray(len(raw_chain), dtype=jnp.int32))
    if not raw_chain:
        return None
    return jax.tree.map(lambda *arrays: jnp.stack(arrays, axis=0), *raw_chain)


def chain_forward(model: RealNVPBounded, stacked_chain: Optional[Any], x: Array) -> Tuple[Array, Array]:
    """Apply a chain of flow steps stored as a stacked pytree (leading axis = K).
    Uses lax.scan so the JIT trace graph is O(1) regardless of chain length."""
    x = as2d_jax(x)
    logdet = jnp.zeros((x.shape[0],), dtype=x.dtype)
    if stacked_chain is None:
        return x, logdet

    if isinstance(stacked_chain, PaddedChain):
        def padded_body(carry, inputs):
            i, p = inputs

            def apply_step(state):
                y, ld = model.apply({"params": p}, state[0], method=model.f)
                return y, state[1] + ld

            return jax.lax.cond(i < stacked_chain.length, apply_step,
                                lambda state: state, carry), None

        capacity = jax.tree.leaves(stacked_chain.params)[0].shape[0]
        (x, logdet), _ = jax.lax.scan(
            padded_body, (x, logdet),
            (jnp.arange(capacity), stacked_chain.params),
        )
        return x, logdet

    def body(carry, p):
        x_, ld_acc = carry
        y, ld = model.apply({"params": p}, x_, method=model.f)
        return (y, ld_acc + ld), None

    (x, logdet), _ = jax.lax.scan(body, (x, logdet), stacked_chain)
    return x, logdet


def chain_inverse(model: RealNVPBounded, stacked_chain: Optional[Any], y: Array) -> Tuple[Array, Array]:
    """Invert a chain of flow steps stored as a stacked pytree.
    Uses lax.scan on the reversed chain so the JIT trace graph is O(1)."""
    y = as2d_jax(y)
    logdet = jnp.zeros((y.shape[0],), dtype=y.dtype)
    if stacked_chain is None:
        return y, logdet

    if isinstance(stacked_chain, PaddedChain):
        def padded_body(carry, inputs):
            i, p = inputs

            def apply_step(state):
                x, ld = model.apply({"params": p}, state[0], method=model.finv)
                return x, state[1] + ld

            return jax.lax.cond(i < stacked_chain.length, apply_step,
                                lambda state: state, carry), None

        capacity = jax.tree.leaves(stacked_chain.params)[0].shape[0]
        (y, logdet), _ = jax.lax.scan(
            padded_body, (y, logdet),
            (jnp.arange(capacity), stacked_chain.params), reverse=True,
        )
        return y, logdet

    stacked_rev = jax.tree.map(lambda arr: jnp.flip(arr, axis=0), stacked_chain)

    def body(carry, p):
        y_, ld_acc = carry
        x, ld = model.apply({"params": p}, y_, method=model.finv)
        return (x, ld_acc + ld), None

    (y, logdet), _ = jax.lax.scan(body, (y, logdet), stacked_rev)
    return y, logdet


def component_log_q(
    model: RealNVPBounded,
    x: Array,
    base: Any,
    chain_params: Sequence[Any],
    *,
    whiten: bool = False,
    coordinate_x_knots: Optional[Array] = None,
    coordinate_u_knots: Optional[Array] = None,
    coordinate_chol: Optional[Array] = None,
    coordinate_as_base: bool = False,
) -> Array:
    if coordinate_x_knots is not None:
        u, ld_x_to_u = quantile_coordinate_inverse(
            x, coordinate_x_knots, coordinate_u_knots, coordinate_chol
        )
        u0, ld_flow_inv = chain_inverse(model, chain_params, u)
        if coordinate_as_base:
            log_base = -0.5 * (
                jnp.sum(u0 * u0, axis=-1) + u0.shape[-1] * jnp.log(2.0 * jnp.pi)
            )
            return log_base + ld_flow_inv + ld_x_to_u
        z, ld_u_to_x = quantile_coordinate_forward(
            u0, coordinate_x_knots, coordinate_u_knots, coordinate_chol
        )
        return base.log_prob(z) + ld_flow_inv + ld_x_to_u + ld_u_to_x
    mu = base.mu[None, :]                                      # (1, d)
    x_c = x - mu
    if whiten:
        # The flow acts on u = L^{-1}(x-mu), with Sigma = L L^T.  Evaluating
        # base.log_prob(mu + L z) below contributes the missing -log|det L|,
        # while ld_inv is the inverse flow Jacobian in whitened coordinates.
        L = base.chol()
        x_c = jax.scipy.linalg.solve_triangular(L, x_c.T, lower=True).T
        z_c, ld_inv = chain_inverse(model, chain_params, x_c)
        z = mu + z_c @ L.T
    else:
        z_c, ld_inv = chain_inverse(model, chain_params, x_c)
        z = z_c + mu
    return base.log_prob(z) + ld_inv


def mixture_log_q(
    model: RealNVPBounded,
    x: Array,
    weights: Array,
    bases: Sequence[Any],
    stacked_chains: Sequence[Optional[Any]],
    *,
    whiten: bool = False,
    coordinate_x_knots: Optional[Sequence[Array]] = None,
    coordinate_u_knots: Optional[Array] = None,
    coordinate_chols: Optional[Sequence[Array]] = None,
    coordinate_as_base: bool = False,
) -> Array:
    logw = _safe_log(weights / jnp.sum(weights))
    cols = []
    for j in range(len(stacked_chains)):
        knots_j = None if coordinate_x_knots is None else coordinate_x_knots[j]
        chol_j = None if coordinate_chols is None else coordinate_chols[j]
        cols.append(component_log_q(
            model, x, bases[j], stacked_chains[j], whiten=whiten,
            coordinate_x_knots=knots_j, coordinate_u_knots=coordinate_u_knots,
            coordinate_chol=chol_j, coordinate_as_base=coordinate_as_base,
        ) + logw[j])
    mat = jnp.stack(cols, axis=1)
    return jax.scipy.special.logsumexp(mat, axis=1)


def sample_mixture(
    model: RealNVPBounded,
    key: Array,
    n: int,
    weights: Array,
    bases: Sequence[Any],
    stacked_chains: Sequence[Optional[Any]],
    dtype=jnp.float64,
    *,
    whiten: bool = False,
    coordinate_x_knots: Optional[Sequence[Array]] = None,
    coordinate_u_knots: Optional[Array] = None,
    coordinate_chols: Optional[Sequence[Array]] = None,
    coordinate_as_base: bool = False,
) -> Tuple[Array, Array]:
    J = len(bases)
    weights = weights / jnp.sum(weights)
    key_cat, key_samp = jr.split(key)
    comp = jr.categorical(key_cat, logits=_safe_log(weights), shape=(int(n),)).astype(jnp.int32)

    # Sample n points from every mode, push through its chain, then select
    # per sample using comp.  Avoids Python-list indexing with a traced integer.
    keys_j = jr.split(key_samp, J)
    per_mode = []
    for j in range(J):
        mu_j = bases[j].mu[None, :]                                      # (1, d)
        if coordinate_x_knots is not None:
            if coordinate_as_base:
                u_j = jr.normal(keys_j[j], (n, mu_j.shape[-1]), dtype=dtype)
            else:
                z_j = bases[j].sample(keys_j[j], n, dtype=dtype)
                u_j, _ = quantile_coordinate_inverse(
                    z_j, coordinate_x_knots[j], coordinate_u_knots,
                    None if coordinate_chols is None else coordinate_chols[j],
                )
            y_u, _ = chain_forward(model, stacked_chains[j], u_j)
            x_j, _ = quantile_coordinate_forward(
                y_u, coordinate_x_knots[j], coordinate_u_knots,
                None if coordinate_chols is None else coordinate_chols[j],
            )
        elif whiten:
            L_j = bases[j].chol().astype(dtype)
            z_c = jr.normal(keys_j[j], (n, mu_j.shape[-1]), dtype=dtype)
            y_c, _ = chain_forward(model, stacked_chains[j], z_c)
            x_j = mu_j + y_c @ L_j.T
        else:
            z_j = bases[j].sample(keys_j[j], n, dtype=dtype)
            z_c = z_j - mu_j
            y_c, _ = chain_forward(model, stacked_chains[j], z_c)         # centered
            x_j = y_c + mu_j                                               # un-center
        per_mode.append(x_j)

    X_all = jnp.stack(per_mode, axis=0)                           # (J, n, d)
    x = X_all[comp, jnp.arange(int(n))]                           # (n, d)
    return x, comp
