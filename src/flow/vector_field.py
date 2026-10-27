# vector_field.py
"""General time-conditioned vector field shared by `cat` and `cnf`.

    v_x(beta, x) = L_j f(u, beta),      u = L_j^{-1}(x - mu_j),

with f an MLP (plus an optional per-coordinate cubic-spline term) on the
standardized coordinates u and a smooth embedding of the time beta. Both
methods use exactly this parametrization (same layers, same initialization,
same exact divergence), so they differ only in how the field is trained.

The output layer is zero-initialized: the initial field is exactly zero, the
first rollout is the identity flow and q_beta^j = b_j at every beta.
The divergence is exact and cheap: div_x v = tr(L (df/du) L^{-1}) = tr(df/du),
obtained by carrying du-Jacobians forward through the layers in one pass.
"""
from __future__ import annotations

from typing import Callable, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

Array = jax.Array


def _activation(name: str) -> Callable[[Array], Array]:
    if name == "tanh":
        return jnp.tanh
    if name == "softplus":
        return nn.softplus
    if name == "silu":
        return nn.silu
    raise ValueError(
        f"Unknown activation {name!r}; the exact divergence propagates activation "
        "derivatives, so only smooth activations (tanh, softplus, silu) are allowed."
    )


def _activation_derivs(name: str) -> Callable[[Array], Tuple[Array, Array, Array]]:
    """(sigma(z), sigma'(z), sigma''(z)) for the smooth activations."""
    if name == "silu":
        def f(z):
            s = jax.nn.sigmoid(z)
            ds = s * (1.0 - s)
            return z * s, s + z * ds, ds * (2.0 + z * (1.0 - 2.0 * s))
    elif name == "tanh":
        def f(z):
            t = jnp.tanh(z)
            dt = 1.0 - t * t
            return t, dt, -2.0 * t * dt
    elif name == "softplus":
        def f(z):
            s = jax.nn.sigmoid(z)
            return nn.softplus(z), s, s * (1.0 - s)
    else:
        _activation(name)   # raises
    return f


def beta_features(beta: Array) -> Array:
    """Fixed smooth embedding of the annealing time."""
    beta = jnp.asarray(beta)
    return jnp.stack([beta, beta * beta, jnp.sin(jnp.pi * beta), jnp.cos(jnp.pi * beta)])


def _bspline3(t: Array) -> Tuple[Array, Array, Array]:
    """Uniform cubic B-spline kernel B(t) (support |t| < 2) and B', B''."""
    a = jnp.abs(t)
    r = 2.0 - a
    inner, outer = a < 1.0, (a >= 1.0) & (a < 2.0)
    b = jnp.where(inner, 2.0 / 3.0 - a * a + 0.5 * a ** 3, jnp.where(outer, r ** 3 / 6.0, 0.0))
    db = jnp.where(inner, -2.0 * t + 1.5 * t * a, jnp.where(outer, -0.5 * jnp.sign(t) * r * r, 0.0))
    d2b = jnp.where(inner, -2.0 + 3.0 * a, jnp.where(outer, r, 0.0))
    return b, db, d2b


def _spline_coef(params, beta_feat: Array, act: Callable, d: int) -> Array:
    """(d, M) spline coefficients from the small beta network (see VectorMLP)."""
    z = act(beta_feat @ params["spline_k0"] + params["spline_b0"])
    return (z @ params["spline_k1"] + params["spline_b1"]).reshape(d, -1)


def _spline_vec(u, coef, bound):
    """s_i(u_i) = sum_m coef[i, m] B((u_i - c_m) / h) and ds_i/du_i."""
    M = coef.shape[-1]
    h = 2.0 * bound / (M - 1)
    c = jnp.linspace(-bound, bound, M, dtype=u.dtype)
    b, db, _ = _bspline3((u[..., :, None] - c) / h)
    return jnp.sum(coef * b, -1), jnp.sum(coef * db, -1) / h


class VectorMLP(nn.Module):
    """Vector field f(u, beta) in R^d on standardized coordinates u.

    spline_knots > 0 adds f_i += s_i(beta, u_i), a cubic B-spline in every
    coordinate on [-spline_bound, spline_bound] whose coefficients come from a
    small beta network (zero-initialized output): it can bend a coordinate
    sharply (e.g. one-sided skew) without the MLP having to.

    gradient=True makes the MLP a scalar potential phi(u, beta) instead; the
    velocity is then the physical gradient grad_x phi (curl-free, so the
    continuity equation fixes it uniquely) and the divergence is the
    Laplacian. No spline term.
    """
    hidden: int = 128
    depth: int = 3
    act: str = "silu"
    spline_knots: int = 0
    spline_bound: float = 6.0
    spline_hidden: int = 32
    gradient: bool = False

    @nn.compact
    def __call__(self, u: Array, beta_feat: Array) -> Array:
        act = _activation(self.act)
        d = u.shape[-1]
        z = jnp.concatenate([u, beta_feat], axis=-1)
        for i in range(self.depth):
            z = act(nn.Dense(self.hidden, name=f"dense_{i}")(z))
        out = nn.Dense(1 if self.gradient else d, kernel_init=nn.initializers.zeros,
                       bias_init=nn.initializers.zeros, name="dense_out")(z)
        if self.gradient:
            return out[..., 0]
        if self.spline_knots:
            H, M, nb = self.spline_hidden, self.spline_knots, beta_feat.shape[-1]
            zeros = nn.initializers.zeros
            p = dict(
                spline_k0=self.param("spline_k0", nn.initializers.lecun_normal(), (nb, H)),
                spline_b0=self.param("spline_b0", zeros, (H,)),
                spline_k1=self.param("spline_k1", zeros, (H, d * M)),
                spline_b1=self.param("spline_b1", zeros, (d * M,)),
            )
            out = out + _spline_vec(u, _spline_coef(p, beta_feat, act, d), self.spline_bound)[0]
        return out


def count_params(params) -> int:
    return int(sum(np.prod(a.shape) for a in jax.tree_util.tree_leaves(params)))


def make_vec_fns(model: VectorMLP, field_dtype=None):
    """Per-particle (v_x, div_x v) for one mode's (params, mu, chol).

    Returns (shared, per_beta): both are vmapped over particles x (N, d);
    `shared` takes one scalar beta, `per_beta` one beta per particle (N,).
    field_dtype: dtype of the network evaluation (params must match). Inputs
    are cast down to it and outputs cast back to the caller's dtype.
    """
    act3 = _activation_derivs(model.act)
    act = _activation(model.act)
    depth, knots, bound = int(model.depth), int(model.spline_knots), float(model.spline_bound)

    gradient = bool(model.gradient)
    if gradient and knots:
        raise ValueError("gradient fields do not support spline_knots")

    def vel_div_grad(params, x, beta, mu, chol):
        # Forward Laplacian: (z, dz/dx, Lap z) through the layers. du/dx = L^{-1} is
        # constant, so it does not depend on the particle (vmap leaves it unbatched).
        out_dtype = x.dtype
        if field_dtype is not None:
            x, beta, mu, chol = (jnp.asarray(a).astype(field_dtype) for a in (x, beta, mu, chol))
        d = x.shape[0]
        Linv = jax.scipy.linalg.solve_triangular(chol, jnp.eye(d, dtype=x.dtype), lower=True)
        u = Linv @ (x - mu)
        k0, b0 = params["dense_0"]["kernel"], params["dense_0"]["bias"]
        z = u @ k0[:d] + beta_features(beta) @ k0[d:] + b0
        J = k0[:d].T @ Linv                                     # (H, d) = dz/dx
        lap = jnp.zeros_like(z)
        for i in range(depth):
            if i > 0:
                k, b = params[f"dense_{i}"]["kernel"], params[f"dense_{i}"]["bias"]
                z = a @ k + b
                J = k.T @ J
                lap = lap @ k
            a, d1, d2 = act3(z)
            lap = d1 * lap + d2 * jnp.sum(J * J, axis=-1)
            J = d1[:, None] * J
        w = params["dense_out"]["kernel"][:, 0]
        return (J.T @ w).astype(out_dtype), jnp.dot(lap, w).astype(out_dtype)

    def vel_div(params, x, beta, mu, chol):
        out_dtype = x.dtype
        if field_dtype is not None:
            x, beta, mu, chol = (jnp.asarray(a).astype(field_dtype) for a in (x, beta, mu, chol))
        d = x.shape[0]
        u = jax.scipy.linalg.solve_triangular(chol, x - mu, lower=True)
        k0, b0 = params["dense_0"]["kernel"], params["dense_0"]["bias"]
        z = u @ k0[:d] + beta_features(beta) @ k0[d:] + b0
        J = k0[:d].T                                            # (H, d) = dz/du
        for i in range(depth):
            if i > 0:
                k, b = params[f"dense_{i}"]["kernel"], params[f"dense_{i}"]["bias"]
                z = a @ k + b
                J = k.T @ J
            a, d1, _ = act3(z)
            J = d1[:, None] * J
        W, bo = params["dense_out"]["kernel"], params["dense_out"]["bias"]     # (H, d)
        f = a @ W + bo
        div = jnp.sum(W * J)                                    # tr(W^T J)
        if knots:
            s, ds = _spline_vec(u, _spline_coef(params, beta_features(beta), act, d), bound)
            f, div = f + s, div + jnp.sum(ds)
        return (chol @ f).astype(out_dtype), div.astype(out_dtype)

    fn = vel_div_grad if gradient else vel_div
    return (jax.vmap(fn, in_axes=(None, 0, None, None, None)),
            jax.vmap(fn, in_axes=(None, 0, 0, None, None)))
