# nets.py
from __future__ import annotations

import math
from typing import Optional, Tuple

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from flax import linen as nn


Array = jnp.ndarray


def odd_even_split_scheme(
    dim: int,
) -> Tuple[Tuple[int, Tuple[int, ...]], ...]:
    """One RealNVP scheme separating zero-based even and odd coordinates.

    The returned permutation places ``(0, 2, 4, ...)`` in the first block
    and ``(1, 3, 5, ...)`` in the second.  ``RealNVPBounded`` alternates the
    conditioning direction across successive layers, so every adjacent pair
    in a chain-structured target can directly interact in either direction.
    """
    dim = int(dim)
    if dim < 2:
        raise ValueError("odd/even splitting requires dim >= 2")
    even = tuple(range(0, dim, 2))
    odd = tuple(range(1, dim, 2))
    return ((len(even), even + odd),)


def _inverse_softplus(x: float) -> float:
    """Stable scalar inverse of softplus, used only for initialization."""
    return math.log(math.expm1(float(x)))


def _rqs_bin_forward(
    theta: Array,
    y0: Array,
    y1: Array,
    dx: Array,
    d0: Array,
    d1: Array,
) -> Tuple[Array, Array]:
    """Rational-quadratic bin value and derivative ``dy/dx``.

    ``theta`` is the normalized position in an input bin of width ``dx``;
    ``y0``/``y1`` are the corresponding output knots and ``d0``/``d1`` are
    positive knot derivatives.  This is the scalar monotone spline from
    Durkan et al. (2019), vectorized over samples and dimensions.
    """
    height = y1 - y0
    delta = height / dx
    omt = 1.0 - theta
    theta_omt = theta * omt
    denom = delta + (d1 + d0 - 2.0 * delta) * theta_omt
    numer = height * (delta * theta * theta + d0 * theta_omt)
    y = y0 + numer / denom
    deriv_numer = delta * delta * (
        d1 * theta * theta
        + 2.0 * delta * theta_omt
        + d0 * omt * omt
    )
    return y, deriv_numer / (denom * denom)


# ============================================================
# Smooth bump clamp (JAX, no in-place)
# ============================================================

def smoothstep5(t: Array) -> Array:
    # Quintic smoothstep: C2, slope 0 at 0 and 1
    return t * t * t * (10.0 - 15.0 * t + 6.0 * t * t)


def bump_clip(x: Array, B: float, delta: float) -> Array:
    """
    C^2 bump clamp.
    - Identity for |x| <= B
    - Constant = sign(x)*B for |x| >= B + delta
    - Smooth transition in [B, B+delta]
    """
    B = float(B)
    delta = float(delta)

    if delta <= 0.0:
        return jnp.clip(x, -B, B)

    s = jnp.abs(x)
    sign = jnp.sign(x)

    inside = s <= B
    outside = s >= (B + delta)
    mid = (~inside) & (~outside)

    # mid formula (only matters on mid region)
    sm = s
    t = (sm - B) / delta
    w = smoothstep5(t)
    val = (1.0 - w) * sm + w * B
    mid_y = sign * val

    y = jnp.where(inside, x, jnp.where(outside, sign * B, mid_y))
    return y


# ============================================================
# MLP (Flax) with zero-init final layer
# ============================================================

class MLP(nn.Module):
    in_dim: int
    out_dim: int
    hidden: int = 128
    depth: int = 3
    act: str = "relu"

    @nn.compact
    def __call__(self, x: Array) -> Array:
        if self.depth < 1:
            raise ValueError("MLP depth must be >= 1")

        def activation(u: Array) -> Array:
            if self.act == "relu":
                return nn.relu(u)
            elif self.act == "tanh":
                return jnp.tanh(u)
            elif self.act == "gelu":
                return nn.gelu(u)
            else:
                raise ValueError(f"Unknown activation: {self.act}")

        # depth-1 hidden layers
        for i in range(self.depth - 1):
            x = nn.Dense(self.hidden, name=f"dense_{i}")(x)
            x = activation(x)

        # final layer: zero init => starts as near-identity coupling
        x = nn.Dense(
            self.out_dim,
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,
            name="dense_out",
        )(x)
        return x


# ============================================================
# Unconditional marginal rational-quadratic spline (Flax)
# ============================================================

class MarginalRationalQuadraticSpline(nn.Module):
    """Independent trainable monotone spline for every coordinate.

    The input knots are fixed and equally spaced on ``[-bound, bound]``.
    Trainable positive output-bin heights and interior derivatives determine
    the warp; outside the interval the map has identity tails.  All raw
    parameters are initialized to zero, which makes the complete transform
    exactly identity.  This gives an incremental flow a direct way to learn
    skewed one-dimensional marginals without manufacturing them through
    cross-coordinate conditioning in an affine coupling layer.
    """

    dim: int
    n_bins: int = 8
    bound: float = 6.0
    min_bin_size: float = 1e-3
    min_derivative: float = 1e-3
    inverse_iters: int = 12

    def setup(self):
        if self.dim < 1:
            raise ValueError("MarginalRationalQuadraticSpline requires dim >= 1.")
        if self.n_bins < 2:
            raise ValueError("n_bins must be >= 2.")
        if self.bound <= 0.0:
            raise ValueError("bound must be > 0.")
        if self.min_bin_size <= 0.0 or self.n_bins * self.min_bin_size >= 2.0 * self.bound:
            raise ValueError("min_bin_size must be positive and leave nonzero spline range.")
        if not (0.0 < self.min_derivative < 1.0):
            raise ValueError("min_derivative must lie in (0, 1).")

        self.raw_heights = self.param(
            "raw_heights", nn.initializers.zeros, (self.dim, self.n_bins)
        )
        self.raw_derivatives = self.param(
            "raw_derivatives", nn.initializers.zeros, (self.dim, self.n_bins - 1)
        )

    def _spline_params(self, dtype) -> Tuple[Array, Array, Array]:
        total = jnp.asarray(2.0 * self.bound, dtype=dtype)
        min_bin = jnp.asarray(self.min_bin_size, dtype=dtype)
        heights = min_bin + (total - self.n_bins * min_bin) * nn.softmax(
            self.raw_heights.astype(dtype), axis=-1
        )
        y_knots = jnp.concatenate(
            [
                jnp.full((self.dim, 1), -self.bound, dtype=dtype),
                -self.bound + jnp.cumsum(heights, axis=-1),
            ],
            axis=-1,
        )

        derivative_offset = _inverse_softplus(1.0 - self.min_derivative)
        interior = self.min_derivative + nn.softplus(
            self.raw_derivatives.astype(dtype) + derivative_offset
        )
        derivatives = jnp.concatenate(
            [
                jnp.ones((self.dim, 1), dtype=dtype),
                interior,
                jnp.ones((self.dim, 1), dtype=dtype),
            ],
            axis=-1,
        )
        x_knots = jnp.linspace(
            -self.bound, self.bound, self.n_bins + 1, dtype=dtype
        )
        return x_knots, y_knots, derivatives

    def __call__(self, x: Array) -> Tuple[Array, Array]:
        """Forward marginal warp, returning ``(y, log|det dy/dx|)``."""
        x_knots, y_knots, derivatives = self._spline_params(x.dtype)
        dx = x_knots[1] - x_knots[0]
        idx = jnp.floor((x + self.bound) / dx).astype(jnp.int32)
        idx = jnp.clip(idx, 0, self.n_bins - 1)
        dims = jnp.arange(self.dim)[None, :]

        x0 = x_knots[idx]
        theta = jnp.clip((x - x0) / dx, 0.0, 1.0)
        y0 = y_knots[dims, idx]
        y1 = y_knots[dims, idx + 1]
        d0 = derivatives[dims, idx]
        d1 = derivatives[dims, idx + 1]
        y_inside, dy_dx = _rqs_bin_forward(theta, y0, y1, dx, d0, d1)

        outside = (x <= -self.bound) | (x >= self.bound)
        y = jnp.where(outside, x, y_inside)
        dy_dx = jnp.where(outside, jnp.ones_like(dy_dx), dy_dx)
        return y, jnp.sum(jnp.log(dy_dx), axis=-1)

    def inverse(self, y: Array) -> Tuple[Array, Array]:
        """Inverse marginal warp, returning ``(x, log|det dx/dy|)``."""
        x_knots, y_knots, derivatives = self._spline_params(y.dtype)
        dx = x_knots[1] - x_knots[0]

        # Each dimension has its own learned output knots.  searchsorted is
        # vmapped over dimensions and remains compatible with JIT/vmap.
        idx = jax.vmap(
            lambda knots_j, values_j: jnp.searchsorted(
                knots_j, values_j, side="right"
            ) - 1,
            in_axes=(0, 1),
            out_axes=1,
        )(y_knots, y)
        idx = jnp.clip(idx, 0, self.n_bins - 1)
        dims = jnp.arange(self.dim)[None, :]

        x0 = x_knots[idx]
        y0 = y_knots[dims, idx]
        y1 = y_knots[dims, idx + 1]
        d0 = derivatives[dims, idx]
        d1 = derivatives[dims, idx + 1]

        # A fixed-count Newton solve avoids data-dependent control flow and is
        # differentiable.  The monotone bin and clipped iterate make it robust
        # at boundaries; twelve iterations reach float64 precision in tests.
        theta = jnp.clip((y - y0) / (y1 - y0), 0.0, 1.0)
        for _ in range(int(self.inverse_iters)):
            y_hat, dy_dx = _rqs_bin_forward(theta, y0, y1, dx, d0, d1)
            theta = jnp.clip(theta - (y_hat - y) / (dy_dx * dx), 0.0, 1.0)
        _, dy_dx = _rqs_bin_forward(theta, y0, y1, dx, d0, d1)
        x_inside = x0 + theta * dx

        outside = (y <= -self.bound) | (y >= self.bound)
        x = jnp.where(outside, y, x_inside)
        dy_dx = jnp.where(outside, jnp.ones_like(dy_dx), dy_dx)
        return x, -jnp.sum(jnp.log(dy_dx), axis=-1)


# ============================================================
# AffineCouplingBounded (Flax)
# ============================================================

class AffineCouplingBounded(nn.Module):
    dim: int
    hidden: int = 128
    depth: int = 3
    s_max: float = 3.0
    s_bump: float = 1.0
    t_max: float = float("inf")
    t_bump: float = 1.0
    split: Optional[int] = None
    """Size of the first block (x[:, :split]); the second block is
    dim - split. None (default) reproduces the original dim//2 split
    exactly -- same param shapes, same computation."""
    swap: bool = False
    """False: first block conditions, second is transformed (original
    behavior). True: roles reversed -- second block conditions, first is
    transformed. Coordinates are never reordered within a block, only
    which whole block plays conditioner vs. transformed."""
    perm: Optional[Tuple[int, ...]] = None
    """Fixed permutation of range(dim), applied before the split/swap logic
    above and undone after -- lets `split` express a non-contiguous block
    (e.g. {theta, phi_2} vs. {phi_1}) by permuting it into a contiguous
    prefix/suffix first. Block 1 in *permuted* coordinates is
    `perm[:split]`. None (default): identity, original coordinate order."""

    def setup(self):
        if self.dim < 2:
            raise ValueError("AffineCoupling requires dim >= 2.")
        s = self.dim // 2 if self.split is None else self.split
        if not (0 < s < self.dim):
            raise ValueError("split must satisfy 0 < split < dim.")
        self._s = s
        if self.perm is not None:
            if sorted(self.perm) != list(range(self.dim)):
                raise ValueError("perm must be a permutation of range(dim).")
            inv = [0] * self.dim
            for j, p in enumerate(self.perm):
                inv[p] = j
            self._perm = jnp.array(self.perm)
            self._inv_perm = jnp.array(inv)
        else:
            self._perm = None
            self._inv_perm = None
        cond_size, trans_size = (s, self.dim - s) if not self.swap else (self.dim - s, s)
        self.s_net = MLP(cond_size, trans_size, hidden=self.hidden, depth=self.depth, name="s_net")
        self.t_net = MLP(cond_size, trans_size, hidden=self.hidden, depth=self.depth, name="t_net")

    def _to_canon(self, x: Array) -> Array:
        return x if self._perm is None else x[:, self._perm]

    def _from_canon(self, x: Array) -> Array:
        return x if self._inv_perm is None else x[:, self._inv_perm]

    def _bound(self, s_raw: Array, t_raw: Array) -> Tuple[Array, Array]:
        # bound translation. `t_max`/`s_max` are static Python floats (module
        # config, not traced arrays), so branch in plain Python: `jnp.where`
        # would evaluate `bump_clip(..., B=inf, ...)` eagerly on the untaken
        # branch even when isinf(t_max), and B=inf makes bump_clip produce a
        # bare (masked-out-in-value-but-not-in-gradient) NaN -- a classic
        # jnp.where NaN-gradient trap that silently NaNs every training step.
        t = t_raw if math.isinf(self.t_max) else bump_clip(t_raw, B=self.t_max, delta=self.t_bump)
        log_s = s_raw if math.isinf(self.s_max) else bump_clip(s_raw, B=self.s_max, delta=self.s_bump)
        return log_s, t

    def __call__(self, x: Array) -> Tuple[Array, Array]:
        """
        Forward coupling: (x -> y), returns (y, logdet) with logdet shape (N,)
        x: (N, dim)
        """
        xc = self._to_canon(x)
        s = self._s
        c, x2 = (xc[:, :s], xc[:, s:]) if not self.swap else (xc[:, s:], xc[:, :s])

        log_s, t = self._bound(self.s_net(c), self.t_net(c))

        y2 = x2 * jnp.exp(log_s) + t
        yc = jnp.concatenate([c, y2], axis=-1) if not self.swap else jnp.concatenate([y2, c], axis=-1)
        y = self._from_canon(yc)
        logdet = jnp.sum(log_s, axis=-1)
        return y, logdet

    def inverse(self, y: Array) -> Tuple[Array, Array]:
        """
        Inverse coupling: (y -> x), returns (x, logdet_inv) with logdet_inv shape (N,)
        """
        yc = self._to_canon(y)
        s = self._s
        c, y2 = (yc[:, :s], yc[:, s:]) if not self.swap else (yc[:, s:], yc[:, :s])  # same split as forward

        log_s, t = self._bound(self.s_net(c), self.t_net(c))

        x2 = (y2 - t) * jnp.exp(-log_s)
        xc = jnp.concatenate([c, x2], axis=-1) if not self.swap else jnp.concatenate([x2, c], axis=-1)
        x = self._from_canon(xc)
        logdet = -jnp.sum(log_s, axis=-1)
        return x, logdet


# ============================================================
# DiagonalAffineBounded (Flax)
# ============================================================

class DiagonalAffineBounded(nn.Module):
    """Unconditional per-dimension affine flow: `y = x * exp(log_s) + t`,
    with `log_s`/`t` *global* learnable parameters (not functions of `x`,
    unlike `AffineCouplingBounded`'s conditional scale/shift networks),
    zero-initialized so the flow starts at the exact identity map.

    Because `log_s` doesn't depend on `x`, `logdet = sum(log_s)` is a single
    scalar shared by *every* particle in a batch (broadcast to shape `(N,)`
    below to match `RealNVPBounded`'s `f`/`finv` interface) -- adding the
    same constant to every entry of a softmax input leaves the resulting
    distribution unchanged, so this flow family structurally cannot skew
    the *relative* self-normalized-importance weights of a particle batch,
    unlike a general conditional coupling flow (`RealNVPBounded`), whose
    per-particle-varying log-det can. `src/samplers/aft.py` uses this flow
    (rather than `RealNVPBounded`, which every other flow baseline in this
    repo shares) specifically to sidestep that failure mode, matching
    Arbel et al. 2021's own choice of flow family for their highest-
    dimensional (d=1600) example -- see that paper's Sec. 6.4 and this
    repo's `aft.py` module docstring for the mechanism this avoids.
    """
    dim: int
    s_max: float = 3.0
    s_bump: float = 1.0
    t_max: float = float("inf")
    t_bump: float = 1.0

    def setup(self):
        self.log_s_param = self.param("log_s", nn.initializers.zeros, (self.dim,))
        self.t_param = self.param("t", nn.initializers.zeros, (self.dim,))

    def _bound(self) -> Tuple[Array, Array]:
        t = self.t_param if math.isinf(self.t_max) else bump_clip(self.t_param, B=self.t_max, delta=self.t_bump)
        log_s = self.log_s_param if math.isinf(self.s_max) else bump_clip(self.log_s_param, B=self.s_max, delta=self.s_bump)
        return log_s, t

    def f(self, x: Array) -> Tuple[Array, Array]:
        """Forward: x -> z. Returns (z, logdet) where logdet is (N,)."""
        log_s, t = self._bound()
        z = x * jnp.exp(log_s)[None, :] + t[None, :]
        logdet = jnp.full((x.shape[0],), jnp.sum(log_s), dtype=x.dtype)
        return z, logdet

    def finv(self, z: Array) -> Tuple[Array, Array]:
        """Inverse: z -> x. Returns (x, logdet_inv) where logdet_inv is (N,)."""
        log_s, t = self._bound()
        x = (z - t[None, :]) * jnp.exp(-log_s)[None, :]
        logdet = jnp.full((z.shape[0],), -jnp.sum(log_s), dtype=z.dtype)
        return x, logdet


# ============================================================
# RealNVPBounded (Flax)
# ============================================================

class RealNVPBounded(nn.Module):
    dim: int
    n_coupling: int = 6
    hidden: int = 128
    depth: int = 3
    s_max: float = float("inf")
    s_bump: float = 1.0
    t_max: float = float("inf")
    t_bump: float = 1.0
    marginal_spline: bool = False
    """Prepend an unconditional coordinatewise rational-quadratic spline.
    It is identity-initialized and gives each incremental map a direct
    marginal-skewness transform before the conditional affine couplings."""
    marginal_spline_bins: int = 8
    marginal_spline_bound: float = 6.0
    marginal_spline_min_bin: float = 1e-3
    marginal_spline_min_derivative: float = 1e-3
    split_at: Optional[int] = None
    """Fixed block boundary for semantic coupling (e.g. `theta_len` to split
    theta|phi as whole blocks) instead of the default dim//2 index split.
    When set (and `split_schemes` is None), every layer partitions x into
    x[:, :split_at] / x[:, split_at:] and alternates *which whole block*
    conditions the other, coordinates are never reordered within a block
    (unlike the default's flip-based mixing). Equivalent to
    `split_schemes=((split_at, None),)`. Ignored if `split_schemes` is set.
    When both are None (default), reproduces the original dim//2 +
    full-array-flip mixing exactly."""
    split_schemes: Optional[Tuple[Tuple[int, Optional[Tuple[int, ...]]], ...]] = None
    """Interleave multiple semantic block partitions across layers instead
    of one fixed boundary. Each entry is `(split, perm)` -- see
    `AffineCouplingBounded.split`/`.perm` (`perm=None` for a plain
    contiguous split at `split`, e.g. to express a non-contiguous block like
    {theta, phi_2} vs. {phi_1}). Layer `i` uses scheme `i % len(schemes)`
    with `swap = (i // len(schemes)) % 2 == 1`, so with `n_coupling` a
    multiple of `2 * len(schemes)` every scheme is used in both directions
    equally often. Takes priority over `split_at` when both are set."""

    def _schemes(self) -> Optional[Tuple[Tuple[int, Optional[Tuple[int, ...]]], ...]]:
        if self.split_schemes is not None:
            return self.split_schemes
        if self.split_at is not None:
            return ((self.split_at, None),)
        return None

    def setup(self):
        if self.dim < 2:
            raise ValueError("RealNVPBounded requires dim >= 2.")
        self.marginal = (
            MarginalRationalQuadraticSpline(
                dim=self.dim,
                n_bins=self.marginal_spline_bins,
                bound=self.marginal_spline_bound,
                min_bin_size=self.marginal_spline_min_bin,
                min_derivative=self.marginal_spline_min_derivative,
                name="marginal_spline",
            )
            if self.marginal_spline
            else None
        )
        schemes = self._schemes()
        if schemes is not None:
            for split, perm in schemes:
                if not (0 < split < self.dim):
                    raise ValueError("each scheme's split must satisfy 0 < split < dim.")
                if perm is not None and len(perm) != self.dim:
                    raise ValueError("each scheme's perm must have length dim.")
        n = len(schemes) if schemes is not None else 1
        self.blocks = [
            AffineCouplingBounded(
                dim=self.dim,
                hidden=self.hidden,
                depth=self.depth,
                s_max=self.s_max,
                s_bump=self.s_bump,
                t_max=self.t_max,
                t_bump=self.t_bump,
                split=(schemes[i % n][0] if schemes is not None else None),
                perm=(schemes[i % n][1] if schemes is not None else None),
                swap=(schemes is not None and (i // n) % 2 == 1),
                name=f"blk_{i}",
            )
            for i in range(self.n_coupling)
        ]

    def f(self, x: Array) -> Tuple[Array, Array]:
        """
        Forward: x -> z
        Returns (z, logdet) where logdet is (N,)
        """
        z = x
        logdet = jnp.zeros((x.shape[0],), dtype=x.dtype)
        if self.marginal is not None:
            z, ld = self.marginal(z)
            logdet = logdet + ld
        use_flip = self._schemes() is None

        for i, blk in enumerate(self.blocks):
            if use_flip and (i % 2) == 1:
                z = jnp.flip(z, axis=-1)
                z, ld = blk(z)
                z = jnp.flip(z, axis=-1)
            else:
                z, ld = blk(z)
            logdet = logdet + ld

        return z, logdet

    def finv(self, z: Array) -> Tuple[Array, Array]:
        """
        Inverse: z -> x
        Returns (x, logdet_inv) where logdet_inv is (N,)
        """
        x = z
        logdet = jnp.zeros((z.shape[0],), dtype=z.dtype)
        use_flip = self._schemes() is None

        for i, blk in reversed(list(enumerate(self.blocks))):
            if use_flip and (i % 2) == 1:
                x = jnp.flip(x, axis=-1)
                x, ld = blk.inverse(x)
                x = jnp.flip(x, axis=-1)
            else:
                x, ld = blk.inverse(x)
            logdet = logdet + ld

        if self.marginal is not None:
            x, ld = self.marginal.inverse(x)
            logdet = logdet + ld

        return x, logdet
