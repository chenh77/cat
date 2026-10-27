# cat.py
"""CAT: per-mode (component-wise) continuous annealing transport.

Given the modes (mu_j, Sigma_j, a_j) of a shared mode discovery, CAT transports
each component b_j = N(mu_j, Sigma_j) of the Laplace mixture b with one
time-conditioned *general* vector field per mode,

    v_j(beta, x) = L_j f_j(u, beta),     u = L_j^{-1}(x - mu_j),

(`vector_field.VectorMLP`, the same parametrization as `cnf`),
integrated in beta along the fixed local annealing path

    gamma_beta^j(x) = b_j(x) exp{beta [log a_j + h(x)]},
    h(x) = log p~(x) - log b(x),   b(x) = sum_j a_j b_j(x).

The network sees standardized coordinates u but the velocity is the *physical*
one, so the transport metric does not depend on Sigma_j. Particles and their
log densities are advanced by the same fixed-step RK4 stages,

    dX/dbeta = v,       d log q / dbeta = -div_x v,

with the exact divergence (one forward pass through the MLP).

Training minimizes the squared continuity residual
    R = div v + s_beta . v + h - c_eta(beta),     s_beta = grad_x log b_j + beta grad_x h,
with a jointly trained scalar network c_eta(beta) per mode (the parametrized
log-normalizer rate, c ~ E_{p_beta}[h]), at the particles of the frozen-flow
rollout at every checkpoint. For a general field the continuity equation has
many solutions; `velocity_l2_weight` adds lambda * E|v|^2, which selects the
minimal-kinetic-energy one (the gradient field).

The beta grid is adaptive: with `adaptive_target_ess`, every
refresh places the checkpoints greedily so that each step has (about) the
target worst-mode incremental relative ESS, measured on the latest rollout of
the current field (`_ess_betas`). The same grid is used for the ODE
integration in training, validation and the final sampling; the number of
checkpoints K is an output. A fixed grid (`betas=...`) is still supported.

The outer loop alternates: freeze field -> place grid -> fresh rollout from
b_j -> collocation pool -> `train_steps_per_refresh` optimizer steps -> fresh
validation rollout for diagnostics. Mixture weights are calibrated at the end
from endpoint importance weights (delayed weight calibration),
log C_j = logmeanexp(log gamma_1^j - log q_1^j).
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jr
import flax.linen as nn
import numpy as np
import optax
from jax.scipy.special import logsumexp

from .compile_timer import compile_seconds
from ..mode_explore.explore import Mode
from ..flow.vector_field import VectorMLP, _activation, beta_features, make_vec_fns
from ..utils.numeric import sym_jax, chol_jitter_jax

Array = jax.Array
LogPiBatch = Callable[[Array], Array]            # (N,d) -> (N,)
LogPiSingle = Callable[[Array], Array]           # (d,) -> scalar
GradLogPiSingle = Callable[[Array], Array]       # (d,) -> (d,)

_LOG_2PI = math.log(2.0 * math.pi)


# ============================================================
# Normalizer-rate network
# ============================================================
class RateMLP(nn.Module):
    """Scalar normalizer-rate network c_eta(beta) ~ E_{p_beta^j}[h] .

    Output is `scale * MLP(beta)` with a zero-initialized last layer; `scale`
    is a fixed preconditioner (the spread of h on the first collocation pool)
    so Adam does not need thousands of steps to reach the O(|h|) range. The
    network itself is still trained jointly with the potential.
    """
    hidden: int = 32
    depth: int = 2
    act: str = "silu"

    @nn.compact
    def __call__(self, beta_feat: Array, scale: Array) -> Array:
        act = _activation(self.act)
        z = beta_feat
        for i in range(self.depth):
            z = act(nn.Dense(self.hidden, name=f"dense_{i}")(z))
        out = nn.Dense(
            1, kernel_init=nn.initializers.zeros, bias_init=nn.initializers.zeros,
            name="dense_out",
        )(z)
        return scale * out[..., 0]




# ============================================================
# ODE: fixed-step RK4 in beta, particles and log density together
# ============================================================
def _rk4_rollout(vel_lap, params, x0, logq0, betas, mu, chol, substeps: int):
    """Integrate (x, log q) from betas[0] to betas[-1] for one mode.

    Returns checkpoint states xs (K, n, d) and logqs (K, n), including the
    initial state. Works for decreasing `betas` as well (inverse ODE).
    """

    def f(beta, x):
        v, lap = vel_lap(params, x, beta, mu, chol)
        return v, -lap

    def rk4_step(state, beta, dt):
        x, lq = state
        k1x, k1l = f(beta, x)
        k2x, k2l = f(beta + 0.5 * dt, x + 0.5 * dt * k1x)
        k3x, k3l = f(beta + 0.5 * dt, x + 0.5 * dt * k2x)
        k4x, k4l = f(beta + dt, x + dt * k3x)
        x = x + dt / 6.0 * (k1x + 2.0 * k2x + 2.0 * k3x + k4x)
        lq = lq + dt / 6.0 * (k1l + 2.0 * k2l + 2.0 * k3l + k4l)
        return x, lq

    def interval(state, b01):
        b0, b1 = b01[0], b01[1]
        dt = (b1 - b0) / substeps

        def sub(s, i):
            return rk4_step(s, b0 + i * dt, dt), None

        state, _ = jax.lax.scan(sub, state, jnp.arange(substeps, dtype=x0.dtype))
        return state, state

    pairs = jnp.stack([betas[:-1], betas[1:]], axis=1)
    _, (xs, lqs) = jax.lax.scan(interval, (x0, logq0), pairs)
    xs = jnp.concatenate([x0[None], xs], axis=0)
    lqs = jnp.concatenate([logq0[None], lqs], axis=0)
    return xs, lqs


def _gauss_logpdf(x: Array, mu: Array, chol: Array) -> Array:
    """log N(x | mu, L L^T), x: (..., d)."""
    y = jax.scipy.linalg.solve_triangular(
        chol, (x - mu).reshape(-1, mu.shape[0]).T, lower=True
    ).T.reshape(x.shape)
    d = mu.shape[0]
    return -0.5 * (
        jnp.sum(y * y, axis=-1) + d * _LOG_2PI
    ) - jnp.sum(jnp.log(jnp.diag(chol)))


def _trapezoid_weights(betas: Array) -> Array:
    db = jnp.diff(betas)
    w = jnp.zeros_like(betas)
    w = w.at[:-1].add(0.5 * db)
    w = w.at[1:].add(0.5 * db)
    return w


def _ess_ratio(logw: Array, axis: int = -1) -> Array:
    lw = logw - jnp.max(logw, axis=axis, keepdims=True)
    w = jnp.exp(lw)
    w = w / jnp.sum(w, axis=axis, keepdims=True)
    return 1.0 / jnp.sum(w * w, axis=axis) / logw.shape[axis]


def _logmeanexp(a: Array, axis: int = -1) -> Array:
    return logsumexp(a, axis=axis) - jnp.log(a.shape[axis])




def _ess_betas(betas, logw, hs, rho_star: float, delta_max: float = 1.0,
               max_betas: int = 200, bucket: int = 1, iters: int = 40) -> np.ndarray:
    """Adaptive checkpoint grid on [0, 1]: every step has worst-mode incremental
    relative ESS >= rho_star,
    measured on the current frozen-flow rollout.

    From beta, the particles of the nearest checkpoint beta_k <= beta,
    reweighted to p_beta^j by log r = log w_k + (beta - beta_k) h, give
        rho_j(beta, delta) = (sum r e^{delta h})^2 / (sum r * sum r e^{2 delta h}),
    which is 1 at delta = 0 and nonincreasing in delta. Steps are taken
    greedily: the largest delta <= min(1 - beta, delta_max) with
    min_j rho_j >= rho_star (bisection). The number of points is an output;
    if it is not a multiple of `bucket`, the costliest steps are split at
    their midpoints (this only raises the step ESS, and keeps the number of
    distinct grid shapes, hence recompilations, small). Uses only the
    rollout's values of h, no new oracle calls.

    betas: (K0,) grid of the rollout; logw, hs: (K0, J, n) its log weights
    (log gamma - log q) and h. Returns the new grid (K,).
    """
    b = np.asarray(betas, dtype=np.float64)
    lw = np.asarray(logw, dtype=np.float64)
    h = np.asarray(hs, dtype=np.float64)
    bad = ~(np.isfinite(lw) & np.isfinite(h))
    lw = np.where(bad, -np.inf, lw)
    h = np.where(bad, 0.0, h)

    def lse(a):
        m = np.max(a, axis=-1, keepdims=True)
        m = np.where(np.isfinite(m), m, 0.0)
        return np.log(np.sum(np.exp(a - m), axis=-1)) + m[..., 0]

    def min_rho(beta, delta):
        k = int(np.searchsorted(b, beta, side="right")) - 1
        base = lw[k] + (beta - b[k]) * h[k]                      # (J, n)
        rho = np.exp(2.0 * lse(base + delta * h[k]) - lse(base) - lse(base + 2.0 * delta * h[k]))
        return float(np.min(np.nan_to_num(rho, nan=1.0)))

    grid, beta = [0.0], 0.0
    while beta < 1.0 - 1e-12:
        if len(grid) >= max_betas:
            raise ValueError(f"adaptive grid did not reach beta=1 within max_betas={max_betas} "
                             f"(at beta={beta:.6g}); lower adaptive_target_ess or raise adaptive_max_betas")
        cap = min(1.0 - beta, float(delta_max))
        if min_rho(beta, cap) >= rho_star:
            delta = cap
        else:
            lo, hi = 0.0, cap
            for _ in range(iters):
                mid = 0.5 * (lo + hi)
                lo, hi = (mid, hi) if min_rho(beta, mid) >= rho_star else (lo, mid)
            delta = max(lo, 1e-9)
        beta = 1.0 if beta + delta > 1.0 - 1e-12 else beta + delta
        grid.append(beta)
    while len(grid) % max(int(bucket), 1) != 0:
        costs = [min_rho(a, c - a) for a, c in zip(grid[:-1], grid[1:])]
        i = int(np.argmin(costs))
        grid.insert(i + 1, 0.5 * (grid[i] + grid[i + 1]))
    return np.asarray(grid)


# ============================================================
# Result object
# ============================================================
@dataclass
class CATResult:
    field_params: Any                      # stacked over modes (leading axis J)
    normalizer_rate_params: Any            # stacked over modes (learned c_eta)
    modes: List[Mode]
    betas_used: Array                      # final checkpoint grid (adaptive: an output)

    final_snapshot: Array        # (n, d) mixture draws
    final_comp_snapshot: Array   # (n,) component index of each draw
    final_logq: Array            # (n,) mixture log q at the draws
    final_logC: Array            # (J,) log hat C_j
    final_weights: Array         # (J,) hat w_j

    train_loss_history: List[float]
    ess_ratio_history: List[Array]        # per validation: (J, K) ESS / n (K = that round's grid)
    kinetic_energy_history: List[Array]   # per validation: (J, K) E_q |v|^2
    normalizer_rate_integrals: Array                   # (J,) log a_j + int_0^1 c_eta

    snapshots: List[Array]        # per beta: (J, n_validation, d), last validation rollout
    logw_snapshots: List[Array]   # per beta: (J, n_validation)

    train_ess_history: List[Array] = field(default_factory=list)  # (J, K) on training rollouts
    grid_history: List[np.ndarray] = field(default_factory=list)  # grid used at every refresh
    diagnostics: List[Dict[str, Any]] = field(default_factory=list)
    train_step_losses: List[Array] = field(default_factory=list)  # per refresh: (steps,)
    wall_time: float = 0.0
    n_target_evals: int = 0
    n_target_grad_evals: int = 0
    n_divergence_evals: int = 0
    selected_refresh: int = -1
    phase_times: Dict[str, float] = field(default_factory=dict)   # wall seconds per phase
    train_start: Optional[float] = None    # wall time (s) to the first training step (setup, compile)
    compile_time: float = 0.0              # JAX compile wall time (s), within wall_time

    # Filled by `cat`; closures over the compiled field functions.
    _rollout_fn: Optional[Callable] = field(default=None, repr=False)
    _inverse_fn: Optional[Callable] = field(default=None, repr=False)
    _velocity_fn: Optional[Callable] = field(default=None, repr=False)
    _rate_fn: Optional[Callable] = field(default=None, repr=False)

    def rollout(self, key: Array, n: int, return_path: bool = True):
        """Fresh forward rollout of every mode from x_0 ~ b_j.

        Returns (xs, logqs): (K, J, n, d) and (K, J, n) if `return_path`,
        otherwise only the beta=1 endpoint: (J, n, d) and (J, n).
        """
        xs, lqs = self._rollout_fn(self.field_params, key, int(n))
        if return_path:
            return xs, lqs
        return xs[-1], lqs[-1]

    def sample(self, key: Array, n: int) -> Tuple[Array, Array]:
        """n draws from the calibrated mixture sum_j hat w_j q_1^j.

        Returns (x, comp) with x: (n, d) and comp: (n,). Each mode is rolled
        out with n particles; only the selected rows are kept.
        """
        k_comp, k_roll = jr.split(key)
        comp = jr.categorical(k_comp, jnp.log(self.final_weights), shape=(int(n),))
        x1, _ = self.rollout(k_roll, n, return_path=False)
        return x1[comp, jnp.arange(int(n))], comp

    def component_log_prob(self, x: Array, j: int) -> Array:
        """log q_1^j(x) by integrating the inverse ODE from beta=1 to beta=0.

        Cost: one full RK4 pass with exact divergences per call (the same as a
        forward rollout), i.e. much slower than sampling. The inverse is a
        separate RK4 discretization, so it agrees with the forward-rollout
        log q only up to integrator error.
        """
        x = jnp.atleast_2d(jnp.asarray(x))
        return self._inverse_fn(self.field_params, x)[j]

    def mixture_log_prob(self, x: Array) -> Array:
        """log sum_j hat w_j q_1^j(x) via J inverse ODE solves (see above)."""
        x = jnp.atleast_2d(jnp.asarray(x))
        lq = self._inverse_fn(self.field_params, x)   # (J, n)
        return logsumexp(lq + jnp.log(self.final_weights)[:, None], axis=0)

    def velocity(self, x: Array, beta: float, j: int = 0) -> Array:
        """Physical velocity v_j(beta, x), x: (n, d)."""
        return self._velocity_fn(self.field_params, jnp.atleast_2d(x), beta)[j]

    def normalizer_rate(self, betas: Array) -> Array:
        """c_eta^j(beta) for all modes, (J, len(betas))."""
        return self._rate_fn(self.normalizer_rate_params, bs=jnp.asarray(betas))


# ============================================================
# Main entry point
# ============================================================
def _validate_betas(betas, dtype) -> Array:
    b = jnp.asarray(betas, dtype=dtype)
    if b.ndim != 1 or b.shape[0] < 2:
        raise ValueError(f"betas must be a 1D array with >= 2 entries, got shape {b.shape}")
    if float(b[0]) != 0.0 or float(b[-1]) != 1.0:
        raise ValueError(f"betas must start at 0 and end at 1, got {float(b[0])}, {float(b[-1])}")
    if not bool(jnp.all(jnp.diff(b) > 0)):
        raise ValueError("betas must be strictly increasing")
    return b


def cat(
    *,
    key: Array,
    logpi: LogPiBatch,
    logpi_single: LogPiSingle,
    modes: List[Mode],
    betas: Optional[Sequence[float]] = None,
    adaptive_target_ess: Optional[float] = None,
    adaptive_delta_max: float = 1.0,
    grad_logpi_single: Optional[GradLogPiSingle] = None,
    n: int = 4096,
    n_train: int = 1024,
    n_validation: int = 256,
    n_refresh: int = 20,
    train_steps_per_refresh: int = 250,
    batch_size: Optional[int] = None,
    rk4_substeps: int = 4,
    hidden: int = 64,
    depth: int = 3,
    activation: str = "silu",
    spline_knots: int = 0,
    gradient_field: bool = False,
    lr: float = 1e-3,
    grad_clip: float = 5.0,
    lr_warmup: int = 0,
    lr_final_frac: float = 1.0,
    velocity_l2_weight: float = 0.0,
    center_h: bool = False,
    residual_scale: str = "none",
    residual_loss: str = "l2",
    huber_delta: float = 3.0,
    residual_diagnostics: bool = False,
    validate_every: int = 1,
    select_best: bool = True,
    particle_chunk: Optional[int] = None,
    replay_refreshes: int = 1,
    replay_max_gb: float = 8.0,
    field_dtype=jnp.float32,
    final_calibration: Optional[int] = None,
    mvn_jitter0: float = 1e-10,
    dtype=jnp.float64,
    verbose: bool = True,
    callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> CATResult:
    """Fit one continuous vector-field flow per mode along the annealing path.

    Args:
      betas:        fixed increasing checkpoint grid with betas[0]=0, betas[-1]=1.
                    Exactly one of `betas` and `adaptive_target_ess` is given.
      adaptive_target_ess: target worst-mode incremental relative ESS in (0, 1]
                    of every step. The grid is
                    placed before every refresh from the latest rollout of the
                    current field (`_ess_betas`; the first placement uses the
                    identity-flow draws x_0 ~ b_j, which needs no rollout) and
                    is used for the training and validation rollouts, the
                    collocation pools, the inverse ODE and the final sampling.
                    Its size K is an output (`grid_history`, `betas_used`),
                    rounded up to an even number to limit recompilation; the
                    oracle cost per rollout is proportional to K.
      adaptive_delta_max: cap on any step.
      n:            number of final mixture draws in `final_snapshot`.
      n_train:      training trajectories per mode per refresh; every one of
                    them is a collocation point at each checkpoint.
      n_validation: fresh validation trajectories per mode per refresh (ESS).
      batch_size:   collocation points per mode per optimizer step (default
                    n_train). Each point costs one divergence (one forward pass).
      lr_warmup, lr_final_frac: linear warm-up over this many optimizer steps to `lr`, then
                    cosine decay over all n_refresh * train_steps_per_refresh steps to
                    lr * lr_final_frac (defaults: constant lr). The Adam state persists
                    across refreshes, so the schedule runs over the whole training.
      velocity_l2_weight: lambda in the penalty lambda * E|v|^2 added to the
                    squared continuity residual. The continuity equation does
                    not determine a general field; the penalty picks the
                    minimal-kinetic-energy one.
      center_h:     subtract a fixed per-mode offset h0_j = mean of h over the first
                    collocation pool from h and write c_eta = h0_j + scale_j * MLP(beta),
                    with scale_j = max(std(h - h0_j), 1). The residual is unchanged, but
                    training no longer depends on an additive constant in log pi (False:
                    no offset and scale_j = std(h) + |mean(h)|, which grows with the
                    constant).
      residual_scale: "none": raw residual. "beta": divide R by
                    sigma_{j,k} = max(std of h over the (mode j, beta_k) slice, 1), computed
                    from each refresh's pool and kept with that pool in the replay buffer.
                    The velocity penalty is not rescaled.
      residual_loss: "l2": mean of z^2 with z the (scaled) residual. "pseudo_huber":
                    mean of 2 delta^2 (sqrt(1 + (z/delta)^2) - 1), which is ~z^2 for
                    |z| << delta and grows linearly beyond (delta = huber_delta).
      residual_diagnostics: at every validation, record per (mode, beta) statistics of
                    the residual on the latest collocation pool (no oracle calls):
                    quantiles of |R|, and the RMS of R, h - c, div v and s.v.
      validate_every: run the validation rollout (and its oracle calls) only
                    every this many refreshes; the last refresh always
                    validates. Histories then have one entry per validation.
      select_best:  return the field from the validation with the highest
                    mean endpoint ESS over modes instead of the last one.
      particle_chunk: evaluate rollouts, inverse ODEs, target values and
                    gradients in chunks of this many particles (bounds memory).
      replay_refreshes: train on the collocation pools of the last this many
                    refreshes (1 = current pool only). The residual is a
                    pointwise PDE identity, so older points and their stored
                    target values/gradients stay valid; this adds training
                    data without extra oracle calls. Every pool is kept with
                    its own grid's quadrature weights (pools of different
                    grids mix freely); minibatches are gathered on the host.
      replay_max_gb: cap on the host memory of the replay pools (stored in
                    float32; the oldest pools are dropped first, the latest
                    is always kept), for high-dimensional targets.
      spline_knots: > 0 adds a per-coordinate cubic B-spline term with this
                    many knots (see VectorMLP).
      field_dtype:  dtype of the field/rate networks (default float32, ~2x
                    faster on CPU). Particle states, log densities, target
                    terms and weights always stay in `dtype`. None: `dtype`.
      final_calibration: None: the endpoint weights log C_j come from rolling
                    out all `n` final particles under every mode. An int m
                    estimates log C_j from m fresh particles per mode, draws
                    the component labels from the weights, and pushes only
                    max_j n_j particles per mode. log q still uses the
                    inverse ODE under every mode.
      callback:     called after every refresh's training (validation or not) with a dict: refresh, elapsed (wall
                    seconds of training so far), train_start (elapsed at the first training
                    step), compile_s (JAX compile seconds so far, within elapsed),
                    params (field), rate_params, grid,
                    rollout(key, n) -> (xs, logqs) on that grid (K, J, n, ...),
                    inverse(x) -> log q_1^j(x) (J, n), velocity(x, beta) -> (J, n, d),
                    h(x) = log p~ - log b, loga, mu, chol. Its time is excluded from the
                    wall time and its target calls from the oracle counters (evaluation
                    only; it does not change the run).
    """
    t_start = time.time()
    t_callback = 0.0
    cb_windows: List[Tuple[float, float]] = []    # callback intervals (not charged)
    train_start = None                            # elapsed time at the first training step
    if rk4_substeps < 1:
        raise ValueError("rk4_substeps must be >= 1")
    if len(modes) == 0:
        raise ValueError("modes must be non-empty")
    if int(validate_every) < 1 or int(replay_refreshes) < 1:
        raise ValueError("validate_every and replay_refreshes must be >= 1")
    if int(spline_knots) < 0 or int(spline_knots) in (1, 2, 3):
        raise ValueError("spline_knots must be 0 (off) or >= 4")
    if residual_scale not in ("none", "beta"):
        raise ValueError(f"residual_scale must be 'none' or 'beta', got {residual_scale!r}")
    if residual_loss not in ("l2", "pseudo_huber"):
        raise ValueError(f"residual_loss must be 'l2' or 'pseudo_huber', got {residual_loss!r}")
    if residual_loss == "pseudo_huber" and not huber_delta > 0:
        raise ValueError("huber_delta must be > 0")

    adaptive = betas is None
    if adaptive:
        if adaptive_target_ess is None or not (0.0 < adaptive_target_ess <= 1.0):
            raise ValueError("betas=None requests adaptive scheduling: give adaptive_target_ess in (0, 1]")
        if adaptive_delta_max <= 0.0:
            raise ValueError("adaptive_delta_max must be > 0")
        grid = None                          # placed at the first refresh
    else:
        if adaptive_target_ess is not None:
            raise ValueError("give either betas or adaptive_target_ess, not both")
        grid = _validate_betas(betas, dtype)
    del betas
    J = len(modes)
    d = int(jnp.asarray(modes[0].mu).shape[0])
    substeps = int(rk4_substeps)
    n_train = int(n_train)
    n_validation = int(n_validation)
    batch_size = n_train if batch_size is None else int(batch_size)

    # ---------------- anchors ----------------
    mu_all = jnp.stack([
        jnp.asarray(m.anchor_mu if m.anchor_mu is not None else m.mu, dtype=dtype)
        for m in modes
    ])                                                          # (J, d)
    Sigma_all = jnp.stack([sym_jax(jnp.asarray(m.Sigma, dtype=dtype)) for m in modes])
    chol_all = chol_jitter_jax(Sigma_all, jitter0=mvn_jitter0, max_tries=12)
    w_raw = jnp.asarray([float(m.w) for m in modes], dtype=dtype)
    if bool(jnp.any(w_raw <= 0)):
        raise ValueError("All mode weights must be positive")
    loga_all = jnp.log(w_raw)                                   # (J,)

    def log_b_components(x):   # (..., d) -> (..., J)
        return jnp.stack(
            [_gauss_logpdf(x, mu_all[j], chol_all[j]) for j in range(J)], axis=-1
        )

    def log_b_components_single(x):  # (d,) -> (J,)
        return log_b_components(x[None])[0]

    def h_batch(x):   # (N, d) -> (N,), log p~ - log b
        lb = logsumexp(log_b_components(x) + loga_all, axis=-1)
        return logpi(x) - lb

    if grad_logpi_single is None:
        grad_logpi_single = jax.grad(logpi_single)

    def score_terms_single(x):
        """(grad log b_j (J,d), grad h (d,)) at a single point."""
        jac_lbj = jax.jacrev(log_b_components_single)(x)        # (J, d)
        lbj = log_b_components_single(x)
        resp = jax.nn.softmax(lbj + loga_all)                  # (J,)
        grad_lb = resp @ jac_lbj
        return jac_lbj, grad_logpi_single(x) - grad_lb

    score_terms = jax.jit(jax.vmap(score_terms_single))
    h_jit = jax.jit(h_batch)

    # ---------------- networks ----------------
    fdt = dtype if field_dtype is None else field_dtype
    model = VectorMLP(hidden=int(hidden), depth=int(depth), act=activation,
                      spline_knots=int(spline_knots), gradient=bool(gradient_field))
    vel_div, vel_div_b = make_vec_fns(model, None if field_dtype is None else fdt)
    k_init, key = jr.split(key)
    init_keys = jr.split(k_init, J)
    init_one = lambda k: model.init(
        k, jnp.zeros((d,), fdt), beta_features(jnp.asarray(0.0, fdt))
    )["params"]
    phi_params = jax.vmap(init_one)(init_keys)
    phi_params = jax.tree_util.tree_map(lambda a: a.astype(fdt), phi_params)

    rate_model = RateMLP(act=activation)
    k_rate, key = jr.split(key)
    init_rate = lambda k: rate_model.init(
        k, beta_features(jnp.asarray(0.0, fdt)), jnp.asarray(1.0, fdt)
    )["params"]
    rate_params = jax.vmap(init_rate)(jr.split(k_rate, J))
    rate_params = jax.tree_util.tree_map(lambda a: a.astype(fdt), rate_params)
    rate_scale = jnp.ones((J,), dtype)   # set from the first collocation pool
    h_off = jnp.zeros((J,), dtype)       # c_eta offset h0_j (center_h), same

    def rate_single(rp, beta, scale):
        out = rate_model.apply(
            {"params": rp}, beta_features(jnp.asarray(beta).astype(fdt)),
            jnp.asarray(scale).astype(fdt),
        )
        return out.astype(dtype)

    # ---------------- rollouts ----------------
    chunk = None if particle_chunk is None else int(particle_chunk)

    def chunked(fn, args, n_, in_axis=0, out_axis=0):
        """fn(*args) evaluated in particle chunks along in_axis (last chunk
        padded by repetition), outputs concatenated along out_axis."""
        if chunk is None or n_ <= chunk:
            return fn(*args)
        pad = (-n_) % chunk
        if pad:
            idx = jnp.concatenate([jnp.arange(n_), jnp.full((pad,), n_ - 1)])
            args = [jnp.take(a, idx, axis=in_axis) for a in args]
        outs = [fn(*[jax.lax.slice_in_dim(a, i, i + chunk, axis=in_axis) for a in args])
                for i in range(0, n_ + pad, chunk)]
        return jax.tree_util.tree_map(
            lambda *o: jax.lax.slice_in_dim(jnp.concatenate(o, axis=out_axis), 0, n_, axis=out_axis),
            *outs)

    @partial(jax.jit, static_argnames=("n_",))
    def initial_draws(key_, n_):
        eps = jr.normal(key_, (J, n_, d), dtype=dtype)
        x0 = mu_all[:, None, :] + jnp.einsum("jab,jnb->jna", chol_all, eps)
        return x0, jax.vmap(_gauss_logpdf)(x0, mu_all, chol_all)     # (J,n,d), (J,n)

    # The checkpoint grid `bs` is a traced argument everywhere below, so a moved
    # grid of the same size reuses the compiled functions.
    @jax.jit
    def push(params, x0, lq0, bs):
        xs, lqs = jax.vmap(
            lambda p, x, l, m, c: _rk4_rollout(vel_div, p, x, l, bs, m, c, substeps)
        )(params, x0, lq0, mu_all, chol_all)                         # (J,K,n,d), (J,K,n)
        return jnp.swapaxes(xs, 0, 1), jnp.swapaxes(lqs, 0, 1)       # (K,J,n,d), (K,J,n)

    def rollout_all(params, key_, n_, bs):
        """Fresh draws x_0 ~ b_j pushed along the frozen field, all modes."""
        x0, lq0 = initial_draws(key_, int(n_))
        return chunked(lambda a, b: push(params, a, b, bs), (x0, lq0), int(n_), in_axis=1, out_axis=2)

    def inverse_all(params, x, bs):
        return chunked(lambda xc: _inverse_all(params, xc, bs), (x,), x.shape[0], 0, 1)

    @jax.jit
    def _inverse_all(params, x, bs):
        """log q_1^j(x) for all j via the inverse ODE, (J, n)."""
        n_ = x.shape[0]
        zeros = jnp.zeros((n_,), dtype)
        rev = bs[::-1]

        def one(p, m, c):
            xs, acc = _rk4_rollout(vel_div, p, x, zeros, rev, m, c, substeps)
            # acc(0) = int_1^0 (-div) dbeta = int_0^1 div dbeta
            return _gauss_logpdf(xs[-1], m, c) - acc[-1]

        return jax.vmap(one)(params, mu_all, chol_all)

    @jax.jit
    def velocity_all(params, x, beta):
        return jax.vmap(lambda p, m, c: vel_div(p, x, beta, m, c)[0])(params, mu_all, chol_all)

    @jax.jit
    def rate_all(rp, scale, off, bs):
        """c_eta^j(beta) = off_j + scale_j * MLP_j(beta) for all modes, (J, len(bs))."""
        return off[:, None] + jax.vmap(
            lambda r, s: jax.vmap(lambda b: rate_single(r, b, s))(bs)
        )(rp, scale)

    @jax.jit
    def log_gamma(xs, hs, bs):
        """log gamma_beta^j at checkpoint particles; xs: (K,J,n,d), hs: (K,J,n)."""
        lbj = jax.vmap(jax.vmap(
            lambda x, m, c: _gauss_logpdf(x, m, c), in_axes=(0, 0, 0)
        ), in_axes=(0, None, None))(xs, mu_all, chol_all)            # (K,J,n)
        return lbj + bs[:, None, None] * (loga_all[None, :, None] + hs)

    def eval_h(xs):
        shp = xs.shape
        flat = xs.reshape(-1, d)
        return chunked(h_jit, (flat,), flat.shape[0]).reshape(shp[:-1])

    # ---------------- loss ----------------
    def ce_terms(phi_p, rate_p, scale, off, x, beta, h, glbj, gh, m, c):
        """(div v, s.v, h - c, v) at a batch of (beta_i, x_i) for one mode; the offset
        is taken out of h in `dtype` before the float32 network part of c is subtracted."""
        v, div = vel_div_b(phi_p, x, beta, m, c)
        s = glbj + beta[:, None] * gh
        cval = jax.vmap(lambda b: rate_single(rate_p, b, scale))(beta)
        return div, jnp.sum(s * v, axis=-1), (h - off) - cval, v

    def ce_residuals(phi_p, rate_p, scale, off, x, beta, h, glbj, gh, m, c):
        """(R, v) at a batch of (beta_i, x_i) for one mode; data inputs are stop-grad."""
        v, div = vel_div_b(phi_p, x, beta, m, c)
        s = glbj + beta[:, None] * gh
        cval = jax.vmap(lambda b: rate_single(rate_p, b, scale))(beta)
        return div + jnp.sum(s * v, axis=-1) + (h - off) - cval, v

    def ce_loss(params, batch, scale, off):
        phi_p, rate_p = params
        r, v = jax.vmap(ce_residuals, in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0))(
            phi_p, rate_p, scale, off, batch["x"], batch["beta"], batch["h"],
            batch["glbj"], batch["gh"], mu_all, chol_all,
        )                                                            # (J, B), (J, B, d)
        z2 = r * r if residual_scale == "none" else batch["rw"] * r * r
        if residual_loss == "l2":
            loss_val = jnp.mean(z2)
        else:
            dl2 = huber_delta ** 2
            loss_val = jnp.mean(2.0 * dl2 * (jnp.sqrt(1.0 + z2 / dl2) - 1.0))
        if velocity_l2_weight:
            loss_val = loss_val + velocity_l2_weight * jnp.mean(jnp.sum(v * v, axis=-1))
        return loss_val, r

    # ---------------- optimizer ----------------
    total_steps = int(n_refresh) * int(train_steps_per_refresh)
    if lr_warmup or lr_final_frac != 1.0:
        lr_sched = optax.warmup_cosine_decay_schedule(
            init_value=0.0 if lr_warmup else lr, peak_value=lr, warmup_steps=int(lr_warmup),
            decay_steps=total_steps, end_value=lr * lr_final_frac)
    else:
        lr_sched = lr
    tx = optax.chain(optax.clip_by_global_norm(grad_clip), optax.adam(lr_sched))
    train_params = (phi_params, rate_params)
    opt_state = tx.init(train_params)

    @jax.jit
    def train_batches(params, opt_state, batches, scale, off):
        """Optimizer steps on minibatches gathered in advance (leading axis = steps).
        Returns the losses and the gradient norms (total, field part, rate part)."""
        def step(carry, b):
            params, opt_state = carry
            (l, _), g = jax.value_and_grad(ce_loss, has_aux=True)(params, b, scale, off)
            upd, opt_state = tx.update(g, opt_state, params)
            params = optax.apply_updates(params, upd)
            gns = jnp.stack([optax.global_norm(g), optax.global_norm(g[0]), optax.global_norm(g[1])])
            return (params, opt_state), (l, gns)

        (params, opt_state), (ls, gn) = jax.lax.scan(step, (params, opt_state), batches)
        return params, opt_state, ls, gn

    @jax.jit
    def _slice_terms(params, x, beta, h, glbj, gh, scale, off):
        """(div, s.v, h - c) at (J, m) points of one checkpoint beta."""
        bb = jnp.full(h.shape, beta, dtype)
        div, sv, hc, _ = jax.vmap(ce_terms)(params[0], params[1], scale, off, x, bb, h, glbj, gh,
                                            mu_all, chol_all)
        return div, sv, hc

    def residual_stats(params, pool, bs, scale, off):
        """Per (mode, beta) residual statistics on a pool (K, J, n, ...), each (J, K),
        evaluated in chunks of batch_size points per mode."""
        K, _, n_ = pool["h"].shape
        pad = (-n_) % batch_size
        stats = {k: np.zeros((J, K)) for k in ("abs_r_median", "abs_r_p95", "abs_r_p99", "abs_r_max",
                                                "r_mean", "r_rms", "hc_rms", "div_rms", "sv_rms")}
        rms = lambda a: np.sqrt(np.mean(a * a, axis=-1))
        for k in range(K):
            idx = jnp.concatenate([jnp.arange(n_), jnp.full((pad,), n_ - 1)])
            sl = [jnp.take(pool[f][k], idx, axis=1) for f in ("x", "h", "glbj", "gh")]
            parts = []
            for i in range(0, n_ + pad, batch_size):
                x, h, glbj, gh = (jax.lax.slice_in_dim(a, i, i + batch_size, axis=1) for a in sl)
                parts.append(_slice_terms(params, x, bs[k], h, glbj, gh, scale, off))
            div, sv, hc = (np.concatenate([np.asarray(p[t]) for p in parts], axis=1)[:, :n_] for t in range(3))
            r = div + sv + hc
            q = np.quantile(np.abs(r), [0.5, 0.95, 0.99], axis=-1)
            for name, val in (("abs_r_median", q[0]), ("abs_r_p95", q[1]), ("abs_r_p99", q[2]),
                              ("abs_r_max", np.max(np.abs(r), -1)), ("r_mean", np.mean(r, -1)),
                              ("r_rms", rms(r)), ("hc_rms", rms(hc)), ("div_rms", rms(div)),
                              ("sv_rms", rms(sv))):
                stats[name][:, k] = val
        return stats

    @jax.jit
    def kinetic(params, xs, bs):
        """|v|^2 at checkpoint particles, (K, J, n)."""
        def one_mode(p, x, m, c):
            return jax.vmap(
                lambda xk, bk: jnp.sum(vel_div(p, xk, bk, m, c)[0] ** 2, axis=-1)
            )(x, bs)
        return jnp.swapaxes(
            jax.vmap(one_mode, in_axes=(0, 1, 0, 0))(params, xs, mu_all, chol_all), 0, 1
        )

    # ---------------- helpers for one refresh ----------------
    counters = dict(target=0, grad=0, div=0)

    def div_cost(K):    # divergence evaluations per particle of one rollout
        return (K - 1) * substeps * 4

    def do_rollout(params, key_, n_, bs):
        xs, lqs = rollout_all(params, key_, n_, bs)
        hs = eval_h(xs)
        lg = log_gamma(xs, hs, bs)
        logw = lg - lqs
        counters["target"] += int(bs.shape[0]) * J * n_
        counters["div"] += div_cost(int(bs.shape[0])) * J * n_
        return xs, lqs, hs, logw

    def build_pool(xs, hs):
        """Collocation pool on the current grid: the rollout's own particles at
        every checkpoint, with the target-side terms at them (K, J, n_train, ...).
        Costs one gradient of log pi per point; h comes from the rollout."""
        K = xs.shape[0]
        x = jax.lax.stop_gradient(xs)
        flat = x.reshape(-1, d)
        glbj_all, gh = chunked(score_terms, (flat,), flat.shape[0])   # (M, J, d), (M, d)
        glbj_all = glbj_all.reshape(K, J, n_train, J, d)
        # own-mode gradient of log b_j
        glbj = jnp.stack([glbj_all[:, j, :, j, :] for j in range(J)], axis=1)
        counters["grad"] += K * J * n_train
        # residual weight 1/sigma_{j,k}^2 of each (beta, mode) slice (residual_scale="beta")
        rw = (1.0 / jnp.maximum(jnp.std(hs, axis=-1), 1.0) ** 2 if residual_scale == "beta"
              else jnp.ones((K, J), dtype))
        return dict(x=x, h=hs, glbj=glbj, gh=gh.reshape(K, J, n_train, d), rw=rw)

    def flatten_pool(pool, bs):
        """Host copy of a pool as per-mode point arrays (J, K*n, ...) with the
        probability pw of drawing each point: its slice's trapezoid weight of
        *its own* grid, split evenly over the slice's points."""
        K, _, n_ = pool["x"].shape[:3]
        fl = lambda a: np.asarray(jnp.swapaxes(a, 0, 1).reshape((J, K * n_) + a.shape[3:]))
        out = {f: fl(pool[f]) for f in ("x", "h", "glbj", "gh")}
        for f in ("x", "glbj", "gh"):     # the (.., d) arrays dominate host memory
            out[f] = out[f].astype(np.float32)
        out["beta"] = np.asarray(jnp.broadcast_to(bs[None, :, None], (J, K, n_)).reshape(J, K * n_))
        out["pw"] = np.asarray(jnp.broadcast_to((_trapezoid_weights(bs) / n_)[None, :, None],
                                                (J, K, n_)).reshape(J, K * n_))
        if residual_scale != "none":
            out["rw"] = np.asarray(jnp.broadcast_to(jnp.swapaxes(pool["rw"], 0, 1)[:, :, None],
                                                    (J, K, n_)).reshape(J, K * n_))
        return out

    def gather_batches(rng, pooled, cdf, steps):
        """`steps` minibatches (J, batch_size) of the concatenated pools `pooled`
        (points drawn by inverse-CDF on the per-mode probabilities `cdf`)."""
        u = rng.random((steps, J, batch_size))
        idx = np.stack([np.searchsorted(cdf[j], u[:, j]) for j in range(J)], axis=1)
        idx = np.minimum(idx, cdf.shape[1] - 1)                     # (steps, J, B)
        jj = np.arange(J)[None, :, None]
        fields = ("x", "h", "glbj", "gh", "beta") + (("rw",) if residual_scale != "none" else ())
        return jax.device_put({f: pooled[f][jj, idx] for f in fields})

    # ---------------- outer loop ----------------
    train_loss_history: List[float] = []
    ess_ratio_history: List[Array] = []
    train_ess_history: List[Array] = []
    kinetic_energy_history: List[Array] = []
    grid_history: List[np.ndarray] = []
    diagnostics: List[Dict[str, Any]] = []
    train_step_losses: List[Array] = []
    last_val = None

    if verbose:
        sched = (f"K adaptive (target ESS {adaptive_target_ess}, delta_max {adaptive_delta_max})"
                 if adaptive else f"K={int(grid.shape[0])}")
        print(f"[cat] J={J} d={d} {sched} substeps={substeps} "
              f"n_train={n_train} n_val={n_validation} refresh={n_refresh}x{train_steps_per_refresh}")

    best_score, best = -jnp.inf, None
    selected_refresh = int(n_refresh) - 1
    phase_times = dict(rollout=0.0, schedule=0.0, pool=0.0, train=0.0, validation=0.0, final=0.0)
    pool_hist: List[Dict[str, np.ndarray]] = []
    last_roll = None        # (grid, logw, h) of the latest rollout of the current field
    np_rng = np.random.default_rng(int(jr.randint(jr.fold_in(key, 7), (), 0, 2 ** 31 - 1)))

    def place(bs, lw, hs_):
        new = _ess_betas(bs, lw, hs_, adaptive_target_ess, adaptive_delta_max,
                         max_betas=200, bucket=2)
        if verbose:
            print(f"  [grid] K={len(new)} | betas " + " ".join(f"{b:.3g}" for b in new))
        return jnp.asarray(new, dtype)

    def _tick(name, t, out=None):
        """Charge the wall time since t to phase `name` (after the async work is done)."""
        if out is not None:
            jax.block_until_ready([a for a in jax.tree_util.tree_leaves(out)
                                   if isinstance(a, jax.Array)])
        now = time.time()
        phase_times[name] += now - t
        return now

    def _run_callback(r):
        """The evaluation callback on the current field (see `callback`); its time is
        taken out of the wall time."""
        nonlocal t_callback
        if callback is None:
            return
        t_cb = time.time()
        p_now, g_now = train_params[0], grid
        callback(dict(
            refresh=r, elapsed=t_cb - t_start - t_callback, train_start=train_start,
            compile_s=compile_seconds(t_start, t_cb, cb_windows), params=p_now, rate_params=train_params[1],
            grid=g_now, rollout=lambda k_, n_: rollout_all(p_now, k_, int(n_), g_now),
            inverse=lambda x_: inverse_all(p_now, x_, g_now),
            velocity=lambda x_, b_: velocity_all(p_now, x_, b_), h=eval_h,
            loga=loga_all, mu=mu_all, chol=chol_all))
        t_end = time.time()
        t_callback += t_end - t_cb
        cb_windows.append((t_cb, t_end))

    for r in range(int(n_refresh)):
        t0 = time.time()
        tr = dict(phase_times)
        key, k_roll, k_val = jr.split(key, 3)
        phi_frozen = train_params[0]

        t = t0
        if adaptive and r > 0:
            grid = place(*last_roll)
            t = _tick("schedule", t)
        if adaptive and r == 0:
            # The first field is exactly zero (identity flow): every checkpoint
            # holds the same draws x_0 ~ b_j, so the grid is placed on them
            # (log w = 0 at beta = 0) and h is evaluated once, without a rollout.
            x0, lq0 = initial_draws(k_roll, n_train)
            h0 = eval_h(x0)
            counters["target"] += J * n_train
            grid = place(jnp.zeros((1,), dtype), jnp.zeros((1, J, n_train), dtype), h0[None])
            Kc = int(grid.shape[0])
            xs = jnp.broadcast_to(x0[None], (Kc,) + x0.shape)
            lqs = jnp.broadcast_to(lq0[None], (Kc,) + lq0.shape)
            hs = jnp.broadcast_to(h0[None], (Kc,) + h0.shape)
            logw = log_gamma(xs, hs, grid) - lqs
            t = _tick("schedule", t, logw)
        else:
            xs, lqs, hs, logw = do_rollout(phi_frozen, k_roll, n_train, grid)
            t = _tick("rollout", t, logw)
        last_roll = (grid, logw, hs)
        grid_history.append(np.asarray(grid))
        train_ess = _ess_ratio(logw)                          # (K, J)
        train_ess_history.append(jnp.swapaxes(train_ess, 0, 1))

        pool = build_pool(xs, hs)
        t = _tick("pool", t, pool)
        if r == 0:
            # Preconditioner for c_eta: spread of h over the first pool.
            if center_h:
                h_off = jnp.mean(pool["h"], axis=(0, 2)).astype(dtype)
                spread = jnp.std(pool["h"] - h_off[None, :, None], axis=(0, 2))
            else:
                spread = jnp.std(pool["h"], axis=(0, 2)) + jnp.abs(jnp.mean(pool["h"], axis=(0, 2)))
            rate_scale = jnp.maximum(spread, 1.0).astype(dtype)
            if verbose and center_h:
                print(f"  [rate] offset h0 {[round(float(v), 2) for v in h_off]} "
                      f"scale {[round(float(v), 2) for v in rate_scale]}")
        pool_hist = (pool_hist + [flatten_pool(pool, grid)])[-int(replay_refreshes):]
        while len(pool_hist) > 1 and 2 * sum(a.nbytes for P in pool_hist for a in P.values()) > replay_max_gb * 2 ** 30:
            pool_hist = pool_hist[1:]     # keep the concatenated copy below the memory cap
        pooled = {f: np.concatenate([P[f] for P in pool_hist], axis=1) for f in pool_hist[0]}
        cdf = np.cumsum(pooled.pop("pw"), axis=1)
        cdf = cdf / cdf[:, -1:]
        ls_all, gn_all = [], []
        steps_left = int(train_steps_per_refresh)
        while steps_left > 0:       # chunks of <= 50 steps bound device memory
            s_ = min(50, steps_left)
            batches = gather_batches(np_rng, pooled, cdf, s_)
            train_params, opt_state, ls, gn = train_batches(train_params, opt_state, batches,
                                                            rate_scale, h_off)
            if train_start is None:     # dispatched (compiled): the first step runs now; q = b until here
                train_start = time.time() - t_start - t_callback
            ls_all.append(ls)
            gn_all.append(gn)
            del batches
            steps_left -= s_
        del pooled, cdf
        ls, gn3 = jnp.concatenate(ls_all), jnp.concatenate(gn_all)
        gn = gn3[:, 0]                  # total; gn3[:, 1] field part, gn3[:, 2] rate part
        t = _tick("train", t, ls)
        counters["div"] += int(train_steps_per_refresh) * J * batch_size
        train_step_losses.append(ls)
        train_loss_history.append(float(ls[-1]))

        def _split():
            dt = {k: phase_times[k] - tr[k] for k in ("rollout", "pool", "train", "validation")}
            return "roll {rollout:.0f} pool {pool:.0f} train {train:.0f} val {validation:.0f}s".format(**dt)

        if (r + 1) % int(validate_every) != 0 and r != int(n_refresh) - 1:
            if verbose:
                print(f"[refresh {r + 1:3d}/{n_refresh}] K {int(grid.shape[0])} "
                      f"train {train_loss_history[-1]:.4e} "
                      f"| |g| {float(jnp.mean(gn)):.2e} | {time.time() - t0:.1f}s ({_split()})")
            _run_callback(r)
            continue

        # ---- fresh validation rollout with the updated field ----
        phi_now = train_params[0]
        vxs, vlqs, vhs, vlogw = do_rollout(phi_now, k_val, n_validation, grid)
        last_roll = (grid, vlogw, vhs)
        ess = jnp.swapaxes(_ess_ratio(vlogw), 0, 1)           # (J, K)
        ke = kinetic(phi_now, vxs, grid)                          # (K, J, n)
        omega_v = jax.nn.softmax(vlogw, axis=-1)
        ke_mean = jnp.swapaxes(jnp.mean(ke, axis=-1), 0, 1)
        ke_w = jnp.swapaxes(jnp.sum(omega_v * ke, axis=-1), 0, 1)
        diag: Dict[str, Any] = dict(
            refresh=r,
            betas=grid,
            n_betas=int(grid.shape[0]),
            ess=ess,
            train_ess=train_ess_history[-1],
            logw_mean=jnp.swapaxes(jnp.mean(vlogw, -1), 0, 1),
            logw_var=jnp.swapaxes(jnp.var(vlogw, -1), 0, 1),
            kinetic=ke_mean,
            kinetic_weighted=ke_w,
            particle_mean=jnp.swapaxes(jnp.mean(vxs, axis=2), 0, 1),       # (J,K,d)
            # (J,K,d,d) grows quadratically in d; skipped in high dimension.
            particle_cov=jnp.swapaxes(jax.vmap(jax.vmap(
                lambda x: jnp.cov(x, rowvar=False)))(vxs), 0, 1) if d <= 64 else None,
            max_norm=jnp.swapaxes(jnp.max(jnp.linalg.norm(vxs, axis=-1), -1), 0, 1),
            grad_norm_mean=float(jnp.mean(gn)),
            grad_norm_max=float(jnp.max(gn)),
            grad_norm_field_mean=float(jnp.mean(gn3[:, 1])),
            grad_norm_rate_mean=float(jnp.mean(gn3[:, 2])),
            nonfinite=int(jnp.sum(~jnp.isfinite(vxs)) + jnp.sum(~jnp.isfinite(vlqs))),
            unreliable_slices=[(int(j), float(grid[k])) for j, k in zip(*jnp.where(ess < 0.05))],
            time=0.0,
        )

        c_now = rate_all(train_params[1], rate_scale, h_off, grid)           # (J, K)
        diag.update(c_eta=c_now, h_mean_is=jnp.swapaxes(jnp.sum(omega_v * vhs, -1), 0, 1))
        if residual_diagnostics:
            diag["residual"] = residual_stats(train_params, pool, grid, rate_scale, h_off)

        ess_ratio_history.append(ess)
        kinetic_energy_history.append(ke_mean)
        _tick("validation", t, diag)
        diag["time"] = time.time() - t0
        diag["phase_times"] = {k: phase_times[k] - tr[k] for k in phase_times}
        diagnostics.append(diag)
        last_val = (vxs, vlogw, grid)

        score = float(jnp.mean(jnp.nan_to_num(ess[:, -1], nan=0.0)))
        if select_best and diag["nonfinite"] == 0 and score > best_score:
            best_score, selected_refresh = score, r
            best = (train_params, last_val)

        _run_callback(r)

        if verbose:
            e1 = ", ".join(f"{float(v):.3f}" for v in ess[:, -1])
            emin = float(jnp.min(ess))
            msg = (f"[refresh {r + 1:3d}/{n_refresh}] K {int(grid.shape[0])} "
                   f"train {train_loss_history[-1]:.4e} "
                   f"| ESS(beta=1) [{e1}] min {emin:.3f} "
                   f"| |g| {diag['grad_norm_mean']:.2e} | {diag['time']:.1f}s ({_split()})")
            if diag["nonfinite"]:
                msg += f" | NONFINITE {diag['nonfinite']}"
            print(msg)
            print(f"  |g| field {diag['grad_norm_field_mean']:.2e} rate {diag['grad_norm_rate_mean']:.2e}")
            if residual_diagnostics:
                rs = diag["residual"]
                for j in range(J):
                    print(f"  [resid mode {j}] beta " + " ".join(f"{float(b):7.3g}" for b in grid))
                    for name in ("r_rms", "abs_r_median", "abs_r_p99", "abs_r_max", "r_mean",
                                 "hc_rms", "div_rms", "sv_rms"):
                        print(f"    {name:>12} " + " ".join(f"{v:7.3g}" for v in rs[name][j]))
            if diag["unreliable_slices"] and r == int(n_refresh) - 1:
                print(f"  low-ESS (<0.05) training slices (mode, beta): "
                      f"{diag['unreliable_slices'][:10]}"
                      + (" ..." if len(diag["unreliable_slices"]) > 10 else ""))

    if select_best and best is not None:
        train_params, last_val = best
        if verbose:
            print(f"[cat] selected refresh {selected_refresh + 1} "
                  f"(mean endpoint ESS {best_score:.4f})")
    phi_final = train_params[0]
    rate_final = train_params[1]

    # ---------------- endpoint calibration and final draws ----------------
    # Only the beta=1 endpoint needs target values here.
    t_final = time.time()
    key, k_final, k_comp = jr.split(key, 3)
    grid = last_val[2]          # the grid the returned field was validated on
    Kf = int(grid.shape[0])
    if final_calibration is None or J == 1:
        fxs, flqs = rollout_all(phi_final, k_final, int(n), grid)
        fx1, flq1 = fxs[-1], flqs[-1]                             # (J, n, d), (J, n)
        del fxs, flqs
        fh1 = eval_h(fx1)
        flogw1 = (jax.vmap(_gauss_logpdf)(fx1, mu_all, chol_all)
                  + loga_all[:, None] + fh1 - flq1)
        counters["target"] += J * int(n)
        counters["div"] += div_cost(Kf) * J * int(n)
        logC = _logmeanexp(flogw1, axis=-1)                       # (J,)
        weights = jax.nn.softmax(logC)
        comp = jr.categorical(k_comp, jnp.log(weights), shape=(int(n),))
        idx = jnp.arange(int(n))
    else:
        # Calibrate log C_j on m particles per mode, then push only the
        # particles each mode actually contributes to the mixture draw.
        m_cal = int(final_calibration)
        k_final, k_draw = jr.split(k_final)
        fxs, flqs = rollout_all(phi_final, k_final, m_cal, grid)
        cx1, clq1 = fxs[-1], flqs[-1]
        del fxs, flqs
        flogw1 = (jax.vmap(_gauss_logpdf)(cx1, mu_all, chol_all)
                  + loga_all[:, None] + eval_h(cx1) - clq1)
        counters["target"] += J * m_cal
        counters["div"] += div_cost(Kf) * J * m_cal
        logC = _logmeanexp(flogw1, axis=-1)
        weights = jax.nn.softmax(logC)
        comp = jr.categorical(k_comp, jnp.log(weights), shape=(int(n),))
        comp_np = np.asarray(comp)
        counts = np.bincount(comp_np, minlength=J)
        m_draw = int(counts.max())
        fxs, flqs = rollout_all(phi_final, k_draw, m_draw, grid)
        fx1, flq1 = fxs[-1], flqs[-1]                             # (J, m_draw, ...)
        del fxs, flqs
        counters["div"] += div_cost(Kf) * J * m_draw
        # rank of each draw among the draws of its component
        order = np.argsort(comp_np, kind="stable")
        rank = np.empty(int(n), dtype=np.int64)
        rank[order] = np.arange(int(n)) - np.concatenate([[0], np.cumsum(counts)[:-1]])[comp_np[order]]
        idx = jnp.asarray(rank)
    snap = fx1[comp, idx]                                     # (n, d)
    if J == 1:
        final_logq = flq1[0]
    else:
        lq_all = inverse_all(phi_final, snap, grid)               # (J, n)
        lq_all = lq_all.at[comp, jnp.arange(int(n))].set(flq1[comp, idx])
        final_logq = logsumexp(lq_all + jnp.log(weights)[:, None], axis=0)
        counters["div"] += div_cost(Kf) * J * int(n)

    # Fine trapezoid of c_eta on [0,1], plus the log a_j offset.
    bf = jnp.linspace(0.0, 1.0, 1025, dtype=dtype)
    rate_integrals = loga_all + jnp.trapezoid(rate_all(rate_final, rate_scale, h_off, bf), bf, axis=-1)
    _tick("final", t_final, final_logq)

    if verbose:
        print("[cat] phase times " + ", ".join(f"{k} {v:.0f}s" for k, v in phase_times.items()))
        print(f"[cat] done in {time.time() - t_start:.1f}s | K {Kf} | "
              f"final ESS {[round(float(v), 4) for v in _ess_ratio(flogw1)]} | "
              f"log C_j {[round(float(v), 4) for v in logC]} | "
              f"log a + int c {[round(float(v), 4) for v in rate_integrals]}")

    vxs, vlogw, _ = last_val
    return CATResult(
        field_params=phi_final,
        normalizer_rate_params=rate_final,
        modes=list(modes),
        betas_used=grid,
        final_snapshot=snap,
        final_comp_snapshot=comp,
        final_logq=final_logq,
        final_logC=logC,
        final_weights=weights,
        train_loss_history=train_loss_history,
        ess_ratio_history=ess_ratio_history,
        kinetic_energy_history=kinetic_energy_history,
        normalizer_rate_integrals=rate_integrals,
        snapshots=[vxs[k] for k in range(vxs.shape[0])],
        logw_snapshots=[vlogw[k] for k in range(vlogw.shape[0])],
        train_ess_history=train_ess_history,
        grid_history=grid_history,
        diagnostics=diagnostics,
        train_step_losses=train_step_losses,
        wall_time=time.time() - t_start - t_callback,
        train_start=train_start,
        compile_time=compile_seconds(t_start, time.time(), cb_windows),
        n_target_evals=counters["target"],
        n_target_grad_evals=counters["grad"],
        n_divergence_evals=counters["div"],
        _rollout_fn=partial(rollout_all, bs=grid),
        _inverse_fn=partial(inverse_all, bs=grid),
        _velocity_fn=velocity_all,
        _rate_fn=partial(rate_all, scale=rate_scale, off=h_off),
        selected_refresh=selected_refresh,
        phase_times=phase_times,
    )
