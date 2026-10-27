# gat.py
"""Global annealing transport (GAT).

The global-path counterpart of `cat.py`: instead of one
vector field per mode along the component-wise paths b_j exp{beta[log a_j + h]},
a single time-conditioned field is trained along the global geometric path

    gamma_beta(x) = b(x)^{1-beta} p~(x)^beta = b(x) exp{beta h(x)},
    h(x) = log p~(x) - log b(x),

from the Laplace-mixture base b = sum_j w_j N(mu_j, c^2 Sigma_j) built from a
ModeExplore inventory (`_mode_base.LaplaceMixtureBase`, the same base as the
`modes=` option of the other baselines; c = `mode_base_inflate`).

Everything else is cat's machinery with J = 1: the field

    v(beta, x) = L f(u, beta),     u = L^{-1}(x - m),

(`vector_field.VectorMLP`) on coordinates whitened by the moment-matched mean m
and covariance L L^T of b (`field_whitening="mixture"`; "identity" uses raw x),
fixed-step RK4 of (x, log q) with exact divergence, and the squared continuity
residual

    R = div v + s_beta . v + h - c_eta(beta),
    s_beta = grad log b + beta grad h = (1 - beta) grad log b + beta grad log p~,

with a jointly trained normalizer rate c_eta(beta) ~ E_{p_beta}[h], at the
particles of the frozen-field rollout on an adaptive ESS grid (`_ess_betas`),
with replay, best-ESS selection and the same optional loss variants.

Since b is normalized, log Z = int_0^1 E_{p_beta}[h] dbeta; GAT reports both
that rate integral and the endpoint importance estimate
logmeanexp(log p~ - log q_1). No mixture assembly or weight calibration is
needed: the endpoint q_1 is one flow, so its log density comes straight from
the forward rollout. The mode weights of b are *not* corrected separately;
any error in them has to be transported by the field (and shows up in the
endpoint ESS).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax

from .cat import (
    RateMLP, _ess_betas, _ess_ratio, _logmeanexp, _rk4_rollout, _trapezoid_weights,
    _validate_betas,
)
from .compile_timer import compile_seconds
from ..mode_explore.explore import Mode
from ..flow.vector_field import VectorMLP, beta_features, count_params, make_vec_fns
from ..utils.numeric import chol_jitter_jax, sym_jax
from ._mode_base import build_mode_base

Array = jax.Array


@dataclass
class GATResult:
    field_params: Any
    normalizer_rate_params: Any
    betas_used: Array                 # final checkpoint grid (adaptive: an output)

    samples: Array                    # (n, d) endpoint draws x_1
    final_logq: Array                 # (n,) log q_1 from the forward rollout
    final_logw: Array                 # (n,) log p~ - log q_1
    log_Z: float                      # logmeanexp(final_logw)
    log_Z_rate: float                 # int_0^1 c_eta
    final_ess: float                  # relative ESS of final_logw

    train_loss_history: List[float]
    ess_ratio_history: List[Array]        # per validation: (K,) ESS / n on that round's grid
    kinetic_energy_history: List[Array]   # per validation: (K,) E_q |v|^2
    snapshots: List[Array]                # per beta: (n_validation, d), selected validation rollout
    logw_snapshots: List[Array]           # per beta: (n_validation,)

    train_ess_history: List[Array] = field(default_factory=list)
    grid_history: List[np.ndarray] = field(default_factory=list)
    diagnostics: List[Dict[str, Any]] = field(default_factory=list)
    train_step_losses: List[Array] = field(default_factory=list)
    wall_time: float = 0.0
    n_target_evals: int = 0
    n_target_grad_evals: int = 0
    n_divergence_evals: int = 0
    n_modes: int = 0
    n_params: int = 0
    selected_refresh: int = -1
    phase_times: Dict[str, float] = field(default_factory=dict)
    train_start: Optional[float] = None    # wall time (s) to the first training step (setup, compile)
    compile_time: float = 0.0              # JAX compile wall time (s), within wall_time

    _rollout_fn: Optional[Callable] = field(default=None, repr=False)
    _inverse_fn: Optional[Callable] = field(default=None, repr=False)
    _velocity_fn: Optional[Callable] = field(default=None, repr=False)
    _log_b: Optional[Callable] = field(default=None, repr=False)    # (n, d) -> (n,) base log density

    @property
    def final_snapshot(self) -> Array:   # cat's name for the draws
        return self.samples

    def rollout(self, key: Array, n: int, return_path: bool = True):
        """Fresh rollout from x_0 ~ b: (K, n, d), (K, n), or the endpoint only."""
        xs, lqs = self._rollout_fn(self.field_params, key, int(n))
        return (xs, lqs) if return_path else (xs[-1], lqs[-1])

    def sample(self, key: Array, n: int) -> Array:
        return self.rollout(key, n, return_path=False)[0]

    def log_prob(self, x: Array) -> Array:
        """log q_1(x) by the inverse ODE (one RK4 pass; agrees with the forward
        log q up to integrator error)."""
        return self._inverse_fn(self.field_params, jnp.atleast_2d(jnp.asarray(x)))

    def velocity(self, x: Array, beta: float) -> Array:
        return self._velocity_fn(self.field_params, jnp.atleast_2d(x), beta)


def field_param_count(d: int, hidden: int, *, depth: int = 3, spline_knots: int = 0,
                      gradient_field: bool = False) -> int:
    """Parameter count of one `VectorMLP` field on R^d (no allocation)."""
    model = VectorMLP(hidden=int(hidden), depth=int(depth), spline_knots=int(spline_knots),
                      gradient=bool(gradient_field))
    shapes = jax.eval_shape(lambda: model.init(jr.PRNGKey(0), jnp.zeros((int(d),)),
                                               beta_features(jnp.asarray(0.0)))["params"])
    return count_params(shapes)


def width_for_params(d: int, target: int, *, depth: int = 3, spline_knots: int = 0,
                     gradient_field: bool = False) -> int:
    """Width whose field parameter count is closest to `target` (the count
    is increasing in the width, so bisection)."""
    count = lambda h: field_param_count(d, h, depth=depth, spline_knots=spline_knots,
                                        gradient_field=gradient_field)
    lo, hi = 1, 2
    while count(hi) < target:
        lo, hi = hi, 2 * hi
    while hi - lo > 1:
        mid = (lo + hi) // 2
        lo, hi = (mid, hi) if count(mid) < target else (lo, mid)
    return lo if target - count(lo) <= count(hi) - target else hi


def gat_fit(
    key: Array,
    logpi: Callable[[Array], Array],
    logpi_single: Callable[[Array], Array],
    d: int,
    *,
    modes: Sequence[Mode],
    mode_base_inflate: float = 1.0,
    field_whitening: str = "mixture",
    betas: Optional[Sequence[float]] = None,
    adaptive_target_ess: Optional[float] = 0.5,
    adaptive_delta_max: float = 1.0,
    grad_logpi_single: Optional[Callable[[Array], Array]] = None,
    n: int = 4096,
    n_train: int = 1024,
    n_validation: int = 256,
    n_refresh: int = 20,
    train_steps_per_refresh: int = 250,
    batch_size: Optional[int] = None,
    rk4_substeps: int = 4,
    hidden: int = 64,
    target_params: Optional[int] = None,
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
    validate_every: int = 1,
    select_best: bool = True,
    particle_chunk: Optional[int] = None,
    replay_refreshes: int = 1,
    replay_max_gb: float = 8.0,
    field_dtype=jnp.float32,
    dtype=jnp.float64,
    verbose: bool = True,
    callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> GATResult:
    """Fit one vector field along the global path b^{1-beta} p~^beta.

    Arguments share cat's meaning (see there), with J = 1:
      modes, mode_base_inflate: the ModeExplore inventory and the inflation c
                    of the base b (as for the other baselines' `modes=`).
      field_whitening: "mixture" (whiten by b's mean/covariance) or "identity".
      betas / adaptive_target_ess: fixed grid, or (default) adaptive grid with
                    this target incremental relative ESS; give exactly one
                    (pass adaptive_target_ess=None with a fixed grid).
      n_train:      training trajectories per refresh (all of them are
                    collocation points at every checkpoint). cat uses
                    n_train *per mode*; J * n_train here matches its budget.
      n:            number of final draws.
      target_params: if given, overrides `hidden` with the width whose field
                    parameter count is closest to this (`width_for_params`).
                    cat trains J fields of width `hidden`; passing
                    J * (its per-mode count) gives GAT the same capacity.
    """
    t_start = time.time()
    t_callback = 0.0
    cb_windows: List[Tuple[float, float]] = []    # callback intervals (not charged)
    train_start = None                            # elapsed time at the first training step
    if rk4_substeps < 1:
        raise ValueError("rk4_substeps must be >= 1")
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
    if field_whitening not in ("mixture", "identity"):
        raise ValueError(f"field_whitening must be 'mixture' or 'identity', got {field_whitening!r}")

    adaptive = betas is None
    if adaptive:
        if adaptive_target_ess is None or not (0.0 < adaptive_target_ess <= 1.0):
            raise ValueError("betas=None requests adaptive scheduling: give adaptive_target_ess in (0, 1]")
        if adaptive_delta_max <= 0.0:
            raise ValueError("adaptive_delta_max must be > 0")
        grid = None
    else:
        if adaptive_target_ess is not None:
            raise ValueError("give either betas or adaptive_target_ess (set it to None), not both")
        grid = _validate_betas(betas, dtype)
    del betas

    d = int(d)
    mode_base = build_mode_base(modes, mode_base_inflate, dtype)
    if mode_base is None:
        raise ValueError("gat_fit needs a ModeExplore inventory (modes=...)")
    if mode_base.dim != d:
        raise ValueError(f"modes have dimension {mode_base.dim}, expected d={d}.")
    substeps = int(rk4_substeps)
    n_train, n_validation = int(n_train), int(n_validation)
    batch_size = n_train if batch_size is None else int(batch_size)

    # ---------------- base and whitening ----------------
    if field_whitening == "mixture":
        w_b = jnp.exp(mode_base.log_w)
        m_f = w_b @ mode_base.mu
        cov_comp = jnp.einsum("jab,jcb->jac", mode_base.L, mode_base.L)
        dmu = mode_base.mu - m_f
        cov_b = jnp.einsum("j,jab->ab", w_b, cov_comp + dmu[:, :, None] * dmu[:, None, :])
        L_f = chol_jitter_jax(sym_jax(cov_b))
    else:
        m_f = jnp.zeros((d,), dtype)
        L_f = jnp.eye(d, dtype=dtype)

    log_b = mode_base.log_prob
    if grad_logpi_single is None:
        grad_logpi_single = jax.grad(logpi_single)

    @jax.jit
    def eval_terms(x):            # (N, d) -> (log b, h), each (N,)
        lb = log_b(x)
        return lb, logpi(x) - lb

    def score_terms_single(x):    # (grad log b, grad h) at one point
        glb = jax.grad(log_b)(x)
        return glb, grad_logpi_single(x) - glb

    score_terms = jax.jit(jax.vmap(score_terms_single))

    # ---------------- networks ----------------
    fdt = dtype if field_dtype is None else field_dtype
    if target_params is not None:
        hidden = width_for_params(d, int(target_params), depth=depth,
                                  spline_knots=spline_knots, gradient_field=gradient_field)
    model = VectorMLP(hidden=int(hidden), depth=int(depth), act=activation,
                      spline_knots=int(spline_knots), gradient=bool(gradient_field))
    vel_div, vel_div_b = make_vec_fns(model, None if field_dtype is None else fdt)
    k_init, k_rate, key = jr.split(key, 3)
    phi_params = model.init(k_init, jnp.zeros((d,), fdt), beta_features(jnp.asarray(0.0, fdt)))["params"]
    phi_params = jax.tree_util.tree_map(lambda a: a.astype(fdt), phi_params)

    rate_model = RateMLP(act=activation)
    rate_params = rate_model.init(k_rate, beta_features(jnp.asarray(0.0, fdt)), jnp.asarray(1.0, fdt))["params"]
    rate_params = jax.tree_util.tree_map(lambda a: a.astype(fdt), rate_params)
    rate_scale = jnp.asarray(1.0, dtype)     # set from the first collocation pool
    h_off = jnp.asarray(0.0, dtype)          # c_eta offset (center_h)

    def rate_single(rp, beta, scale):
        out = rate_model.apply({"params": rp}, beta_features(jnp.asarray(beta).astype(fdt)),
                               jnp.asarray(scale).astype(fdt))
        return out.astype(dtype)

    @jax.jit
    def rate_on(rp, scale, off, bs):
        return off + jax.vmap(lambda b: rate_single(rp, b, scale))(bs)

    # ---------------- rollouts ----------------
    chunk = None if particle_chunk is None else int(particle_chunk)

    def chunked(fn, args, n_, in_axis=0, out_axis=0):
        """fn(*args) in particle chunks along in_axis (last chunk padded by
        repetition), outputs concatenated along out_axis."""
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
        x0 = mode_base.sample(key_, n_)
        return x0, log_b(x0)

    @jax.jit
    def push(params, x0, lq0, bs):
        return _rk4_rollout(vel_div, params, x0, lq0, bs, m_f, L_f, substeps)   # (K,n,d), (K,n)

    def rollout_all(params, key_, n_, bs):
        x0, lq0 = initial_draws(key_, int(n_))
        return chunked(lambda a, b: push(params, a, b, bs), (x0, lq0), int(n_), in_axis=0, out_axis=1)

    @jax.jit
    def _inverse(params, x, bs):
        xs, acc = _rk4_rollout(vel_div, params, x, jnp.zeros((x.shape[0],), dtype), bs[::-1],
                               m_f, L_f, substeps)
        return log_b(xs[-1]) - acc[-1]

    def inverse_all(params, x, bs):
        return chunked(lambda xc: _inverse(params, xc, bs), (x,), x.shape[0])

    @jax.jit
    def velocity_all(params, x, beta):
        return vel_div(params, x, beta, m_f, L_f)[0]

    @jax.jit
    def kinetic(params, xs, bs):      # (K, n)
        return jax.vmap(lambda xk, bk: jnp.sum(vel_div(params, xk, bk, m_f, L_f)[0] ** 2, -1))(xs, bs)

    def eval_path(xs):                # (K, n, d) -> log b, h at every particle, (K, n)
        flat = xs.reshape(-1, d)
        lb, h = chunked(eval_terms, (flat,), flat.shape[0])
        return lb.reshape(xs.shape[:-1]), h.reshape(xs.shape[:-1])

    # ---------------- loss ----------------
    def residuals(phi_p, rate_p, scale, off, x, beta, h, glb, gh):
        v, div = vel_div_b(phi_p, x, beta, m_f, L_f)
        s = glb + beta[:, None] * gh
        cval = jax.vmap(lambda b: rate_single(rate_p, b, scale))(beta)
        return div + jnp.sum(s * v, axis=-1) + (h - off) - cval, v

    def loss_fn(params, batch, scale, off):
        r, v = residuals(params[0], params[1], scale, off, batch["x"], batch["beta"],
                         batch["h"], batch["glb"], batch["gh"])
        z2 = r * r if residual_scale == "none" else batch["rw"] * r * r
        if residual_loss == "l2":
            loss_val = jnp.mean(z2)
        else:
            dl2 = huber_delta ** 2
            loss_val = jnp.mean(2.0 * dl2 * (jnp.sqrt(1.0 + z2 / dl2) - 1.0))
        if velocity_l2_weight:
            loss_val = loss_val + velocity_l2_weight * jnp.mean(jnp.sum(v * v, axis=-1))
        return loss_val

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
        def step(carry, b):
            params, opt_state = carry
            l, g = jax.value_and_grad(loss_fn)(params, b, scale, off)
            upd, opt_state = tx.update(g, opt_state, params)
            params = optax.apply_updates(params, upd)
            return (params, opt_state), (l, optax.global_norm(g))

        (params, opt_state), (ls, gn) = jax.lax.scan(step, (params, opt_state), batches)
        return params, opt_state, ls, gn

    # ---------------- refresh helpers ----------------
    counters = dict(target=0, grad=0, div=0)

    def div_cost(K):
        return (K - 1) * substeps * 4

    def do_rollout(params, key_, n_, bs):
        xs, lqs = rollout_all(params, key_, n_, bs)
        lbs, hs = eval_path(xs)
        logw = lbs + bs[:, None] * hs - lqs
        counters["target"] += int(bs.shape[0]) * n_
        counters["div"] += div_cost(int(bs.shape[0])) * n_
        return xs, lqs, hs, logw

    def build_pool(xs, hs):
        """Host copy of the collocation pool (K * n_train points) with the
        drawing probability of every point (its slice's trapezoid weight,
        split evenly over the slice)."""
        K = xs.shape[0]
        flat = jax.lax.stop_gradient(xs).reshape(-1, d)
        glb, gh = chunked(score_terms, (flat,), flat.shape[0])
        counters["grad"] += K * n_train
        pool = dict(x=np.asarray(flat, np.float32), h=np.asarray(hs.reshape(-1)),
                    glb=np.asarray(glb, np.float32), gh=np.asarray(gh, np.float32),
                    beta=np.repeat(np.asarray(grid), n_train),
                    pw=np.repeat(np.asarray(_trapezoid_weights(grid)) / n_train, n_train))
        if residual_scale == "beta":
            rw = 1.0 / np.maximum(np.std(np.asarray(hs), axis=-1), 1.0) ** 2
            pool["rw"] = np.repeat(rw, n_train)
        return pool

    fields = ("x", "h", "glb", "gh", "beta") + (("rw",) if residual_scale == "beta" else ())

    def gather_batches(rng, pooled, cdf, steps):
        idx = np.minimum(np.searchsorted(cdf, rng.random((steps, batch_size))), cdf.shape[0] - 1)
        return jax.device_put({f: pooled[f][idx] for f in fields})

    def place(bs, lw, hs_):
        new = _ess_betas(bs, lw[:, None, :], hs_[:, None, :], adaptive_target_ess,
                         adaptive_delta_max, max_betas=200, bucket=2)
        if verbose:
            print(f"  [grid] K={len(new)} | betas " + " ".join(f"{b:.3g}" for b in new))
        return jnp.asarray(new, dtype)

    phase_times = dict(rollout=0.0, schedule=0.0, pool=0.0, train=0.0, validation=0.0, final=0.0)

    def _tick(name, t, out=None):
        if out is not None:
            jax.block_until_ready([a for a in jax.tree_util.tree_leaves(out) if isinstance(a, jax.Array)])
        now = time.time()
        phase_times[name] += now - t
        return now

    # ---------------- outer loop ----------------
    train_loss_history: List[float] = []
    ess_ratio_history: List[Array] = []
    train_ess_history: List[Array] = []
    kinetic_energy_history: List[Array] = []
    grid_history: List[np.ndarray] = []
    diagnostics: List[Dict[str, Any]] = []
    train_step_losses: List[Array] = []
    pool_hist: List[Dict[str, np.ndarray]] = []
    last_val, last_roll = None, None
    best_score, best = -np.inf, None
    selected_refresh = int(n_refresh) - 1
    np_rng = np.random.default_rng(int(jr.randint(jr.fold_in(key, 7), (), 0, 2 ** 31 - 1)))

    if verbose:
        sched = (f"K adaptive (target ESS {adaptive_target_ess}, delta_max {adaptive_delta_max})"
                 if adaptive else f"K={int(grid.shape[0])}")
        print(f"[gat] modes={mode_base.n_modes} d={d} {sched} substeps={substeps} "
              f"n_train={n_train} n_val={n_validation} refresh={n_refresh}x{train_steps_per_refresh} "
              f"hidden={int(hidden)} params={count_params(phi_params)}")

    def _run_callback(r):
        """Evaluation only, as cat's callback (single field); time excluded."""
        nonlocal t_callback
        if callback is None:
            return
        t_cb = time.time()
        p_now, g_now = train_params[0], grid
        callback(dict(
            refresh=r, elapsed=t_cb - t_start - t_callback, train_start=train_start,
            compile_s=compile_seconds(t_start, t_cb, cb_windows), params=p_now, grid=g_now,
            rollout=lambda k_, n_: rollout_all(p_now, k_, int(n_), g_now),
            velocity=lambda x_, b_: velocity_all(p_now, x_, b_)))
        t_end = time.time()
        t_callback += t_end - t_cb
        cb_windows.append((t_cb, t_end))

    for r in range(int(n_refresh)):
        t0 = time.time()
        tr = dict(phase_times)
        key, k_roll, k_val = jr.split(key, 3)

        t = t0
        if adaptive and r > 0:
            grid = place(*last_roll)
            t = _tick("schedule", t)
        if adaptive and r == 0:
            # Zero field (identity flow): every checkpoint holds the draws
            # x_0 ~ b, so the grid is placed on them without a rollout.
            x0, lq0 = initial_draws(k_roll, n_train)
            lb0, h0 = eval_terms(x0)
            counters["target"] += n_train
            grid = place(jnp.zeros((1,), dtype), jnp.zeros((1, n_train), dtype), h0[None])
            Kc = int(grid.shape[0])
            xs = jnp.broadcast_to(x0[None], (Kc,) + x0.shape)
            hs = jnp.broadcast_to(h0[None], (Kc,) + h0.shape)
            logw = lb0[None] + grid[:, None] * hs - lq0[None]
            t = _tick("schedule", t, logw)
        else:
            xs, _, hs, logw = do_rollout(train_params[0], k_roll, n_train, grid)
            t = _tick("rollout", t, logw)
        last_roll = (grid, logw, hs)
        grid_history.append(np.asarray(grid))
        train_ess_history.append(_ess_ratio(logw))

        pool = build_pool(xs, hs)
        t = _tick("pool", t)
        if r == 0:
            hp = jnp.asarray(pool["h"])
            if center_h:
                h_off = jnp.mean(hp).astype(dtype)
                spread = jnp.std(hp - h_off)
            else:
                spread = jnp.std(hp) + jnp.abs(jnp.mean(hp))
            rate_scale = jnp.maximum(spread, 1.0).astype(dtype)
        pool_hist = (pool_hist + [pool])[-int(replay_refreshes):]
        while len(pool_hist) > 1 and 2 * sum(a.nbytes for P in pool_hist for a in P.values()) > replay_max_gb * 2 ** 30:
            pool_hist = pool_hist[1:]
        pooled = {f: np.concatenate([P[f] for P in pool_hist]) for f in pool_hist[0]}
        cdf = np.cumsum(pooled.pop("pw"))
        cdf = cdf / cdf[-1]
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
            steps_left -= s_
        del pooled, cdf
        ls, gn = jnp.concatenate(ls_all), jnp.concatenate(gn_all)
        t = _tick("train", t, ls)
        counters["div"] += int(train_steps_per_refresh) * batch_size
        train_step_losses.append(ls)
        train_loss_history.append(float(ls[-1]))

        if (r + 1) % int(validate_every) != 0 and r != int(n_refresh) - 1:
            if verbose:
                print(f"[refresh {r + 1:3d}/{n_refresh}] K {int(grid.shape[0])} "
                      f"train {train_loss_history[-1]:.4e} | |g| {float(jnp.mean(gn)):.2e} "
                      f"| {time.time() - t0:.1f}s")
            _run_callback(r)
            continue

        # ---- fresh validation rollout with the updated field ----
        vxs, vlqs, vhs, vlogw = do_rollout(train_params[0], k_val, n_validation, grid)
        last_roll = (grid, vlogw, vhs)
        ess = _ess_ratio(vlogw)                                   # (K,)
        ke = jnp.mean(kinetic(train_params[0], vxs, grid), axis=-1)
        omega_v = jax.nn.softmax(vlogw, axis=-1)
        diag: Dict[str, Any] = dict(
            refresh=r, betas=grid, n_betas=int(grid.shape[0]), ess=ess,
            train_ess=train_ess_history[-1], logw_var=jnp.var(vlogw, -1), kinetic=ke,
            c_eta=rate_on(train_params[1], rate_scale, h_off, grid),
            h_mean_is=jnp.sum(omega_v * vhs, -1),
            grad_norm_mean=float(jnp.mean(gn)), grad_norm_max=float(jnp.max(gn)),
            nonfinite=int(jnp.sum(~jnp.isfinite(vxs)) + jnp.sum(~jnp.isfinite(vlqs))),
        )
        ess_ratio_history.append(ess)
        kinetic_energy_history.append(ke)
        _tick("validation", t, diag)
        diag["time"] = time.time() - t0
        diag["phase_times"] = {k: phase_times[k] - tr[k] for k in phase_times}
        diagnostics.append(diag)
        last_val = (vxs, vlogw, grid)

        score = float(jnp.nan_to_num(ess[-1], nan=0.0))
        if select_best and diag["nonfinite"] == 0 and score > best_score:
            best_score, selected_refresh = score, r
            best = (train_params, last_val)

        _run_callback(r)

        if verbose:
            msg = (f"[refresh {r + 1:3d}/{n_refresh}] K {int(grid.shape[0])} "
                   f"train {train_loss_history[-1]:.4e} | ESS(beta=1) {float(ess[-1]):.3f} "
                   f"min {float(jnp.min(ess)):.3f} | |g| {diag['grad_norm_mean']:.2e} "
                   f"| {diag['time']:.1f}s")
            if diag["nonfinite"]:
                msg += f" | NONFINITE {diag['nonfinite']}"
            print(msg)

    if select_best and best is not None:
        train_params, last_val = best
        if verbose:
            print(f"[gat] selected refresh {selected_refresh + 1} (endpoint ESS {best_score:.4f})")
    phi_final, rate_final = train_params

    # ---------------- final draws ----------------
    t_final = time.time()
    key, k_final = jr.split(key)
    grid = last_val[2]
    Kf = int(grid.shape[0])
    fxs, flqs = rollout_all(phi_final, k_final, int(n), grid)
    x1, lq1 = fxs[-1], flqs[-1]
    del fxs, flqs
    lpi1 = chunked(jax.jit(logpi), (x1,), int(n))
    counters["target"] += int(n)
    counters["div"] += div_cost(Kf) * int(n)
    logw1 = lpi1 - lq1
    log_Z = float(_logmeanexp(logw1))
    final_ess = float(_ess_ratio(logw1))
    bf = jnp.linspace(0.0, 1.0, 1025, dtype=dtype)
    log_Z_rate = float(jnp.trapezoid(rate_on(rate_final, rate_scale, h_off, bf), bf))
    _tick("final", t_final, logw1)

    if verbose:
        print("[gat] phase times " + ", ".join(f"{k} {v:.0f}s" for k, v in phase_times.items()))
        print(f"[gat] done in {time.time() - t_start:.1f}s | K {Kf} | final ESS {final_ess:.4f} "
              f"| log Z (IS) {log_Z:.4f} | log Z (rate) {log_Z_rate:.4f}")

    vxs, vlogw, _ = last_val
    return GATResult(
        field_params=phi_final,
        normalizer_rate_params=rate_final,
        betas_used=grid,
        samples=x1,
        final_logq=lq1,
        final_logw=logw1,
        log_Z=log_Z,
        log_Z_rate=log_Z_rate,
        final_ess=final_ess,
        train_loss_history=train_loss_history,
        ess_ratio_history=ess_ratio_history,
        kinetic_energy_history=kinetic_energy_history,
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
        n_modes=mode_base.n_modes,
        n_params=count_params(phi_params),
        selected_refresh=selected_refresh,
        phase_times=phase_times,
        _rollout_fn=partial(rollout_all, bs=grid),
        _inverse_fn=partial(inverse_all, bs=grid),
        _velocity_fn=velocity_all,
        _log_b=log_b,
    )
