"""Per-mode free-form continuous normalizing flow, trained by reverse KL.

The baseline for `cat` that keeps everything but the path: the same
anchors b_j = N(mu_j, Sigma_j), the same per-mode endpoint targets
    pi_j(x) = a_j b_j(x) exp(h(x)) = pi(x) a_j b_j(x) / sum_k a_k b_k(x)
(so sum_j C_j = Z, as in cat), the same vector-field network
(`vector_field.VectorMLP`, shared code) on whitened coordinates
u = L_j^{-1}(x - mu_j) with the same time features, the same
fixed-step RK4 on the same time grid, exact divergence, the same validation
and best-ESS selection, and the same mixture assembly at the end.

What differs: there is no prescribed path between b_j and pi_j; the field is trained by
minimizing KL(q_j || pi_j / C_j) = E_{x_0 ~ b_j}[log q_j(x_1) - log pi_j(x_1)]
+ const, backpropagating through the solver (discretize-then-optimize). Each
step needs log pi and grad log pi at every endpoint particle, which is what
the oracle budget pays for.
"""
from __future__ import annotations

import time
from functools import partial
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import jax.random as jr
import flax.linen as nn
import numpy as np
import optax
from jax.scipy.special import logsumexp

from .compile_timer import compile_seconds
from ..mode_explore.explore import Mode
from .cat import _ess_ratio, _gauss_logpdf, _logmeanexp, _rk4_rollout, _validate_betas
from ..flow.vector_field import VectorMLP, beta_features, count_params, make_vec_fns
from ..utils.numeric import sym_jax, chol_jitter_jax

Array = jax.Array


@dataclass
class CNFResult:
    params: Any
    modes: List[Mode]
    betas_used: Array
    final_snapshot: Array
    final_comp_snapshot: Array
    final_logq: Array
    final_logC: Array
    final_weights: Array
    train_loss_history: List[float]
    ess_ratio_history: List[Array]          # per validation: (J,) endpoint ESS
    wall_time: float
    n_target_evals: int
    n_target_grad_evals: int
    n_params_per_mode: int
    selected_refresh: Optional[int]
    phase_times: Dict[str, float]
    train_start: Optional[float] = None    # wall time (s) to the first training step (setup, compile)
    compile_time: float = 0.0              # JAX compile wall time (s), within wall_time
    diagnostics: List[Dict[str, Any]] = field(default_factory=list)
    _inverse_fn: Optional[Callable] = field(default=None, repr=False)
    _path_fn: Optional[Callable] = field(default=None, repr=False)
    _velocity_fn: Optional[Callable] = field(default=None, repr=False)

    def rollout(self, key: Array, n: int):
        """Fresh draws x_0 ~ b_j pushed along the field, all modes, at every grid
        point: (K, J, n, d) and (K, J, n) (the same layout as `CATResult.rollout`)."""
        return self._path_fn(self.params, key, int(n))

    def velocity(self, x: Array, beta: float) -> Array:
        """v_j(t=beta, x) of every mode, (J, n, d)."""
        return self._velocity_fn(self.params, jnp.atleast_2d(x), beta)

    def mixture_log_prob(self, x: Array) -> Array:
        """log sum_j hat w_j q_1^j(x) via J inverse ODE solves (one RK4 pass each)."""
        lq = self._inverse_fn(self.params, jnp.atleast_2d(jnp.asarray(x)))   # (J, n)
        return logsumexp(lq + jnp.log(self.final_weights)[:, None], axis=0)


def cnf(
    *,
    key: Array,
    logpi: Callable[[Array], Array],
    modes: List[Mode],
    betas: Sequence[float],
    n: int = 4096,
    batch_size: int = 64,
    n_validation: int = 256,
    n_refresh: int = 20,
    train_steps_per_refresh: int = 250,
    validate_every: int = 5,
    select_best: bool = True,
    rk4_substeps: int = 4,
    hidden: int = 128,
    depth: int = 3,
    activation: str = "silu",
    spline_knots: int = 0,
    gradient_field: bool = False,
    lr: float = 1e-3,
    grad_clip: float = 5.0,
    final_calibration: Optional[int] = None,
    field_dtype=jnp.float32,
    mvn_jitter0: float = 1e-10,
    dtype=jnp.float64,
    verbose: bool = True,
    callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> CNFResult:
    """Train J independent CNFs b_j -> pi_j by reverse KL; return the mixture.

      logpi:       batched unnormalized log target (N, d) -> (N,), differentiable.
      betas:       solver time grid on [0, 1] (checkpoints of the fixed-step
                   RK4, rk4_substeps steps per interval), as in cat.
      batch_size:  endpoint particles per mode per optimizer step; each costs
                   one value and one gradient of log pi.
      n_validation: fresh particles per mode per validation (values only),
                   scored by endpoint ESS; the best round is returned if
                   select_best.
      final_calibration: as in cat.
      callback:    as in cat, called after every refresh with refresh, elapsed,
                   params, grid (betas), rollout(key, n) -> (xs, logqs) along the
                   whole grid (K, J, n, ...), inverse(x) -> (J, n), velocity(x, t),
                   loga, mu, chol; its time is excluded from the wall time.
    Oracle cost: n_refresh * steps * J * batch_size values+grads,
    J * n_validation values per validation, J * n (or J * m) values at the end.
    """
    t_start = time.time()
    t_callback = 0.0
    cb_windows: List[Tuple[float, float]] = []    # callback intervals (not charged)
    train_start = None                            # elapsed time at the first training step
    betas = _validate_betas(betas, dtype)
    J = len(modes)
    d = int(jnp.asarray(modes[0].mu).shape[0])
    substeps, B = int(rk4_substeps), int(batch_size)
    mu_all = jnp.stack([jnp.asarray(m.anchor_mu if m.anchor_mu is not None else m.mu, dtype=dtype)
                        for m in modes])
    Sigma_all = jnp.stack([sym_jax(jnp.asarray(m.Sigma, dtype=dtype)) for m in modes])
    chol_all = chol_jitter_jax(Sigma_all, jitter0=mvn_jitter0, max_tries=12)
    loga_all = jnp.log(jnp.asarray([float(m.w) for m in modes], dtype=dtype))

    def log_pi_j(x):
        """(J, N, d) -> (J, N): log pi_j = log a_j + log b_j + log pi - log sum_k a_k b_k."""
        flat = x.reshape(-1, d)
        lb = jnp.stack([_gauss_logpdf(flat, mu_all[k], chol_all[k]) for k in range(J)], -1)  # (JN, J)
        lmix = logsumexp(lb + loga_all, -1)
        lp = logpi(flat) - lmix
        own = jax.vmap(_gauss_logpdf)(x, mu_all, chol_all)         # log b_j at mode j's particles
        return own + loga_all[:, None] + lp.reshape(J, -1)

    fdt = dtype if field_dtype is None else field_dtype
    model = VectorMLP(hidden=int(hidden), depth=int(depth), act=activation,
                      spline_knots=int(spline_knots), gradient=bool(gradient_field))
    vel_div = make_vec_fns(model, None if field_dtype is None else fdt)[0]
    vel_div_train = jax.checkpoint(vel_div)   # rematerialize in the backward pass (memory)
    # Same key handling as cat: identical initialization per mode.
    k_init, key = jr.split(key)
    init_one = lambda k: model.init(k, jnp.zeros((d,), fdt), beta_features(jnp.asarray(0.0, fdt)))["params"]
    params = jax.vmap(init_one)(jr.split(k_init, J))
    params = jax.tree_util.tree_map(lambda a: a.astype(fdt), params)
    n_params = count_params(jax.tree_util.tree_map(lambda a: a[0], params))

    def draws(key_, n_):
        eps = jr.normal(key_, (J, n_, d), dtype=dtype)
        x0 = mu_all[:, None, :] + jnp.einsum("jab,jnb->jna", chol_all, eps)
        return x0, jax.vmap(_gauss_logpdf)(x0, mu_all, chol_all)

    def endpoint(p, x0, lq0, vd):
        """(J, n, d), (J, n) at t = 1."""
        xs, lqs = jax.vmap(lambda pj, x, l, m, c: _rk4_rollout(vd, pj, x, l, betas, m, c, substeps))(
            p, x0, lq0, mu_all, chol_all)
        return xs[:, -1], lqs[:, -1]

    tx = optax.chain(optax.clip_by_global_norm(grad_clip), optax.adam(lr))
    opt_state = tx.init(params)

    def loss_fn(p, key_):
        x0, lq0 = draws(key_, B)
        x1, lq1 = endpoint(p, x0, lq0, vel_div_train)
        per_mode = jnp.mean(lq1 - log_pi_j(x1), -1)             # KL(q_j || pi_j) - log C_j
        return jnp.sum(per_mode), per_mode

    @jax.jit
    def train_chunk(p, s, key_):
        def step(carry, k):
            p, s = carry
            (l, _), g = jax.value_and_grad(loss_fn, has_aux=True)(p, k)
            upd, s = tx.update(g, s, p)
            return (optax.apply_updates(p, upd), s), l
        (p, s), ls = jax.lax.scan(step, (p, s), jr.split(key_, int(train_steps_per_refresh)))
        return p, s, ls

    @partial(jax.jit, static_argnums=2)
    def rollout_logw(p, key_, n_):
        x0, lq0 = draws(key_, n_)
        x1, lq1 = endpoint(p, x0, lq0, vel_div)
        return x1, lq1, log_pi_j(x1) - lq1

    @partial(jax.jit, static_argnums=2)
    def path_all(p, key_, n_):
        """(K, J, n, d), (K, J, n) along the grid."""
        x0, lq0 = draws(key_, n_)
        xs, lqs = jax.vmap(lambda pj, x, l, m, c: _rk4_rollout(vel_div, pj, x, l, betas, m, c, substeps))(
            p, x0, lq0, mu_all, chol_all)
        return jnp.swapaxes(xs, 0, 1), jnp.swapaxes(lqs, 0, 1)

    @jax.jit
    def velocity_all(p, x, beta):
        return jax.vmap(lambda pj, m, c: vel_div(pj, x, beta, m, c)[0])(p, mu_all, chol_all)

    @jax.jit
    def inverse_all(p, x):
        """log q_1^j(x) for all j, (J, n)."""
        zeros = jnp.zeros((x.shape[0],), dtype)
        def one(pj, m, c):
            xs, acc = _rk4_rollout(vel_div, pj, x, zeros, betas[::-1], m, c, substeps)
            return _gauss_logpdf(xs[-1], m, c) - acc[-1]
        return jax.vmap(one)(p, mu_all, chol_all)

    counters = dict(value=0, grad=0)
    phase = dict(train=0.0, validation=0.0, final=0.0)
    losses, ess_hist, diags = [], [], []
    best_score, best, selected = -np.inf, None, None
    if verbose:
        print(f"[cnf] J={J} d={d} K={len(betas)} params/mode={n_params} batch={B} "
              f"steps={n_refresh}x{train_steps_per_refresh}")
    for r in range(int(n_refresh)):
        t = time.time()
        key, k = jr.split(key)
        params, opt_state, ls = train_chunk(params, opt_state, k)
        if train_start is None:     # dispatched (compiled): the first step runs now; q = b until here
            train_start = time.time() - t_start - t_callback
        ls.block_until_ready()
        phase["train"] += time.time() - t
        counters["value"] += int(train_steps_per_refresh) * J * B
        counters["grad"] += int(train_steps_per_refresh) * J * B
        losses.append(float(jnp.mean(ls)))
        if (r + 1) % int(validate_every) == 0 or r == int(n_refresh) - 1:
            t = time.time()
            key, k = jr.split(key)
            _, _, vlw = rollout_logw(params, k, int(n_validation))
            counters["value"] += J * int(n_validation)
            ess = np.asarray(_ess_ratio(vlw))
            nonfinite = int(np.sum(~np.isfinite(np.asarray(vlw))))
            ess_hist.append(ess)
            score = float(np.mean(np.nan_to_num(ess)))
            diags.append(dict(refresh=r, ess=ess, nonfinite=nonfinite, logC=np.asarray(_logmeanexp(vlw))))
            if select_best and nonfinite == 0 and score > best_score:
                best_score, best, selected = score, params, r
            phase["validation"] += time.time() - t
            if verbose:
                print(f"[cnf refresh {r + 1}/{n_refresh}] loss {losses[-1]:.4f} "
                      f"ESS [{', '.join(f'{e:.3f}' for e in ess)}] mean {score:.3f} "
                      f"({time.time() - t_start:.0f}s)", flush=True)
        if callback is not None:
            t_cb = time.time()
            p_now = params
            callback(dict(
                refresh=r, elapsed=t_cb - t_start - t_callback, train_start=train_start,
                compile_s=compile_seconds(t_start, t_cb, cb_windows), params=p_now, grid=betas,
                rollout=lambda k_, n_: path_all(p_now, k_, int(n_)),
                inverse=lambda x_: inverse_all(p_now, x_),
                velocity=lambda x_, b_: velocity_all(p_now, x_, b_),
                loga=loga_all, mu=mu_all, chol=chol_all))
            t_end = time.time()
            t_callback += t_end - t_cb
            cb_windows.append((t_cb, t_end))
    if select_best and best is not None:
        params = best

    # ---- final draws and mixture, as cat ----
    t = time.time()
    key, k_final, k_comp, k_draw = jr.split(key, 4)
    if final_calibration is None:
        fx1, flq1, flogw = rollout_logw(params, k_final, int(n))
        counters["value"] += J * int(n)
        logC = _logmeanexp(flogw)
        weights = jax.nn.softmax(logC)
        comp = jr.categorical(k_comp, jnp.log(weights), shape=(int(n),))
        idx = jnp.arange(int(n))
    else:
        m = int(final_calibration)
        _, _, clogw = rollout_logw(params, k_final, m)
        counters["value"] += J * m
        logC = _logmeanexp(clogw)
        weights = jax.nn.softmax(logC)
        comp = jr.categorical(k_comp, jnp.log(weights), shape=(int(n),))
        comp_np = np.asarray(comp)
        counts = np.bincount(comp_np, minlength=J)
        x0, lq0 = draws(k_draw, int(counts.max()))
        fx1, flq1 = jax.jit(partial(endpoint, vd=vel_div))(params, x0, lq0)
        order = np.argsort(comp_np, kind="stable")
        rank = np.empty(int(n), dtype=np.int64)
        rank[order] = np.arange(int(n)) - np.concatenate([[0], np.cumsum(counts)[:-1]])[comp_np[order]]
        idx = jnp.asarray(rank)
    snap = fx1[comp, idx]
    if J == 1:
        final_logq = flq1[0, idx]
    else:
        lq_all = inverse_all(params, snap).at[comp, jnp.arange(int(n))].set(flq1[comp, idx])
        final_logq = logsumexp(lq_all + jnp.log(weights)[:, None], axis=0)
    final_logq.block_until_ready()
    phase["final"] += time.time() - t
    if verbose:
        print(f"[cnf] done {time.time() - t_start:.0f}s | phases "
              + ", ".join(f"{k} {v:.0f}s" for k, v in phase.items())
              + f" | selected {None if selected is None else selected + 1} | log C {np.round(np.asarray(logC), 3)}")
    return CNFResult(
        params=params, modes=list(modes), betas_used=betas, final_snapshot=snap,
        final_comp_snapshot=comp, final_logq=final_logq, final_logC=logC, final_weights=weights,
        train_loss_history=losses, ess_ratio_history=ess_hist, wall_time=time.time() - t_start - t_callback,
        train_start=train_start, compile_time=compile_seconds(t_start, time.time(), cb_windows),
        n_target_evals=counters["value"], n_target_grad_evals=counters["grad"],
        n_params_per_mode=n_params, selected_refresh=selected, phase_times=phase, diagnostics=diags,
        _inverse_fn=inverse_all, _path_fn=path_all, _velocity_fn=velocity_all,
    )


def cnf_cost(*, n, batch_size, n_validation, n_modes, n_refresh, train_steps_per_refresh,
                    validate_every, final_calibration=None):
    """(values, value+grads) mirroring cnf's counters."""
    j = int(n_modes)
    n_val_rounds = sum(1 for r in range(int(n_refresh))
                       if (r + 1) % int(validate_every) == 0 or r == int(n_refresh) - 1)
    train = int(n_refresh) * int(train_steps_per_refresh) * j * int(batch_size)
    final = j * int(n if final_calibration is None else final_calibration)
    return train + n_val_rounds * j * int(n_validation) + final, train
