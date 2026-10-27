# temperflow.py
"""TemperFlow (Qiu & Wang, "Efficient Multimodal Sampling via Tempered Distribution
Flow", JASA 119(546), 2024; arXiv:2304.03933; reference code at
`github.com/yixuan/temperflow`).

TemperFlow trains a *single global* normalizing flow (bounded RealNVP,
`src.flow.RealNVPBounded`) along an increasing sequence of tempered targets pi^beta_k,
warm-starting each stage from the previous stage's parameters.

**Adaptive temperature schedule.** As in the reference code (`tf/temperflow/flow.py`,
`train_flow` / `kl_grad`): after training at beta_k, self-normalized importance sampling
with the flow's own draws as the proposal estimates KL(pi_beta_k || pi) and its derivative
in beta (`_kl_grad_estimate`), and a Newton-type step in log beta targets a `kl_decay`
fraction reduction of that KL (`_temperflow_next_beta`): delta = -(1 - kl_decay) kl / beta
/ kl_grad, damped by an exponential moving average of past |delta| (factor 0.9).
log beta >= 0 terminates at beta = 1; `max_stages` caps the ladder. One addition: delta is
floored at `min_delta` so that Monte Carlo noise in kl_grad cannot stall the ladder.

**Per-stage loss.** Reverse KL against pi^beta_{k+1}: draw z from the base, push it
through the flow, x = f(z), and minimize E_z[-logdet(x; z) - beta_{k+1} log pi(x)] (the base
entropy is a beta-independent constant). This is the reverse-KL loss shared by the
flow baselines here, not the paper's L2-sampler objective.

**Mode-informed variant** (`modes=...`): mode information enters through the flow's base.
With `mode_base_tempered=True` (default), stage k+1 draws z ~ b_beta
(`_mode_base.LaplaceMixtureBase`), the Laplace approximation of pi^beta at the discovered
modes (covariances Sigma_j / beta, weights tempered alongside), so the identity flow is
exactly the tempered Laplace approximation at every stage and training learns a
correction. `mode_base_tempered=False` uses the fixed beta = 1 mixture b at every stage;
`modes=None` is the N(0, I) base.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jr
import optax

from ..flow import RealNVPBounded as RealNVP
from ._mode_base import build_mode_base

Array = jax.Array
LogPiBatch = Callable[[Array], Array]      # (N,d) -> (N,)
LogPiSingle = Callable[[Array], Array]     # (d,) -> scalar


def _ess_from_logw(logw: Array) -> Array:
    logw = logw - jnp.max(logw)
    w = jnp.exp(logw)
    w = w / jnp.sum(w)
    return 1.0 / jnp.sum(w * w)


def _select_next_beta(
    logq: Array,
    lpi: Array,
    beta_k: float,
    target_ess: float,
    tol: float = 1e-3,
    max_iters: int = 30,
    min_step: float = 1e-4,
    min_growth: float = 2.0,
) -> float:
    """Bisection search for the largest `beta_next` in `(beta_k, 1]` such
    that the self-normalized importance weights `beta_next*lpi - logq`
    reach ESS ratio `target_ess`. `logq`/`lpi` are precomputed at a probe
    batch drawn from the *current* proposal (the flow being tempered). Falls back to a minimum
    forward step if even the smallest step fails the target, to guarantee
    the ladder always progresses -- guaranteed via a *multiplicative* floor
    (`beta_k * min_growth`) once `beta_k > 0`, not an additive one: an
    additive floor is catastrophically slow once beta has to span
    several orders of magnitude (crossing from 1e-4 to 1 takes ~10000
    additive 1e-4 steps, vs. ~14 steps at a 2x multiplicative floor) --
    the reason the fixed ladders of the run scripts (`beta_schedule_geom`)
    are geometric, not arithmetic. The additive
    `min_step` is kept only for the very first step (`beta_k == 0`, where a
    multiplicative floor is degenerate): forcing a larger first jump from a
    fresh, untrained flow can destabilize training.
    """
    n_ = logq.shape[0]

    def ess_ratio_at(beta: float) -> float:
        logw = beta * lpi - logq
        return float(_ess_from_logw(logw)) / n_

    if ess_ratio_at(1.0) >= target_ess:
        return 1.0

    lo, hi = float(beta_k), 1.0
    for _ in range(int(max_iters)):
        mid = 0.5 * (lo + hi)
        if ess_ratio_at(mid) >= target_ess:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break

    beta_next = lo
    floor = beta_k * min_growth if beta_k > 0.0 else min_step
    if beta_next <= floor:
        beta_next = min(1.0, floor)
    return float(beta_next)


def _kl_grad_estimate(
    logpi_batch: LogPiBatch, x: Array, log_q_x: Array, beta: float
) -> Tuple[float, float]:
    """Self-normalized-importance-sampling estimate of KL(pi_beta || pi_true)
    and its derivative w.r.t. beta, vendored verbatim (math only, JAX instead
    of TensorFlow) from `github.com/yixuan/temperflow`'s `TemperFlow.kl_grad`
    (Qiu & Wang 2024, Sec. 3). `x` are samples from the importance proposal
    (here: the just-trained flow itself, `log_q_x` its log-density there --
    this repo's `ref_cutoff=None` simplification of the reference's own
    staleness-managed `ref` distribution, see module docstring). Their
    "energy" convention is `pi(x) = exp(-E(x))`, i.e. `E(x) = -logpi(x)`.
    """
    E = -logpi_batch(x)
    logr = log_q_x
    n_ = E.shape[0]
    logn = jnp.log(jnp.asarray(n_, dtype=E.dtype))
    beta_arr = jnp.asarray(beta, dtype=E.dtype)
    log_ratio = -beta_arr * E - logr
    zeta_1 = jax.scipy.special.logsumexp(-E - logr) - logn
    zeta_beta = jax.scipy.special.logsumexp(log_ratio) - logn
    ratio = jax.nn.softmax(log_ratio - zeta_beta) * n_
    Emean = jnp.mean(ratio * E)
    kl = (1.0 - beta_arr) * Emean + zeta_1 - zeta_beta
    kl_grad = (beta_arr - 1.0) * jnp.mean(ratio * (E - Emean) ** 2)
    return float(kl), float(kl_grad)


def _temperflow_next_beta(
    logbeta: float,
    kl: float,
    kl_grad: float,
    kl_decay: float,
    past_deltas: Sequence[float],
    is_first: bool,
    ema_fac: float = 0.9,
    min_delta: float = 1e-4,
) -> Tuple[float, float]:
    """Advance log(beta) via the reference implementation's own Newton-type
    update (`train_flow`): `delta = -(1-kl_decay)*kl/beta/kl_grad`, damped
    against an exponential moving average of `abs(past_deltas)` (their
    `exp_move_avg`, `fac=0.9`) once more than one step has been taken;
    `min(delta, 1.0)` only, on the very first step. `min_delta` is this
    repo's own addition (not in the reference) guaranteeing forward
    progress if MC noise flips `kl_grad`'s sign -- see module docstring.

    Returns `(beta_next, delta_used)`; `beta_next == 1.0` signals
    `logbeta + delta >= 0`, i.e. termination.
    """
    beta = math.exp(logbeta)
    kl_grad = kl_grad if abs(kl_grad) > 1e-300 else -1e-300
    delta = -(1.0 - kl_decay) * kl / beta / kl_grad

    if is_first:
        delta = min(delta, 1.0)
    else:
        past = np.abs(np.asarray(list(past_deltas), dtype=float))
        i = np.linspace(past.size - 1, 0, num=past.size)
        weights = ema_fac ** i
        delta_abs_avg = float(np.sum(past * weights) / np.sum(weights))
        delta = float(np.clip(abs(delta), 0.5 * delta_abs_avg, 2.0 * delta_abs_avg))

    delta = max(delta, min_delta)
    logbeta_new = logbeta + delta
    if logbeta_new >= 0.0:
        return 1.0, delta
    return math.exp(logbeta_new), delta


@dataclass
class TemperFlowResult:
    samples: Array                 # (n, d) final-stage flow samples
    q_log_prob: Any                # callable (N,d) -> (N,): flow log-density
    snapshots: List[Array]         # per-stage samples, aligned with betas_used[1:]
    ess_ratio_history: List[float]
    info_history: List[Dict[str, Any]]
    betas_used: List[float]        # the discovered adaptive ladder, starts at 0.0, ends at 1.0
    topup_steps: int = 0           # extra training steps at the final beta (`min_total_train_steps`)


def temperflow_fit(
    key: Array,
    logpi: LogPiBatch,
    logpi_single: Optional[LogPiSingle],
    d: int,
    *,
    beta0: float = 0.1,
    kl_decay: float = 0.8,
    max_stages: int = 30,
    n_probe: Optional[int] = None,
    n: int = 2048,
    n_train: Optional[int] = None,
    n_snapshot: Optional[int] = None,
    train_steps: int = 500,
    n_coupling: int = 6,
    hidden: int = 128,
    depth: int = 3,
    s_max: float = 3.0,
    s_bump: float = 1.0,
    t_max: float = float("inf"),
    t_bump: float = 1.0,
    lr: float = 1e-3,
    grad_clip: float = 5.0,
    modes: Optional[Sequence[Any]] = None,
    mode_base_inflate: float = 1.0,
    mode_base_tempered: bool = True,
    min_total_train_steps: int = 0,
    dtype=jnp.float64,
    verbose: bool = False,
) -> TemperFlowResult:
    """`min_total_train_steps`: if the adaptive ladder reaches beta=1 after fewer
    than this many training steps in total (stages x `train_steps`), the flow
    keeps training at the final beta for the difference. Used to keep the
    realized cost at or above a matched budget when the ladder is short."""
    if not 0.0 < beta0 < 1.0:
        raise ValueError("beta0 must lie in (0, 1).")
    if not 0.0 < kl_decay < 1.0:
        raise ValueError("kl_decay must lie in (0, 1).")

    d = int(d)
    n = int(n)
    _n_train = int(n) if n_train is None else int(n_train)
    _n_snapshot = int(n) if n_snapshot is None else int(n_snapshot)
    _n_probe = int(n) if n_probe is None else int(n_probe)

    mode_base = build_mode_base(modes, mode_base_inflate, dtype)
    if mode_base is not None and mode_base.dim != d:
        raise ValueError(f"modes have dimension {mode_base.dim}, expected d={d}.")

    def base_beta(beta):
        # beta of the base distribution used while training at `beta`.
        return beta if mode_base_tempered else 1.0

    def base_sample(key_, m, beta):
        if mode_base is None:
            return jr.normal(key_, (int(m), d), dtype=dtype)
        return mode_base.sample(key_, m, base_beta(beta))

    def base_log_prob(z, beta):
        if mode_base is None:
            d_ = jnp.asarray(z.shape[-1], dtype=z.dtype)
            return -0.5 * (jnp.sum(z * z, axis=-1) + d_ * jnp.log(jnp.asarray(2.0 * jnp.pi, dtype=z.dtype)))
        return mode_base.log_prob(z, base_beta(beta))

    flow_model = RealNVP(
        dim=d, n_coupling=n_coupling, hidden=hidden, depth=depth,
        s_max=s_max, s_bump=s_bump, t_max=t_max, t_bump=t_bump,
    )
    optimizer = optax.chain(optax.clip_by_global_norm(grad_clip), optax.adam(lr))

    def loss_fn(params, beta_kp1, key_batch):
        z = base_sample(key_batch, _n_train, beta_kp1)
        x, logdet = flow_model.apply({"params": params}, z, method=flow_model.f)
        logp_x = logpi(x)
        return jnp.mean(-logdet - beta_kp1 * logp_x)

    def one_step(params, opt_state, beta_kp1, key_batch):
        loss, grad = jax.value_and_grad(loss_fn)(params, beta_kp1, key_batch)
        updates, opt_state2 = optimizer.update(grad, opt_state, params)
        params2 = optax.apply_updates(params, updates)
        return params2, opt_state2, loss

    def run_training(params, opt_state, beta_kp1, key, n_steps):
        def body(carry, key_batch):
            params_, opt_state_ = carry
            params_, opt_state_, loss = one_step(params_, opt_state_, beta_kp1, key_batch)
            return (params_, opt_state_), loss

        train_keys = jr.split(key, n_steps)
        (params, opt_state), losses = jax.lax.scan(body, (params, opt_state), train_keys)
        return params, opt_state, losses[-1]

    run_training_jit = jax.jit(run_training, static_argnums=4)

    key, key_init = jr.split(key)
    dummy = jnp.zeros((1, d), dtype=dtype)
    params = flow_model.init(key_init, dummy, method=flow_model.f)["params"]
    opt_state = optimizer.init(params)

    def flow_sample(params_, key_, m, beta):
        z = base_sample(key_, m, beta)
        x, _ = flow_model.apply({"params": params_}, z, method=flow_model.f)
        return x

    def flow_log_prob(params_, x, beta):
        z, ld_inv = flow_model.apply({"params": params_}, x, method=flow_model.finv)
        return base_log_prob(z, beta) + ld_inv

    snapshots: List[Array] = []
    ess_ratio_hist: List[float] = []
    info_hist: List[Dict[str, Any]] = []
    betas_used: List[float] = [0.0]

    beta_k = 0.0
    next_beta = float(beta0)
    deltas: List[float] = []
    stage = 0
    while beta_k < 1.0 - 1e-12 and stage < int(max_stages):
        key, key_train, key_snap, key_ess, key_kl = jr.split(key, 5)
        beta_kp1 = next_beta

        params, opt_state, last_loss = run_training_jit(params, opt_state, beta_kp1, key_train, int(train_steps))

        x_snap = flow_sample(params, key_snap, _n_snapshot, beta_kp1)
        snapshots.append(x_snap)

        x_ess = flow_sample(params, key_ess, n, beta_kp1)
        log_q = flow_log_prob(params, x_ess, beta_kp1)
        logw = beta_kp1 * logpi(x_ess) - log_q
        ess_val = float(_ess_from_logw(logw))
        ess_ratio = ess_val / float(n)
        ess_ratio_hist.append(ess_ratio)

        betas_used.append(beta_kp1)
        beta_k = beta_kp1
        stage += 1

        if beta_k < 1.0 - 1e-12:
            # Propose the next beta from the flow just trained at beta_k
            # (the reference implementation's `ref_cutoff=None` mode -- see
            # module docstring).
            x_kl = flow_sample(params, key_kl, _n_probe, beta_k)
            logq_kl = flow_log_prob(params, x_kl, beta_k)
            kl, kl_grad = _kl_grad_estimate(logpi, x_kl, logq_kl, beta_k)
            next_beta, delta = _temperflow_next_beta(
                math.log(beta_k), kl, kl_grad, kl_decay, deltas, is_first=(stage == 1)
            )
            deltas.append(delta)
        else:
            kl = float("nan")

        rec = {
            "k": int(stage - 1), "beta_k": betas_used[-2], "beta_kp1": beta_kp1,
            "ess_ratio": ess_ratio, "loss_last": float(last_loss), "kl_residual": kl,
        }
        info_hist.append(rec)
        if verbose:
            print(
                f"[temperflow] k={stage - 1:02d} | beta {betas_used[-2]:.4f}->{beta_kp1:.4f} | "
                f"ESS_ratio={ess_ratio:.3f} | loss={float(last_loss):.3e} | KL_residual={kl:.3e}"
            )

    final_beta = betas_used[-1]
    topup_steps = max(0, int(min_total_train_steps) - stage * int(train_steps))
    if topup_steps > 0:
        key, key_train, key_snap = jr.split(key, 3)
        params, opt_state, last_loss = run_training_jit(params, opt_state, final_beta, key_train, topup_steps)
        snapshots[-1] = flow_sample(params, key_snap, _n_snapshot, final_beta)
        if verbose:
            print(f"[temperflow] top-up: {topup_steps} extra steps at beta={final_beta:.4f} | "
                  f"loss={float(last_loss):.3e}")

    key, key_final = jr.split(key)
    samples = flow_sample(params, key_final, n, final_beta)
    final_params = params

    def q_log_prob(x: Array) -> Array:
        return flow_log_prob(final_params, jnp.asarray(x), final_beta)

    return TemperFlowResult(
        samples=samples,
        q_log_prob=q_log_prob,
        snapshots=snapshots,
        ess_ratio_history=ess_ratio_hist,
        info_history=info_hist,
        betas_used=betas_used,
        topup_steps=topup_steps,
    )
