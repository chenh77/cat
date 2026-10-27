# aft.py
"""AFT: Annealed Flow Transport Monte Carlo (Arbel, Matthews & Doucet, ICML 2021;
arXiv:2102.07501; reference code at `github.com/google-deepmind/annealed_flow_transport`).

Arbel et al.'s Algorithm 1 -- learned flow transport between successive annealed targets,
reweight, resample, MCMC rejuvenation -- with these components:

  - **Flow**: `src.flow.DiagonalAffineBounded`, an unconditional per-dimension affine map
    y = x exp(log_s) + t, as Arbel et al. use for their highest-dimensional example
    (Sec. 6.4). Its log-determinant is shared by every particle, so it cannot skew the
    self-normalized importance weights at small beta the way a coupling flow's
    per-particle log-determinant can in high dimension.
  - **Adaptive ladder**: ESS bisection on the particle system's own incremental
    importance weights (`_select_next_beta_smc`).
  - **Per-rung flow training**: a freshly initialized (identity) flow per rung, as in
    Algorithm 1, trained for `train_steps` Adam steps on the self-normalized
    importance-weighted transport free energy of the current particles, with a held-out
    `val_frac` split for early stopping.
  - **MCMC rejuvenation**: the MALA kernel of `mala.py`, vmapped over particles, after
    resampling.
  - **Step-size trust region** (an addition): the ESS bisection only checks that
    reweighting the current particles stays healthy at beta_next, not that the flow can
    be trained to realize the jump. `aft_fit` checks the realized post-training loss and
    ESS and, if unhealthy, retries with the step bisected toward beta_k.

**Annealing path**: the geometric bridge V_beta = (1 - beta) V_0 + beta V_target with
V_0(x) = |x|^2 / 2, i.e. log pi_beta(x) = -(1 - beta) |x|^2 / 2 + beta log pi(x), as in
Arbel et al. (Sec. 2). Since AFT carries a fixed particle set across stages, the base term
anchors pi_beta to the distribution the particles were drawn from at small beta.

**Mode-informed variant** (`modes=...`): V_0 becomes -log b(x), b the Laplace mixture of the
discovered modes (`_mode_base.LaplaceMixtureBase`), so pi_beta = b^(1-beta) pi^beta and
x_0 ~ b exactly. `modes=None` is the N(0, I) reference.

AFT's raw output is weighted particles; `aft_fit` resamples once at the end and returns
unweighted samples (no tractable density), with the pre-resample weighted particles and
the per-rung ESS history in `info` / `ess_ratio_history`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jr
import optax

from ..flow import DiagonalAffineBounded, RealNVPBounded
from ._mode_base import build_mode_base
from .mala import run_mala_steps

Array = jax.Array
LogPiBatch = Callable[[Array], Array]      # (N,d) -> (N,)
LogPiSingle = Callable[[Array], Array]     # (d,) -> scalar


def _ess_from_logw(logw: Array) -> Array:
    logw = logw - jnp.max(logw)
    w = jnp.exp(logw)
    w = w / jnp.sum(w)
    return 1.0 / jnp.sum(w * w)


def _select_next_beta_smc(
    logw_k: Array,
    combined_k: Array,
    beta_k: float,
    target_ess: float,
    tol: float = 1e-3,
    max_iters: int = 30,
    min_step: float = 1e-3,
    min_growth: float = 1.3,
) -> float:
    """Bisection search for the largest `beta_next` in `(beta_k, 1]` such
    that reweighting the *current particles* -- `logw_k` (this stage's
    incoming log-weights) plus the incremental term
    `(beta_next - beta_k) * combined_k` -- keeps the ESS ratio at or above
    `target_ess`. `combined_k = logpi(x_k) + 0.5*||x_k||^2` (precomputed
    once), the geometric-bridge increment of `log pi_beta(x_k)` w.r.t. beta
    under `V_beta = (1-beta)*V_0 + beta*V_target` (see `aft_fit`'s
    docstring for why `V_0 = 0.5*||x||^2`, not 0, is required). Same role
    as `temperflow.py`'s `_select_next_beta`, but against the particle
    system's own weights rather than a tractable proposal log-density (see
    module docstring), and with the same multiplicative-floor forward-
    progress guarantee for the same reason (an additive floor is
    catastrophically slow once beta must span several orders of magnitude).
    `aft_fit` calls this with a gentler `min_growth=1.3` than
    `temperflow.py`'s default of 2.0: TemperFlow retrains against fresh
    reparameterized noise every step, so an oversized forced jump just fits worse -- it can't collapse their
    sampling population. AFT carries a fixed, resampled SMC particle set
    across stages, so a forced jump large enough to fail badly enough to
    trigger a resample can multinomial-resample that population down to a
    handful of duplicated locations, which no later bisection retry (of
    that same, now underdiverse, population) can recover from regardless of
    step size. A smaller floor keeps most jumps within what the
    ESS-bisection oracle already vetted as safe.
    """
    n_ = logw_k.shape[0]

    def ess_ratio_at(beta_next: float) -> float:
        logw = logw_k + (beta_next - beta_k) * combined_k
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


def _simple_resample(key: Array, logw: Array, x: Array) -> Array:
    """Multinomial resampling on normalized weights `softmax(logw)`; returns
    resampled particles only -- caller resets log-weights to uniform."""
    n_ = logw.shape[0]
    idx = jr.categorical(key, logw, shape=(n_,))
    return jnp.take(x, idx, axis=0)


@dataclass
class AFTResult:
    samples: Array                     # (n, d) final, unweighted (resampled) particles
    q_log_prob: Any                    # always None: SMC output has no tractable density
    info: Dict[str, Any]               # pre-resample weighted particles + histories
    ess_ratio_history: List[float]     # post-reweight, pre-resample ESS ratio at each rung
    info_history: List[Dict[str, Any]] # per-rung {beta, train_loss, val_loss, ess_ratio, resampled}, aligned with betas_used[1:]
    betas_used: List[float]            # the discovered adaptive ladder, starts at 0.0, ends at 1.0


def _gaussian_ref_lp(y):
    return -0.5 * jnp.sum(y ** 2, axis=-1)


def _build_mala_rejuvenator(logpi_single, step_size: float, num_steps: int, ref_lp=_gaussian_ref_lp):
    """Reuse one batched MALA executable across temperatures in an AFT fit.

    Targets the full geometric-bridge density `pi_beta`, not `beta*logpi`
    alone (see module docstring) -- `run_mala_steps`'s `log_density_scale`
    is a single multiplicative scalar, which can express `beta*logpi` but
    not the `(1-beta)*0.5*||x||^2` base-density term, so that term is
    folded directly into the per-point log-density closure instead.
    `ref_lp` is the bridge's reference log-density (N(0, I) up to a
    constant by default; the Laplace mixture `b` under `modes=`).
    """
    @jax.jit
    def rejuvenate(x, key, beta):
        keys = jr.split(key, x.shape[0])

        def tempered_log_density(point):
            return (1.0 - beta) * ref_lp(point) + beta * logpi_single(point)

        def run_one(point, key_one):
            return run_mala_steps(
                tempered_log_density, point, step_size, num_steps, rng_key=key_one,
                log_density_scale=1.0,
            )

        x_out, _, _ = jax.vmap(run_one)(x, keys)
        return x_out

    return rejuvenate


def aft_fit(
    key: Array,
    logpi: LogPiBatch,
    logpi_single: Optional[LogPiSingle],
    d: int,
    *,
    n: int = 2048,
    beta0: float = 0.05,
    target_ess: float = 0.5,
    max_stages: int = 40,
    train_steps: int = 100,
    val_frac: float = 0.2,
    flow_family: str = "diagonal_affine",
    n_coupling: int = 4,
    hidden: int = 64,
    depth: int = 2,
    s_max: float = 3.0,
    s_bump: float = 1.0,
    t_max: float = float("inf"),
    t_bump: float = 1.0,
    lr: float = 2e-3,
    grad_clip: float = 5.0,
    resample_threshold: float = 0.5,
    mala_step_size: float = 0.1,
    mala_steps: int = 5,
    mala_steps_resample: Optional[int] = None,
    schedule_min_growth: float = 1.3,
    max_step_retries: int = 8,
    min_ess_ratio: float = 1e-6,
    modes: Optional[Sequence[Any]] = None,
    mode_base_inflate: float = 1.0,
    dtype=jnp.float64,
    verbose: bool = False,
) -> AFTResult:
    """Sequential (single-rung-at-a-time) annealed flow transport.

    Loop, starting from `x_0 ~ N(0, I_d)` with uniform log-weights and
    `beta_0 = 0`:
      1. pick a candidate `beta_{k+1}` via `_select_next_beta_smc` against
         the current particles' own incremental weights;
      2. split the current particles `val_frac`/`1-val_frac` into a held-out
         validation batch and a training batch (Arbel et al.'s own
         Algorithm 2/3, App. F: separate train/validation particle sets,
         "to avoid over-fitting of the NF to the N particles"), then train
         a freshly reinitialized `DiagonalAffineBounded` flow for
         `train_steps` Adam steps against the training batch's self-
         normalized-importance-weighted transport free energy `E_{x ~ x_k,
         w_k}[log pi_beta_k(x) - log pi_beta_{k+1}(T(x)) - logdet]`
         (`pi_beta` the geometric bridge -- see module docstring), tracking the
         validation batch's free energy after every step and keeping the
         best-validation checkpoint (Algorithm 3's own model-selection
         rule) rather than just the final iterate -- a flow that starts
         exploiting a few training particles' idiosyncrasies shows up as
         validation free energy getting *worse* on the disjoint held-out
         set, long before the on-batch loss would reveal it;
      3. push the actual particles (the *full* batch, not just the training
         split) through the best-validation flow and reweight by that same
         delta. If the realized post-training loss/ESS is non-finite or
         degenerate (the flow couldn't actually fit the candidate jump --
         see module docstring), reject the step and bisect the interval
         toward `beta_k`, retraining the *same* rung's flow init against
         the smaller candidate (up to `max_step_retries` times) -- a
         numerical trust region around step 1's oracle, not a change to
         the algorithm;
      4. resample if the post-reweight ESS ratio drops below
         `resample_threshold`;
      5. rejuvenate with a few vmapped MALA steps targeting
         `pi_{beta_{k+1}}` (this repo's `mala.py` kernel) -- `mala_steps`
         after an unresampled step, `mala_steps_resample` after a resample.
    Repeat until `beta = 1` or `max_stages` is hit; resample once more at
    the end to hand back plain unweighted samples.

    `modes` (a ModeExplore inventory) replaces the N(0, I) reference with
    the Laplace mixture `b` (covariances scaled by `mode_base_inflate^2`)
    throughout -- see module docstring.

    A resampled population is, by construction, a multinomial draw from
    however many particles actually survived reweighting -- at the
    `min_ess_ratio` floor this can be as few as a handful of duplicated
    locations out of `n`. `mala_steps` (sized for ordinary rejuvenation of an
    already-diverse population) is frequently too few MALA steps to
    re-diversify *that* particular collapse before the next stage's
    reweighting compounds it further -- once this happens the
    ladder can get stuck unable to accept *any* further step, however small
    (every bisection candidate reweights the same underdiverse cloud).
    `mala_steps_resample` (default `max(4 * mala_steps, 20)`) runs strictly
    more rejuvenation specifically after a resample to counter this, without
    changing the per-stage cost model's accounting of ordinary
    (non-resampled) rejuvenation.
    """
    if logpi_single is None:
        raise ValueError("aft_fit requires logpi_single (per-point log-density) for its MALA rejuvenation step.")
    if not 0.0 < beta0 < 1.0:
        raise ValueError("beta0 must lie in (0, 1).")
    if not 0.0 < target_ess <= 1.0:
        raise ValueError("target_ess must lie in (0, 1].")

    d = int(d)
    n = int(n)

    mode_base = build_mode_base(modes, mode_base_inflate, dtype)
    ref_lp = _gaussian_ref_lp if mode_base is None else mode_base.log_prob
    if mode_base is not None and mode_base.dim != d:
        raise ValueError(f"modes have dimension {mode_base.dim}, expected d={d}.")

    # `flow_family="diagonal_affine"` (default, see module docstring) is the
    # collapse-safe choice. `flow_family="coupling"` is an
    # alternative -- a `RealNVPBounded` conditional coupling
    # flow, same as every other flow baseline, but only safe against the
    # per-particle log-det collapse this module documents at length if
    # `s_max` is kept small (the per-layer, per-dimension log-scale bound
    # directly caps how much any single particle's log-det can diverge from
    # the batch); it is on the caller to pick a small enough `s_max`.
    if flow_family == "diagonal_affine":
        del n_coupling, hidden, depth
        flow_model = DiagonalAffineBounded(
            dim=d, s_max=s_max, s_bump=s_bump, t_max=t_max, t_bump=t_bump,
        )
    elif flow_family == "coupling":
        flow_model = RealNVPBounded(
            dim=d, n_coupling=n_coupling, hidden=hidden, depth=depth,
            s_max=s_max, s_bump=s_bump, t_max=t_max, t_bump=t_bump,
        )
    else:
        raise ValueError(f"Unknown flow_family {flow_family!r}; choose 'diagonal_affine' or 'coupling'.")
    # `apply_if_finite` skips an Adam update outright (instead of applying a
    # NaN/Inf update) whenever a step's gradient goes non-finite -- e.g. one
    # heavy-tailed particle at a huge raw magnitude driving the MLP through
    # overflow. Without this, one bad
    # step permanently corrupts Adam's moment estimates for the rest of that
    # rung's `train_steps` (each rung reinitializes fresh -- see module
    # docstring -- so this corruption does not propagate *across* rungs,
    # but it would still ruin the current one). This only ever skips a step
    # that would already be useless (its gradient carries no usable signal);
    # it never changes what a valid step does.
    optimizer = optax.apply_if_finite(
        optax.chain(optax.clip_by_global_norm(grad_clip), optax.adam(lr)),
        max_consecutive_errors=train_steps,
    )

    def transport(params, x):
        return flow_model.apply({"params": params}, x, method=flow_model.f)

    def log_pi_beta(y, beta):
        # `log pi_beta(y) = -(1-beta)*V_0(y) - beta*V_target(y)`, `V_0(y) =
        # 0.5*||y||^2` (or `-log b(y)` under `modes=`), `V_target(y) =
        # -logpi(y)` -- the geometric bridge anchoring `pi_beta` to the
        # actual `x_0` sampling density at every beta, not just beta=0 (see
        # module docstring).
        return (1.0 - beta) * ref_lp(y) + beta * logpi(y)

    def delta(params, x, beta_k, beta_kp1):
        tx, logdet = transport(params, x)
        return log_pi_beta(x, beta_k) - log_pi_beta(tx, beta_kp1) - logdet

    def free_energy(params, x, logw, beta_k, beta_kp1):
        d_ = delta(params, x, beta_k, beta_kp1)
        return jnp.sum(jax.nn.softmax(logw) * d_)

    def one_step(params, opt_state, x, logw, beta_k, beta_kp1):
        loss, grad = jax.value_and_grad(free_energy)(params, x, logw, beta_k, beta_kp1)
        updates, opt_state2 = optimizer.update(grad, opt_state, params)
        return optax.apply_updates(params, updates), opt_state2, loss

    def run_training(params, opt_state, x_tr, logw_tr, x_val, logw_val, beta_k, beta_kp1):
        # Arbel et al.'s own Algorithm 3: track every step's held-out
        # validation free energy and keep the *best*-validation checkpoint,
        # not just the last iterate -- see docstring.
        val0 = free_energy(params, x_val, logw_val, beta_k, beta_kp1)

        def body(carry, _):
            params_, opt_state_, best_params_, best_val_ = carry
            params_, opt_state_, train_loss = one_step(params_, opt_state_, x_tr, logw_tr, beta_k, beta_kp1)
            val_loss = free_energy(params_, x_val, logw_val, beta_k, beta_kp1)
            improved = val_loss < best_val_
            best_params_ = jax.tree_util.tree_map(
                lambda b, c: jnp.where(improved, c, b), best_params_, params_
            )
            best_val_ = jnp.where(improved, val_loss, best_val_)
            return (params_, opt_state_, best_params_, best_val_), (train_loss, val_loss)

        (params, opt_state, best_params, best_val), (train_losses, val_losses) = jax.lax.scan(
            body, (params, opt_state, params, val0), None, length=int(train_steps),
        )
        return best_params, opt_state, train_losses[-1], best_val

    run_training_jit = jax.jit(run_training)

    key, key_x0 = jr.split(key)
    dummy = jnp.zeros((1, d), dtype=dtype)

    if mode_base is None:
        x = jr.normal(key_x0, (n, d), dtype=dtype)
    else:
        x = mode_base.sample(key_x0, n)
    # Under the geometric bridge (see module docstring), pi_0 == N(0, I_d)
    # (or b) exactly -- matching how `x` was actually sampled -- so uniform
    # initial log-weights are the correct SNIS weight, not an approximation.
    logw = -jnp.log(jnp.asarray(n, dtype=dtype)) * jnp.ones(n, dtype=dtype)

    mala_rejuvenate = _build_mala_rejuvenator(
        logpi_single, mala_step_size, int(mala_steps), ref_lp
    )
    _mala_steps_resample = (
        max(4 * int(mala_steps), 20) if mala_steps_resample is None else int(mala_steps_resample)
    )
    mala_rejuvenate_resample = (
        mala_rejuvenate if _mala_steps_resample == int(mala_steps)
        else _build_mala_rejuvenator(logpi_single, mala_step_size, _mala_steps_resample, ref_lp)
    )

    n_val = max(1, min(n - 1, int(round(n * val_frac))))
    n_train = n - n_val

    ess_ratio_hist: List[float] = []
    info_history: List[Dict[str, Any]] = []
    betas_used: List[float] = [0.0]
    beta_k = 0.0
    stage = 0
    print_every = max(1, int(max_stages) // 10)

    while beta_k < 1.0 - 1e-12 and stage < int(max_stages):
        key, key_resample, key_mala = jr.split(key, 3)

        lpi_k = logpi(x)
        if beta_k == 0.0:
            # First rung: seed at the fixed `beta0` (Arbel et al. Algorithm 1
            # starts the ladder at a small non-zero temperature) rather than
            # letting the largest-safe-jump bisection run on the very first,
            # riskiest transition -- x is still a naive N(0, I) cloud here,
            # so the current-particles' ESS oracle has no way to know how
            # much the *flow* would have to stretch to reach a large beta.
            beta_kp1 = beta0
        else:
            # Geometric-bridge increment of log pi_beta(x) w.r.t. beta (see
            # `_select_next_beta_smc`'s docstring): d/dbeta[-(1-beta)*0.5*
            # ||x||^2 + beta*logpi(x)] = 0.5*||x||^2 + logpi(x)
            # (logpi(x) - log b(x) under `modes=`).
            combined_k = lpi_k - ref_lp(x)
            beta_kp1 = _select_next_beta_smc(
                logw, combined_k, beta_k, target_ess, min_growth=schedule_min_growth,
            )

        # Held-out validation split for this rung's flow fit (Arbel et al.'s
        # own Algorithm 2/3 -- see docstring); a fixed positional split is
        # fine since `x`'s particle order carries no meaning.
        x_tr, x_val = x[:n_train], x[n_train:]
        logw_tr, logw_val = logw[:n_train], logw[n_train:]

        # Trust region around the no-move ESS oracle above: only accept the
        # candidate beta_kp1 if the flow, once actually trained, produces a
        # finite loss and a non-degenerate ESS; otherwise bisect toward
        # beta_k and retry. Each retry (like each rung) starts from a fresh
        # flow init -- see module docstring's per-rung-reinitialization
        # discussion for why warm-starting off a diverged/degenerate step
        # (or off any prior rung at all) is exactly the failure mode this
        # trust region exists to catch.
        key, key_reinit = jr.split(key)
        params_init = flow_model.init(key_reinit, dummy, method=flow_model.f)["params"]
        opt_state_init = optimizer.init(params_init)
        for retry in range(int(max_step_retries) + 1):
            params_try, opt_state_try, train_loss, val_loss = run_training_jit(
                params_init, opt_state_init, x_tr, logw_tr, x_val, logw_val, beta_k, beta_kp1
            )
            tx, logdet = transport(params_try, x)
            d_kp1 = log_pi_beta(x, beta_k) - log_pi_beta(tx, beta_kp1) - logdet
            logw_new = jax.nn.log_softmax(logw - d_kp1)
            ess_ratio = float(_ess_from_logw(logw_new)) / n

            # A finite scalar loss/ESS can still hide a collapsed step -- if
            # `apply_if_finite` above spent the whole rung skipping updates,
            # the flow never actually moved and the resulting weights pile
            # onto a couple of particles (ess_ratio ~ 1-2/n): finite, but not
            # a fit worth accepting. `ess_floor` catches that degeneracy.
            ess_floor = max(min_ess_ratio, 2.0 / n)
            healthy = (
                bool(jnp.isfinite(val_loss))
                and bool(jnp.isfinite(ess_ratio))
                and bool(jnp.all(jnp.isfinite(tx)))
                and ess_ratio >= ess_floor
            )
            if healthy or retry == int(max_step_retries):
                break
            if verbose:
                print(f"[aft] stage={stage:03d} | beta_candidate={beta_kp1:.4e} rejected "
                      f"(val_loss={float(val_loss):.3e}, ess_ratio={ess_ratio:.3e}) -- bisecting")
            beta_kp1 = beta_k + 0.5 * (beta_kp1 - beta_k)

        if not healthy:
            raise RuntimeError(
                f"aft_fit: stage {stage} could not find a trainable beta step above beta_k={beta_k:.4e} "
                f"after {max_step_retries} bisections (last candidate={beta_kp1:.4e}, val_loss={float(val_loss):.3e}, "
                f"ess_ratio={ess_ratio:.3e}). The flow/target combination cannot fit even the smallest step tried."
            )
        ess_ratio_hist.append(ess_ratio)

        resampled = ess_ratio < resample_threshold
        if resampled:
            x_resampled = _simple_resample(key_resample, logw_new, tx)
            logw_new = -jnp.log(jnp.asarray(n, dtype=dtype)) * jnp.ones(n, dtype=dtype)
        else:
            x_resampled = tx

        rejuvenator = mala_rejuvenate_resample if resampled else mala_rejuvenate
        x = rejuvenator(x_resampled, key_mala, beta_kp1)
        logw = logw_new

        betas_used.append(beta_kp1)
        info_history.append({
            "beta": beta_kp1, "train_loss": float(train_loss), "val_loss": float(val_loss),
            "ess_ratio": ess_ratio, "resampled": resampled,
        })
        beta_k = beta_kp1
        stage += 1
        if verbose and (stage % print_every == 0 or beta_k >= 1.0 - 1e-12 or stage == int(max_stages)):
            print(f"[aft] stage={stage:03d} | beta={beta_k:.4e} | ess_ratio={ess_ratio:.3f} | val_loss={float(val_loss):.4e}")

    key, key_final = jr.split(key)
    resampled_samples = _simple_resample(key_final, logw, x)

    info = {
        "num_stages": stage,
        "final_beta": beta_k,
        "pre_resample_samples": x,
        "pre_resample_log_weights": logw,
        "mala_steps_resample": _mala_steps_resample,
        "n_resampled": sum(1 for h in info_history if h["resampled"]),
        "n_modes": 0 if mode_base is None else mode_base.n_modes,
    }

    return AFTResult(
        samples=resampled_samples, q_log_prob=None, info=info,
        ess_ratio_history=ess_ratio_hist, info_history=info_history, betas_used=betas_used,
    )
