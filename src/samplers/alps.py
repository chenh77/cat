# alps.py
"""ALPS: the Annealed Leap-Point Sampler (Tawn, Moores, Queniat & Roberts,
"Annealed Leap-Point Sampler for Multimodal Target Distributions",
arXiv:2112.12908), implemented after the paper's Sections 3-4 and Algorithm 1 (Appendix C).

Given a mode collection {(w_j, mu_j, Sigma_j)} (the exploration component's
output; here the shared ModeExplore inventory), ALPS's parallel-annealing
component runs one Markov chain on the product of n+1 levels
beta_0 = 1 < beta_1 < ... < beta_n = beta_max -- *colder* than the target --
each targeting the Hessian-adjusted tempered (HAT) density (Def. 3.1)

    pi_beta(x) ∝ pi(x)^beta pi(mu_{A(x,beta)})^(1-beta),
    A(x,beta) = argmax_j w_j N(x; mu_j, Sigma_j / beta)                  (Eq. 3)

which (approximately) preserves every mode's weight across levels (Appendix
B.1). One iteration (Algorithm 1):
  1. levels 0..n-1: pre-conditioned random-walk Metropolis (Sec 3.4), the
     increment covariance h_k^2 Sigma_{A(x,beta_k)} with h_k tuned to the
     0.234 acceptance rate; since A depends on x the proposal is asymmetric
     and the MH ratio includes the forward/reverse proposal densities;
  2. level n: the mode-leaping independence sampler (Sec 3.5) with proposal
     q_beta_max(x) = sum_j w_j N(x; mu_j, Sigma_j / beta_max)          (Eq. 6, 7);
  3. s QuanTA-aided swaps between a random adjacent pair (Sec 3.3, Eq. 5):
     each state is rescaled about its own mode by (beta_i/beta_{i+1})^(±1/2),
     rejected if that changes its mode assignment, else accepted with the
     HAT ratio (the map's Jacobians cancel).

Tuning, where the paper leaves a choice to the practitioner:
  - beta_max: the paper picks it so the cold-level leap acceptance reaches a
    target a (Sec 3.5 suggests a = 0.5; Theorem 5.1 gives beta_max = O(d)).
    Here it is the smallest power of 2 (up to `beta_max_cap`) whose
    estimated leap acceptance is >= `target_leap_acc`, estimated before the
    run from iid draws y, y' ~ q_beta with w = pi_beta / q_beta:
        a(beta) = E_{x ~ pi_beta, y ~ q_beta} min(1, w(y)/w(x))
                = E_{y, y' ~ q_beta}[min(w(y), w(y'))] / E_{y ~ q_beta}[w(y)].
    The draws are charged to the oracle count.
  - the ladder: geometric from 1 to beta_max with ratio <= `level_ratio`
    (Sec 3.2: "start with a geometrically-spaced schedule ... then
    fine-tuned"); during burn-in the interior levels are re-spaced in
    log beta to equalize the swap rejection rates (Syed et al. 2021), and
    the RWM scales h_k are tuned to the 0.234 acceptance rate (Sec 3.4).
    Both are frozen after burn-in.
  - `n_chains` independent ALPS chains run in parallel (adaptation pools
    their statistics); the output is `n_samples` beta = 1 states, thinned
    evenly from the post-burn-in trajectories (as `apt.py` thins its one
    long chain).
  - Every chain starts with level k drawn from q_{beta_k} (the mode
    information, as the other baselines' initialization).
Not implemented: the online exploration component (the modes are given)
and the truncated HAT targets of Appendix B.2 (an ad-hoc fix the paper used
for its SUR example).

The output samples are correlated MCMC draws targeting pi, with no tractable density
(as `apt.py`'s).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable, Dict, List, Mapping, Optional

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jr
from jax.scipy.special import logsumexp

from ..mode_explore.explore import (
    Mode, mode_explore, _resolve_kernel, _build_mh_step_batch,
    _build_lbfgs_solver, _build_hess_fn,
)
from ..mode_explore.density import modes_to_arrays
from ..utils.numeric import chol_jitter_jax, sym_jax

Array = jax.Array
LogPiBatch = Callable[[Array], Array]        # (N,d) -> (N,)
LogPiSingle = Callable[[Array], Array]       # (d,) -> scalar
GradLogPiSingle = Callable[[Array], Array]   # (d,) -> (d,)


def discover_modes_alps(
    key: Array,
    logpi_single: LogPiSingle,
    d: int,
    *,
    grad_logpi_single: Optional[GradLogPiSingle] = None,
    n_search: int = 2000,
    n_rounds: int = 3,
    search_scale: float = 3.0,
    mode_explore_kwargs: Optional[Mapping[str, Any]] = None,
    dtype=jnp.float64,
) -> List[Mode]:
    """Mode-search phase: repeated diffuse-restart calls to `mode_explore`
    (Algorithm 4), accumulating the mode inventory across rounds over several
    diffuse restarts, since ALPS is meant to work without an oracle target sampler.
    """
    kwargs = dict(
        beta_hot=0.5, v=20, mh="mala", mh_kwargs={"step_size": 0.3},
        optimize_steps=200, restart_prob=0.05, weight_tol=1e-6,
    )
    if mode_explore_kwargs:
        kwargs.update(mode_explore_kwargs)

    # Build the JIT-compiled MH step, LBFGS solver, and hessian function ONCE
    # and reuse them across every round below -- `mode_explore` otherwise
    # rebuilds all three from fresh closures on each call, and since JAX's
    # (and jaxopt's) compilation cache is keyed on function/solver identity
    # rather than just input shapes, that forces a full XLA recompile every
    # round even though n_search/d/beta_hot never change across rounds.
    _kernel = _resolve_kernel(kwargs["mh"], kwargs.get("mh_kwargs"))
    _mh_step_batch = _build_mh_step_batch(_kernel, logpi_single, kwargs["beta_hot"], grad_logpi_single)
    _lbfgs_solver, _lbfgs_obj_and_grad = _build_lbfgs_solver(
        logpi_single, grad_logpi_single,
        optimize_steps=kwargs.get("optimize_steps", 200),
        lbfgs_tol=kwargs.get("lbfgs_tol", 1e-5),
        lbfgs_history_size=kwargs.get("lbfgs_history_size", 10),
        lbfgs_use_grad_clip=kwargs.get("lbfgs_use_grad_clip", False),
        lbfgs_grad_clip=kwargs.get("lbfgs_grad_clip", 10.0),
    )
    _hess_fn = _build_hess_fn(logpi_single, None)
    kwargs["mh"] = _kernel

    modes: List[Mode] = []
    for r in range(int(n_rounds)):
        key, k_x0, k_me = jr.split(key, 3)
        x0 = float(search_scale) * jr.normal(k_x0, (int(n_search), int(d)), dtype=dtype)
        modes, _info = mode_explore(
            particles=x0,
            logpi_single=logpi_single,
            grad_logpi_single=grad_logpi_single,
            modes=modes,
            key=k_me,
            _mh_step_batch=_mh_step_batch,
            _lbfgs_solver=_lbfgs_solver,
            _lbfgs_obj_and_grad=_lbfgs_obj_and_grad,
            _hess_fn=_hess_fn,
            **kwargs,
        )
    return modes


@dataclass
class ALPSResult:
    samples: Array                  # (n_samples, d) beta = 1 states, thinned post-burn-in
    modes: List[Mode]
    leap_acc_rate: float            # cold-level independence sampler, post-burn-in
    local_acc_rate: float           # RWM acceptance averaged over levels 0..n-1, post-burn-in
    betas: np.ndarray               # (n+1,) final ladder, betas[0] = 1
    n_steps: int
    n_target_evals: int             # log pi evaluations, including the beta_max pilot
    n_target_grad_evals: int = 0    # gradient calls of the Hessians (d per Hessian)
    info: Dict[str, Any] = field(default_factory=dict)


def alps_eval_count(*, n_chains: int, n_steps: int, n_levels: int, n_swaps: int,
                    n_modes: int, n_pilot_evals: int) -> int:
    """log pi evaluations of `alps_sample`: the beta_max pilot, log pi at the
    modes, the initial states, and per chain and step one RWM proposal per
    warm level, one leap proposal and two per QuanTA swap."""
    per_step = (int(n_levels) - 1) + 1 + 2 * int(n_swaps)
    return (int(n_pilot_evals) + int(n_modes) + int(n_chains) * int(n_levels)
            + int(n_chains) * int(n_steps) * per_step)


def alps_sample(
    key: Array,
    logpi_batch: LogPiBatch,
    logpi_single: LogPiSingle,
    d: int,
    *,
    modes: Optional[List[Mode]] = None,
    grad_logpi_single: Optional[GradLogPiSingle] = None,
    n_samples: int = 2048,
    n_chains: int = 256,
    n_steps: Optional[int] = None,
    value_budget: Optional[int] = None,
    target_leap_acc: float = 0.5,
    beta_max: Optional[float] = None,
    beta_max_cap: float = 4096.0,
    n_pilot: int = 1024,
    level_ratio: float = 4.0,
    n_swaps: Optional[int] = None,
    burnin_frac: float = 0.5,
    adapt_every: int = 50,
    target_local_acc: float = 0.234,
    laplace_cov: str = "hessian",
    eig_floor: float = 1e-7,
    grad_weight: Optional[float] = None,
    mode_search_kwargs: Optional[Mapping[str, Any]] = None,
    dtype=jnp.float64,
    verbose: bool = False,
) -> ALPSResult:
    """Run ALPS (module docstring) on the mode collection `modes`.

    The run length is `n_steps`, or with `value_budget` the largest step count
    whose total cost fits it: log pi evaluations (`alps_eval_count`, pilot
    included) plus `grad_weight` (default d) per gradient call of the
    Hessians below.
    `beta_max` fixes the coldest level instead of the leap-acceptance pilot.
    `laplace_cov="hessian"` (default) rebuilds the mode collection the paper's
    way (Sec 3.6) at the given mode locations: S_j = -[Hess log pi(mu_j)]^{-1}
    (eigenvalues floored at `eig_floor`) and w_j ∝ pi(mu_j) |S_j|^{1/2}; each
    Hessian is charged as d gradient calls (`n_target_grad_evals`).
    `"given"` uses the inventory's covariances and weights as they are.
    `modes=None` runs `discover_modes_alps` first (not charged here).
    """
    if modes is None:
        key, k_search = jr.split(key)
        modes = discover_modes_alps(
            k_search, logpi_single, d, grad_logpi_single=grad_logpi_single, dtype=dtype,
            **(mode_search_kwargs or {}),
        )
    if len(modes) == 0:
        raise RuntimeError("ALPS needs at least one mode.")
    if (n_steps is None) == (value_budget is None):
        raise ValueError("give exactly one of n_steps and value_budget")

    d = int(d)
    C = int(n_chains)
    mu, Sigma, logw = modes_to_arrays(modes, dtype=dtype)
    m = int(mu.shape[0])
    lp_mu = jnp.asarray(logpi_batch(mu), dtype=dtype)                   # (m,) log pi at the modes
    n_hessians = 0
    if laplace_cov == "hessian":
        # Sec 3.6: S_j = -[Hess log pi(mu_j)]^{-1}, w_j ∝ pi(mu_j) |S_j|^{1/2}.
        hess = jax.vmap(jax.hessian(logpi_single))(mu)                  # (m, d, d)
        evals, evecs = jnp.linalg.eigh(sym_jax(-hess))
        evals = jnp.maximum(evals, float(eig_floor))
        Sigma = jnp.einsum("mij,mj,mkj->mik", evecs, 1.0 / evals, evecs)
        logw = lp_mu - 0.5 * jnp.sum(jnp.log(evals), axis=-1)
        n_hessians = m
    elif laplace_cov != "given":
        raise ValueError(f"laplace_cov must be 'hessian' or 'given', got {laplace_cov!r}")
    log_w = logw - logsumexp(logw)
    L = chol_jitter_jax(sym_jax(Sigma))                                  # (m, d, d)
    half_logdet = jnp.sum(jnp.log(jnp.diagonal(L, axis1=-2, axis2=-1)), axis=-1)

    # ---- mode geometry (no (n, d, d) gathers: d reaches 10^3 in this repo) ----
    # Every state carries its whitened coordinates U[:, j] = L_j^{-1} (x - mu_j) (n, m, d),
    # computed by one GEMM against the precomputed L_j^{-1} instead of triangular solves:
    # at d ~ 500 this geometry, not log pi, dominates the run time.
    Linv = jax.vmap(lambda L_j: jax.scipy.linalg.solve_triangular(
        L_j, jnp.eye(d, dtype=dtype), lower=True))(L)                   # (m, d, d)
    Linv_mu = jnp.einsum("mij,mj->mi", Linv, mu)                         # (m, d)

    def whiten(x):
        """(n, m, d): L_j^{-1} (x - mu_j) for every mode j."""
        return jnp.einsum("nj,mij->nmi", x, Linv) - Linv_mu

    def comp_logp_u(U, beta):
        """(n, m) from whitened coordinates: log w_j + log N(x; mu_j, Sigma_j / beta)
        up to the shared -d/2 log 2pi."""
        maha = jnp.sum(U * U, axis=-1)                                   # (n, m)
        beta = jnp.reshape(jnp.asarray(beta, dtype=dtype), (-1, 1))
        return log_w - half_logdet + 0.5 * d * jnp.log(beta) - 0.5 * beta * maha

    def comp_logp(x, beta):
        return comp_logp_u(whiten(x), beta)

    def assign_u(U, beta):
        return jnp.argmax(comp_logp_u(U, beta), axis=-1)

    def apply_L(idx, z):
        """L[idx[i]] @ z[i] for every row i."""
        return jnp.einsum("mij,nj->mni", L, z)[idx, jnp.arange(z.shape[0])]

    def pick(U, idx):
        """U[i, idx[i]] for every row i: (n, d)."""
        return U[jnp.arange(U.shape[0]), idx]

    def hat(lp, A, beta):
        """log HAT density (Def. 3.1) up to a level-independent constant."""
        return beta * lp + (1.0 - beta) * lp_mu[A]

    def sample_q(key_, n, beta):
        """n draws from q_beta = sum_j w_j N(mu_j, Sigma_j / beta) (Eq. 6)."""
        k_c, k_e = jr.split(key_)
        comps = jr.categorical(k_c, log_w, shape=(n,))
        eps = jr.normal(k_e, (n, d), dtype=dtype)
        return mu[comps] + apply_L(comps, eps) / jnp.sqrt(beta)

    # ---- beta_max: smallest power of 2 with estimated leap acceptance >= target ----
    pilot = []
    n_pilot_evals = 0
    if beta_max is None:
        @jax.jit
        def leap_acc_estimate(key_, beta):
            y = sample_q(key_, int(n_pilot), beta)
            c = comp_logp(y, beta)
            lw = hat(logpi_batch(y), jnp.argmax(c, axis=-1), beta) - logsumexp(c, axis=-1)
            lw = jnp.where(jnp.isfinite(lw), lw, -jnp.inf)
            h = int(n_pilot) // 2
            est = (logsumexp(jnp.minimum(lw[:h], lw[h:2 * h])) - jnp.log(h)
                   - (logsumexp(lw) - jnp.log(lw.shape[0])))
            return jnp.exp(est)

        beta_c = 1.0
        while True:
            key, k_p = jr.split(key)
            a = float(leap_acc_estimate(k_p, jnp.asarray(beta_c, dtype=dtype)))
            a = a if math.isfinite(a) else 0.0
            n_pilot_evals += int(n_pilot)
            pilot.append((beta_c, a))
            if a >= target_leap_acc or 2.0 * beta_c > beta_max_cap:
                break
            beta_c *= 2.0
        reached = [b for b, a in pilot if a >= target_leap_acc]
        beta_max = reached[0] if reached else max(pilot, key=lambda p: p[1])[0]
    beta_max = float(beta_max)

    # ---- ladder ----
    K = 1 if beta_max <= 1.0 else int(math.ceil(math.log(beta_max) / math.log(level_ratio) - 1e-9)) + 1
    betas = np.geomspace(1.0, beta_max, K) if K > 1 else np.ones(1)
    s = (K - 1) if n_swaps is None else (int(n_swaps) if K > 1 else 0)

    if value_budget is not None:
        fixed = alps_eval_count(n_chains=C, n_steps=0, n_levels=K, n_swaps=s, n_modes=m,
                                n_pilot_evals=n_pilot_evals)
        per_step = C * ((K - 1) + 1 + 2 * s)
        gw = float(d if grad_weight is None else grad_weight)
        n_steps = int((int(value_budget) - fixed - gw * n_hessians * d) // per_step)
    n_steps = int(n_steps)
    n_keep = int(math.ceil(int(n_samples) / C))
    burnin = int(burnin_frac * n_steps)
    thin = (n_steps - burnin) // n_keep
    if thin < 1:
        raise ValueError(f"ALPS: {n_steps} steps (K={K} levels, {C} chains) leave no room for "
                         f"{n_keep} kept states per chain after {burnin} burn-in steps")

    # ---- one iteration of Algorithm 1 for all chains ----
    def step(carry, key_):
        X, LP, U, betas_, log_h = carry
        k_loc, k_leap, k_swap = jr.split(key_, 3)
        stats = {}

        if K > 1:   # 1. pre-conditioned RWM at levels 0..K-2 (Sec 3.4)
            ka, kb = jr.split(k_loc)
            xs = X[:, :K - 1].reshape(-1, d)
            us = U[:, :K - 1].reshape(-1, m, d)
            lps = LP[:, :K - 1].reshape(-1)
            bs = jnp.tile(betas_[:K - 1], C)
            hs = jnp.tile(jnp.exp(log_h), C)
            A = assign_u(us, bs)
            z = jr.normal(ka, xs.shape, dtype=dtype)
            prop = xs + hs[:, None] * apply_L(A, z)
            lpp = logpi_batch(prop)
            up = whiten(prop)
            Ap = assign_u(up, bs)
            v = (pick(us, Ap) - pick(up, Ap)) / hs[:, None]             # L_Ap^{-1} (xs - prop) / h
            log_fwd = -half_logdet[A] - 0.5 * jnp.sum(z * z, axis=-1)
            log_rev = -half_logdet[Ap] - 0.5 * jnp.sum(v * v, axis=-1)
            loga = hat(lpp, Ap, bs) - hat(lps, A, bs) + log_rev - log_fwd
            acc = jnp.log(jr.uniform(kb, lps.shape, dtype=dtype)) < loga
            xs = jnp.where(acc[:, None], prop, xs)
            us = jnp.where(acc[:, None, None], up, us)
            lps = jnp.where(acc, lpp, lps)
            X = X.at[:, :K - 1].set(xs.reshape(C, K - 1, d))
            U = U.at[:, :K - 1].set(us.reshape(C, K - 1, m, d))
            LP = LP.at[:, :K - 1].set(lps.reshape(C, K - 1))
            stats["local_acc"] = jnp.mean(acc.reshape(C, K - 1).astype(dtype), axis=0)

        # 2. mode-leaping independence sampler at the coldest level (Sec 3.5)
        ka, kb = jr.split(k_leap)
        bmax = betas_[K - 1]
        x, ux, lpx = X[:, K - 1], U[:, K - 1], LP[:, K - 1]
        y = sample_q(ka, C, bmax)
        uy = whiten(y)
        lpy = logpi_batch(y)
        cx, cy = comp_logp_u(ux, bmax), comp_logp_u(uy, bmax)
        loga = (hat(lpy, jnp.argmax(cy, axis=-1), bmax) - hat(lpx, jnp.argmax(cx, axis=-1), bmax)
                + logsumexp(cx, axis=-1) - logsumexp(cy, axis=-1))
        acc = jnp.log(jr.uniform(kb, (C,), dtype=dtype)) < loga
        X = X.at[:, K - 1].set(jnp.where(acc[:, None], y, x))
        U = U.at[:, K - 1].set(jnp.where(acc[:, None, None], uy, ux))
        LP = LP.at[:, K - 1].set(jnp.where(acc, lpy, lpx))
        stats["leap_acc"] = jnp.mean(acc.astype(dtype))

        # 3. s QuanTA-aided swaps of a random adjacent pair per chain (Sec 3.3)
        swap_acc = jnp.zeros(max(K - 1, 1), dtype=dtype)
        swap_att = jnp.zeros(max(K - 1, 1), dtype=dtype)
        rows = jnp.arange(C)
        for k_s in jr.split(k_swap, max(s, 1))[:s]:
            ka, kb = jr.split(k_s)
            i = jr.randint(ka, (C,), 0, K - 1)
            xi, xj = X[rows, i], X[rows, i + 1]
            ui, uj = U[rows, i], U[rows, i + 1]
            lpi, lpj = LP[rows, i], LP[rows, i + 1]
            bi, bj = betas_[i], betas_[i + 1]
            Ai, Aj = assign_u(ui, bi), assign_u(uj, bj)
            yj = jnp.sqrt(bi / bj)[:, None] * (xi - mu[Ai]) + mu[Ai]       # T+ (Eq. 5a)
            yi = jnp.sqrt(bj / bi)[:, None] * (xj - mu[Aj]) + mu[Aj]       # T- (Eq. 5b)
            u_y = whiten(jnp.concatenate([yi, yj]))
            u_yi, u_yj = u_y[:C], u_y[C:]
            ok = (assign_u(u_yj, bj) == Ai) & (assign_u(u_yi, bi) == Aj)
            lp_y = logpi_batch(jnp.concatenate([yi, yj]))
            lp_yi, lp_yj = lp_y[:C], lp_y[C:]
            loga = (hat(lp_yi, Aj, bi) + hat(lp_yj, Ai, bj) - hat(lpi, Ai, bi) - hat(lpj, Aj, bj))
            acc = ok & (jnp.log(jr.uniform(kb, (C,), dtype=dtype)) < loga)
            X = X.at[rows, i].set(jnp.where(acc[:, None], yi, xi))
            X = X.at[rows, i + 1].set(jnp.where(acc[:, None], yj, xj))
            U = U.at[rows, i].set(jnp.where(acc[:, None, None], u_yi, ui))
            U = U.at[rows, i + 1].set(jnp.where(acc[:, None, None], u_yj, uj))
            LP = LP.at[rows, i].set(jnp.where(acc, lp_yi, lpi))
            LP = LP.at[rows, i + 1].set(jnp.where(acc, lp_yj, lpj))
            onehot = jax.nn.one_hot(i, K - 1, dtype=dtype)
            swap_acc = swap_acc + jnp.sum(onehot * acc[:, None], axis=0)
            swap_att = swap_att + jnp.sum(onehot, axis=0)
        stats["swap_acc"], stats["swap_att"] = swap_acc, swap_att
        return (X, LP, U, betas_, log_h), stats

    @partial(jax.jit, static_argnames=("n",))
    def run_chunk(key_, X, LP, U, betas_, log_h, n):
        (X, LP, U, _, _), st = jax.lax.scan(step, (X, LP, U, betas_, log_h), jr.split(key_, n))
        return X, LP, U, jax.tree_util.tree_map(lambda a: jnp.sum(a, axis=0), st)

    # ---- initial states: level k from q_{beta_k} ----
    key, k_init = jr.split(key)
    X = jnp.stack([sample_q(k, C, b) for k, b in zip(jr.split(k_init, K), betas)], axis=1)  # (C, K, d)
    LP = logpi_batch(X.reshape(-1, d)).reshape(C, K)
    U = whiten(X.reshape(-1, d)).reshape(C, K, m, d)
    betas_j = jnp.asarray(betas, dtype=dtype)
    log_h = jnp.log(2.38 / math.sqrt(d) / jnp.sqrt(betas_j[:max(K - 1, 1)]))

    # ---- burn-in with adaptation: RWM scales -> 0.234, swap rejections equalized ----
    done = 0
    while done < burnin:
        n = min(int(adapt_every), burnin - done)
        key, k_c = jr.split(key)
        X, LP, U, st = run_chunk(k_c, X, LP, U, betas_j, log_h, n)
        done += n
        if K > 1:
            log_h = log_h + (st["local_acc"] / n - target_local_acc)
        if K > 2:
            rej = 1.0 - np.asarray(st["swap_acc"]) / np.maximum(np.asarray(st["swap_att"]), 1.0)
            u = np.log(np.asarray(betas_j))
            lam = np.concatenate([[0.0], np.cumsum(np.maximum(rej, 1e-3))])
            u_new = np.interp(np.linspace(0.0, lam[-1], K), lam, u)
            u_new[0], u_new[-1] = 0.0, u[-1]
            betas_j = jnp.asarray(np.exp(0.5 * (u + u_new)), dtype=dtype)

    # ---- sampling: frozen tuning, keep level-0 states every `thin` steps ----
    extra = (n_steps - burnin) - n_keep * thin
    if extra:
        key, k_c = jr.split(key)
        X, LP, U, _ = run_chunk(k_c, X, LP, U, betas_j, log_h, extra)
    kept, totals = [], None
    for _ in range(n_keep):
        key, k_c = jr.split(key)
        X, LP, U, st = run_chunk(k_c, X, LP, U, betas_j, log_h, thin)
        kept.append(X[:, 0])
        totals = st if totals is None else jax.tree_util.tree_map(jnp.add, totals, st)
    samples = jnp.stack(kept).reshape(-1, d)[: int(n_samples)]

    n_post = n_keep * thin
    leap_acc = float(totals["leap_acc"]) / n_post
    local_by_level = (np.asarray(totals["local_acc"]) / n_post).tolist() if K > 1 else []
    swap_by_pair = (np.asarray(totals["swap_acc"]) / np.maximum(np.asarray(totals["swap_att"]), 1.0)).tolist() \
        if K > 1 else []
    n_evals = alps_eval_count(n_chains=C, n_steps=n_steps, n_levels=K, n_swaps=s, n_modes=m,
                              n_pilot_evals=n_pilot_evals)
    info = {
        "n_modes": m, "n_chains": C, "n_steps": n_steps, "burnin": burnin, "thin": thin,
        "n_levels": K, "n_swaps": s, "betas": np.asarray(betas_j).tolist(),
        "beta_max_pilot": pilot, "step_sizes": np.exp(np.asarray(log_h)).tolist() if K > 1 else [],
        "leap_acc_rate": leap_acc, "local_acc_by_level": local_by_level,
        "swap_acc_by_pair": swap_by_pair, "n_target_evals": n_evals,
        "laplace_cov": laplace_cov, "n_hessians": n_hessians,
        "mode_weights": np.exp(np.asarray(log_w)).tolist(),
    }
    if verbose:
        print(f"[alps] modes={m} chains={C} levels={K} beta_max={beta_max:g} steps={n_steps} "
              f"(burn-in {burnin}, thin {thin}) | leap_acc={leap_acc:.3f} | "
              f"local_acc={np.round(local_by_level, 3).tolist()} | swap_acc={np.round(swap_by_pair, 3).tolist()} | "
              f"pilot={[(b, round(a, 3)) for b, a in pilot]}")
    return ALPSResult(
        samples=samples, modes=list(modes), leap_acc_rate=leap_acc,
        local_acc_rate=float(np.mean(local_by_level)) if local_by_level else float("nan"),
        betas=np.asarray(betas_j), n_steps=n_steps, n_target_evals=n_evals,
        n_target_grad_evals=n_hessians * d, info=info,
    )
