# mode_explore.py
from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union

import numpy as np
from scipy.stats import norm
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from jaxopt import LBFGS

from .proposals import Kernel, KERNEL_REGISTRY
from ..utils.numeric import as1d_jax, sym_jax, quad_inv_jax, logdet_spd_jax

Array = jax.Array
LogPiSingle = Callable[[Array], Array]       # (d,) -> scalar


def _furthest_point_subsample(X: np.ndarray, m: int, rng: np.random.Generator) -> np.ndarray:
    """Greedy furthest-point subsampling. Returns m indices that are maximally
    spread in Euclidean space. O(n * m) — fast for typical n, m values."""
    n = X.shape[0]
    selected = np.empty(m, dtype=np.intp)
    selected[0] = rng.integers(n)
    min_dists = np.full(n, np.inf)
    for i in range(1, m):
        diff = X - X[selected[i - 1]]
        dists = np.einsum("ij,ij->i", diff, diff)   # squared Euclidean, no sqrt needed
        np.minimum(min_dists, dists, out=min_dists)
        selected[i] = np.argmax(min_dists)
    return selected
GradLogPiSingle = Callable[[Array], Array]   # (d,) -> (d,)
HessLogPiSingle = Callable[[Array], Array]   # (d,) -> (d,d)


@dataclass
class Mode:
    # `mu` remains the actual stationary point found by LBFGS. Mode exploration
    # also keeps `anchor_mu` at this point, including with `profile_max`.
    # The optional field supports explicitly supplied Gaussian bridge centers.
    mu: Array
    Sigma: Array
    w: float = 1.0
    anchor_mu: Optional[Array] = None


def _find_mode_lbfgs(
    x0: Array,
    *,
    logpi_single: LogPiSingle,
    grad_logpi_single: Optional[GradLogPiSingle] = None,
    maxiter: int,
    tol: float,
    history_size: int,
    use_grad_clip: bool = False,
    grad_clip: float = 10.0,
    _solver=None,          # pre-built LBFGS solver (avoids re-creating per call)
    _obj_and_grad=None,    # pre-built value_and_grad function (for grad_clip path)
) -> Array:
    """
    LBFGS ascent on log pi, implemented by minimizing -log pi.
    Pass _solver (and _obj_and_grad when use_grad_clip=True) to reuse the
    solver across calls and avoid repeated JIT recompilation.
    """
    x0 = as1d_jax(x0)

    if grad_logpi_single is None:
        grad_logpi_single = jax.grad(logpi_single)

    if _solver is None:
        def obj(x):
            return -logpi_single(x)

        if use_grad_clip:
            def obj_and_grad(x):
                val = -logpi_single(x)
                g = -grad_logpi_single(x)
                gnorm = jnp.linalg.norm(g)
                scale = jnp.minimum(1.0, grad_clip / (gnorm + 1e-12))
                g = g * scale
                return val, g

            _solver = LBFGS(
                fun=obj,
                value_and_grad=True,
                maxiter=int(maxiter),
                tol=float(tol),
                history_size=int(history_size),
                implicit_diff=False,
            )
            _obj_and_grad = obj_and_grad
        else:
            _solver = LBFGS(
                fun=obj,
                maxiter=int(maxiter),
                tol=float(tol),
                history_size=int(history_size),
                implicit_diff=False,
            )

    if use_grad_clip and _obj_and_grad is not None:
        res = _solver.run(x0, value_and_grad=_obj_and_grad)
    else:
        res = _solver.run(x0)

    return as1d_jax(res.params)


def _profile_halfwidth_vec(
    logpi_single: LogPiSingle,
    mu: Array,
    directions: Array,
    log_p_mu: Array,
    delta: float,
    t_inits: Array,
    t_grow: float = 1.6,
    max_expand: int = 30,
    max_bisect: int = 60,
) -> Tuple[Array, Array]:
    """Vectorized/on-device version of the old per-direction `_profile_halfwidth`
    search: for every row `v` of `directions` (B, d), finds the distance t>0
    along `mu + t*v` at which logpi_single has dropped by `delta` nats from
    its value at `mu`.

    Runs all B probes' expand+bisect search together as two `lax.fori_loop`s
    (fixed max_expand/max_bisect iterations), so the whole thing lowers to a
    couple of compiled XLA loops with no per-step host sync -- unlike a plain
    Python loop calling `float(logpi_single(...))` per step, which forces a
    device round-trip on every single-point evaluation. Returns
    `(t_result, valid)` where `valid[i]` is False exactly where the old
    scalar version would have returned None (density never dropped by
    `delta` within the search range, e.g. an unbounded flat ridge).
    """
    logpi_batched = jax.vmap(logpi_single)

    def f(t: Array) -> Array:
        pts = mu[None, :] + t[:, None] * directions
        return logpi_batched(pts) - (log_p_mu - delta)

    t_hi0 = jnp.maximum(t_inits, 1e-6)
    f_hi0 = f(t_hi0)

    def expand_body(_, state):
        t_hi, f_hi = state
        grow = f_hi > 0.0
        t_hi_next = jnp.where(grow, t_hi * t_grow, t_hi)
        f_hi_next = jnp.where(grow, f(t_hi_next), f_hi)
        return (t_hi_next, f_hi_next)

    t_hi, f_hi = jax.lax.fori_loop(0, max_expand, expand_body, (t_hi0, f_hi0))
    valid = f_hi <= 0.0

    t_lo0 = jnp.zeros_like(t_hi)

    def bisect_body(_, state):
        t_lo, t_hi = state
        t_mid = 0.5 * (t_lo + t_hi)
        go_right = f(t_mid) > 0.0
        t_lo_next = jnp.where(go_right, t_mid, t_lo)
        t_hi_next = jnp.where(go_right, t_hi, t_mid)
        return (t_lo_next, t_hi_next)

    t_lo, t_hi = jax.lax.fori_loop(0, max_bisect, bisect_body, (t_lo0, t_hi))
    return 0.5 * (t_lo + t_hi), valid


@partial(
    jax.jit,
    static_argnames=(
        "logpi_single", "hess_logpi_single", "ridge", "cov_method", "return_anchor_shift",
    ),
)
def _local_covariance(
    mu: Array,
    *,
    logpi_single: LogPiSingle,
    hess_logpi_single: Optional[HessLogPiSingle],
    ridge: float,
    eig_floor: float,
    eig_cap: float,
    cov_method: str = "hessian",
    profile_delta: float = 2.0,
    profile_t_mult: float = 3.0,
    cov_inflate: float = 1.0,
    return_anchor_shift: bool = False,
) -> Union[Array, Tuple[Array, Array]]:
    mu = as1d_jax(mu)
    d = mu.shape[0]

    if hess_logpi_single is not None:
        H = sym_jax(hess_logpi_single(mu))
    else:
        H = sym_jax(jax.hessian(logpi_single)(mu))

    A = sym_jax(-H)
    if ridge > 0.0:
        A = sym_jax(A + ridge * jnp.eye(d, dtype=A.dtype))

    evals, evecs = jnp.linalg.eigh(A)
    evals_clipped = jnp.clip(evals, eig_floor, eig_cap)

    if cov_method == "hessian":
        Sigma = (evecs * (1.0 / evals_clipped)) @ evecs.T
        Sigma = sym_jax(cov_inflate * Sigma)
        if return_anchor_shift:
            return Sigma, jnp.zeros_like(mu)
        return Sigma

    if cov_method not in ("profile", "profile_max"):
        raise ValueError(
            f"Unknown cov_method={cov_method!r}. Use 'hessian', 'profile', "
            "or 'profile_max'."
        )

    # Profile methods: point curvature at the mode can badly underestimate
    # the true width of a mode's basin along near-flat/near-singular
    # directions (e.g. a profile-likelihood ridge). Instead of trusting the
    # curvature alone, probe outward along each Hessian eigendirection and
    # measure the displacement at which log-density actually drops by
    # `profile_delta` nats, then back out an effective std.  `profile` uses
    # the average of the two half-widths, while
    # `profile_max` uses the wider half-width.  The latter is deliberately
    # coverage-oriented for skewed modes: it lets the covariance reach the
    # longer tail while keeping the Gaussian mean at the discovered mode;
    # the flow's marginal spline then learns the asymmetric redistribution.
    # Both variants only widen relative to the Hessian-implied std.
    log_p_mu = logpi_single(mu)

    var_hessian = 1.0 / evals_clipped
    std_hess = jnp.sqrt(var_hessian)
    t_init = jnp.maximum(profile_t_mult * std_hess, 1e-3)
    profile_norm = jnp.sqrt(2.0 * profile_delta)

    # v_k_all[k] is the k-th eigenvector (evecs' columns, transposed to rows)
    # so both probe directions for all k are handled by two batched
    # `_profile_halfwidth_vec` calls instead of a Python loop of 2*d
    # sequential, host-syncing searches.
    v_k_all = evecs.T
    t_plus, valid_plus = _profile_halfwidth_vec(
        logpi_single, mu, v_k_all, log_p_mu, profile_delta, t_init
    )
    t_minus, valid_minus = _profile_halfwidth_vec(
        logpi_single, mu, -v_k_all, log_p_mu, profile_delta, t_init
    )
    any_valid = valid_plus | valid_minus

    if cov_method == "profile_max":
        cand = jnp.stack(
            [jnp.where(valid_plus, t_plus, -jnp.inf), jnp.where(valid_minus, t_minus, -jnp.inf)],
            axis=-1,
        )
        t_profile = jnp.max(cand, axis=-1)
    else:  # "profile": average of whichever half-widths were found
        sum_t = jnp.where(valid_plus, t_plus, 0.0) + jnp.where(valid_minus, t_minus, 0.0)
        cnt = valid_plus.astype(t_plus.dtype) + valid_minus.astype(t_plus.dtype)
        t_profile = sum_t / jnp.maximum(cnt, 1.0)

    std_profile = t_profile / profile_norm
    var_final = jnp.where(any_valid, jnp.maximum(var_hessian, std_profile ** 2), var_hessian)

    # Only floor at 1/eig_cap (the same safety net the Hessian branch uses
    # against pathological near-zero variance) -- deliberately no upper clip
    # at 1/eig_floor here, since the whole point of "profile" is to let the
    # estimate exceed what that floor would otherwise cap it at.
    var_final = jnp.maximum(var_final, 1.0 / eig_cap)
    Sigma = (evecs * var_final[None, :]) @ evecs.T
    Sigma = sym_jax(cov_inflate * Sigma)
    if return_anchor_shift:
        # Preserve the return API while keeping the Gaussian mean at the mode.
        return Sigma, jnp.zeros_like(mu)
    return Sigma


def _resolve_kernel(mh: Union[str, Kernel], mh_kwargs: Optional[Mapping[str, Any]]) -> Kernel:
    mh_kwargs = {} if mh_kwargs is None else dict(mh_kwargs)
    if isinstance(mh, str):
        name = mh.lower()
        if name not in KERNEL_REGISTRY:
            raise ValueError(f"Unknown mh='{mh}'. Available: {list(KERNEL_REGISTRY.keys())}")
        return KERNEL_REGISTRY[name](**mh_kwargs)
    return mh


def _build_mh_step_batch(
    kernel: Kernel,
    logpi_single: LogPiSingle,
    beta_hot: float,
    grad_logpi_single: Optional[GradLogPiSingle],
):
    """Builds the jit(vmap(...)) MH step used to warm up all particles at once.
    Build this ONCE (e.g. in `multi_round_mode_explore`) and pass it into every
    `mode_explore` call via `_mh_step_batch` -- rebuilding it per call creates a
    fresh closure each time, which JAX's jit cache treats as a cache miss (keyed
    on function identity, not just shapes), forcing a full XLA recompile even
    though n/d never change across rounds.
    """
    def _mh_step_one(x, key):
        xn, acc, _ = kernel.step(
            x, logpi=logpi_single, beta=beta_hot, key=key, grad_logpi=grad_logpi_single
        )
        return xn, acc.astype(x.dtype)

    return jax.jit(jax.vmap(_mh_step_one))


def _build_lbfgs_solver(
    logpi_single: LogPiSingle,
    grad_logpi_single: Optional[GradLogPiSingle],
    *,
    optimize_steps: int,
    lbfgs_tol: float,
    lbfgs_history_size: int,
    lbfgs_use_grad_clip: bool,
    lbfgs_grad_clip: float,
):
    """Builds the (solver, obj_and_grad) pair `_find_mode_lbfgs` reuses across
    particles. Same cache-miss concern as `_build_mh_step_batch`: build once
    and reuse across rounds, not once per `mode_explore` call."""
    _neg_logpi = lambda x: -logpi_single(x)
    if bool(lbfgs_use_grad_clip):
        _neg_logpi_grad = (
            lambda x: -grad_logpi_single(x) if grad_logpi_single is not None else -jax.grad(logpi_single)(x)
        )
        _obj_and_grad = lambda x: (
            -logpi_single(x),
            _neg_logpi_grad(x) * jnp.minimum(1.0, float(lbfgs_grad_clip) / (jnp.linalg.norm(_neg_logpi_grad(x)) + 1e-12)),
        )
        solver = LBFGS(
            fun=_neg_logpi, value_and_grad=True,
            maxiter=int(optimize_steps), tol=float(lbfgs_tol),
            history_size=int(lbfgs_history_size), implicit_diff=False,
        )
        return solver, _obj_and_grad

    solver = LBFGS(
        fun=_neg_logpi,
        maxiter=int(optimize_steps), tol=float(lbfgs_tol),
        history_size=int(lbfgs_history_size), implicit_diff=False,
    )
    return solver, None


def _build_hess_fn(
    logpi_single: LogPiSingle, hess_logpi_single: Optional[HessLogPiSingle]
) -> HessLogPiSingle:
    return jax.jit(hess_logpi_single if hess_logpi_single is not None else jax.hessian(logpi_single))


def recompute_mode_weights(modes: List[Mode], logpi_single: LogPiSingle) -> None:
    if not modes:
        return

    log_unnorm = jnp.array(
        [logpi_single(as1d_jax(m.mu)) + 0.5 * logdet_spd_jax(m.Sigma) for m in modes]
    )
    a = jnp.max(log_unnorm)
    w = jnp.exp(log_unnorm - a)
    w = w / jnp.sum(w)

    # print("\n[recompute_mode_weights]")
    # print("log_unnorm:", np.asarray(log_unnorm))
    # print("max log_unnorm:", float(a))
    # print("shifted:", np.asarray(log_unnorm - a))
    # print("raw exp shifted:", np.asarray(jnp.exp(log_unnorm - a)))
    # print("normalized weights:", np.asarray(w))
    # print("min weight:", float(jnp.min(w)))
    # print("max weight:", float(jnp.max(w)))
    # print("sum weights:", float(jnp.sum(w)))

    for m, wk in zip(modes, list(w)):
        m.w = float(wk)


def mode_explore(
    particles: Array,
    logpi_single: LogPiSingle,
    beta_hot: float,
    modes: List[Mode],
    v: int,
    mh: Union[str, Kernel] = "rwmh",
    mh_kwargs: Optional[Mapping[str, Any]] = None,
    *,
    grad_logpi_single: Optional[GradLogPiSingle] = None,
    hess_logpi_single: Optional[HessLogPiSingle] = None,
    alpha: float = 0.05,
    distance_tol: Optional[float] = None,
    max_new_modes: Optional[int] = None,
    key: Optional[Array] = None,
    optimize_steps: int = 200,
    lbfgs_tol: float = 1e-5,
    lbfgs_history_size: int = 10,
    lbfgs_use_grad_clip: bool = False,
    lbfgs_grad_clip: float = 10.0,
    cov_ridge: float = 1e-6,
    eig_floor: float = 1e-7,
    eig_cap: float = 1e7,
    cov_method: str = "hessian",
    profile_delta: float = 2.0,
    profile_t_mult: float = 3.0,
    cov_inflate: float = 1.0,
    restart_prob: float = 0.1,
    restart_scale: float = 25.0,
    restart_scale_max: Optional[float] = None,
    max_lbfgs_particles: Optional[int] = None,
    weight_tol: float = 1e-6,
    _mh_step_batch=None,      # pre-built jit(vmap(...)) MH step (avoids re-compiling per call)
    _lbfgs_solver=None,       # pre-built LBFGS solver (avoids re-compiling per call)
    _lbfgs_obj_and_grad=None,  # pre-built value_and_grad fn, paired with _lbfgs_solver
    _hess_fn=None,            # pre-built jit(hessian) fn (avoids re-compiling per call)
) -> Tuple[List[Mode], Dict[str, Any]]:
    X = jnp.asarray(particles)
    if X.ndim != 2:
        raise ValueError(f"particles must have shape (n,d), got {X.shape}")
    n, d = X.shape

    if key is None:
        key = jax.random.PRNGKey(0)

    if distance_tol is None:
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"alpha must be in (0, 1), got {alpha!r}.")
        distance_tol = float(norm.ppf(1.0 - alpha / 2.0))

    kernel = _resolve_kernel(mh, mh_kwargs)

    if grad_logpi_single is None and hess_logpi_single is None:
        grad_logpi_single = jax.grad(logpi_single)

    # ------------------------------------------------------------------
    # Phase 1 (batched): warm up all particles with MH using one JIT-compiled
    # step applied to all particles at once (vmap over particles, Python loop
    # over v steps).  This avoids the slow lax.scan+vmap compilation that
    # occurs when the body contains a JIT-compiled logpi.
    # ------------------------------------------------------------------

    if _mh_step_batch is None:
        _mh_step_batch = _build_mh_step_batch(kernel, logpi_single, beta_hot, grad_logpi_single)

    X_warmed = X
    n_acc = jnp.zeros(n, dtype=X.dtype)

    for _ in range(int(v)):
        key, k1, k2 = jax.random.split(key, 3)

        if restart_prob > 0.0:
            kr, kn, ks = jax.random.split(k1, 3)
            r = jax.random.uniform(kr, shape=(n,), dtype=X.dtype)
            if restart_scale_max is not None and restart_scale_max > restart_scale:
                scales = jax.random.uniform(
                    ks, shape=(n, 1), dtype=X.dtype,
                    minval=float(restart_scale), maxval=float(restart_scale_max),
                )
            else:
                scales = jnp.full((n, 1), float(restart_scale), dtype=X.dtype)
            x_restarts = scales * jax.random.normal(kn, shape=(n, d), dtype=X.dtype)
            do_restart = r < jnp.asarray(restart_prob, dtype=X.dtype)
            X_warmed = jnp.where(do_restart[:, None], x_restarts, X_warmed)

        step_keys = jax.random.split(k2, n)
        X_mh, acc = _mh_step_batch(X_warmed, step_keys)
        if restart_prob > 0.0:
            # Restarted particles skip the MH update
            X_warmed = jnp.where(do_restart[:, None], X_warmed, X_mh)
            n_acc = n_acc + jnp.where(do_restart, jnp.zeros(n, dtype=acc.dtype), acc)
        else:
            X_warmed = X_mh
            n_acc = n_acc + acc

    acc_rates = (n_acc / max(1, int(v))).tolist()

    # ------------------------------------------------------------------
    # Phase 2 (sequential): for each warmed particle, find local mode and
    # covariance, then deduplicate and add new modes.
    #
    # Optionally subsample particles for LBFGS to reduce cost when n is large.
    # The MH warming above runs on all n particles; only the mode-finding
    # step needs a representative subset.
    # ------------------------------------------------------------------
    if max_lbfgs_particles is not None and n > int(max_lbfgs_particles):
        key, k_sub = jax.random.split(key)
        seed_np = int(jax.random.randint(k_sub, (), 0, 2 ** 31))
        rng_np = np.random.default_rng(seed_np)
        idx = _furthest_point_subsample(np.asarray(X_warmed), int(max_lbfgs_particles), rng_np)
        X_lbfgs = X_warmed[idx]
    else:
        X_lbfgs = X_warmed
    n_lbfgs = X_lbfgs.shape[0]

    # Pre-build LBFGS solver and hessian function (or reuse ones passed in from
    # `multi_round_mode_explore`) so that jaxopt and JAX can reuse the same
    # JIT-compiled computation across all particles -- and across rounds.
    # ------------------------------------------------------------------

    if _lbfgs_solver is None:
        _lbfgs_solver, _lbfgs_obj_and_grad = _build_lbfgs_solver(
            logpi_single, grad_logpi_single,
            optimize_steps=optimize_steps, lbfgs_tol=lbfgs_tol,
            lbfgs_history_size=lbfgs_history_size,
            lbfgs_use_grad_clip=lbfgs_use_grad_clip, lbfgs_grad_clip=lbfgs_grad_clip,
        )

    # `jax.hessian` alone returns a plain Python function that retraces from
    # scratch on every call; wrapping it in `jax.jit` here (reused across all
    # n_lbfgs particles below, and across rounds when passed in) compiles it
    # once instead of once per call.
    if _hess_fn is None:
        _hess_fn = _build_hess_fn(logpi_single, hess_logpi_single)

    n_old = len(modes)   # modes present before this call — never filtered out
    new_added = 0
    n_skipped_nonfinite = 0

    for i in range(n_lbfgs):
        x = as1d_jax(X_lbfgs[i])

        mu_star = _find_mode_lbfgs(
            x,
            logpi_single=logpi_single,
            grad_logpi_single=grad_logpi_single,
            maxiter=int(optimize_steps),
            tol=float(lbfgs_tol),
            history_size=int(lbfgs_history_size),
            use_grad_clip=bool(lbfgs_use_grad_clip),
            grad_clip=float(lbfgs_grad_clip),
            _solver=_lbfgs_solver,
            _obj_and_grad=_lbfgs_obj_and_grad,
        )

        # A particle whose LBFGS ascent starts (or transiently lands) in a
        # region of extreme curvature -- e.g. an unconstrained rate parameter
        # driven so large its likelihood contribution swamps float64 -- can
        # come back with a non-finite mu_star (jaxopt's internal line search
        # overflowing to nan) even though logpi_single itself never returns
        # nan for any *finite* input. Recording such a candidate as a mode
        # would poison every downstream computation that touches it:
        # `quad_inv_jax` distance checks against it go nan (silently
        # blocking future candidates from ever being accepted, since
        # `nan > distance_tol` is False), and `recompute_mode_weights`'s
        # `jnp.max`/normalization propagate that one nan to every mode's
        # weight, so `recompute_mode_weights` en masse -- see the
        # `RuntimeError` below, which is this failure mode's downstream
        # symptom. Skip such particles outright instead.
        if not bool(jnp.all(jnp.isfinite(mu_star))) or not bool(jnp.isfinite(logpi_single(mu_star))):
            n_skipped_nonfinite += 1
            continue

        Sigma_star, anchor_shift_star = _local_covariance(
            mu_star,
            logpi_single=logpi_single,
            hess_logpi_single=_hess_fn,   # pre-built, same object every call
            ridge=float(cov_ridge),
            eig_floor=float(eig_floor),
            eig_cap=float(eig_cap),
            cov_method=cov_method,
            profile_delta=float(profile_delta),
            profile_t_mult=float(profile_t_mult),
            cov_inflate=float(cov_inflate),
            return_anchor_shift=True,
        )

        # Same reasoning as the mu_star check above, one step later: even at
        # a mu_star with finite logpi, the Hessian (or profile probe) driving
        # Sigma_star can come back with non-finite entries in a region of
        # extreme curvature (autodiff through a near-overflowing exp() can
        # blow up second derivatives well before the function value itself
        # does), and `recompute_mode_weights`'s `logdet_spd_jax(Sigma)` term
        # propagates that nan just as readily as a bad mu_star would.
        anchor_mu_star = mu_star + anchor_shift_star
        if (
            not bool(jnp.all(jnp.isfinite(Sigma_star)))
            or not bool(jnp.all(jnp.isfinite(anchor_mu_star)))
        ):
            n_skipped_nonfinite += 1
            continue

        if not modes:
            min_dk = float("inf")
        else:
            dks = []
            for m in modes:
                # d_j = max{Mahalanobis under
                # Sigma_j, Mahalanobis under Sigma*}^{1/2}.
                delta = as1d_jax(m.mu) - mu_star
                dk = float(
                    jnp.sqrt(
                        jnp.maximum(
                            quad_inv_jax(jnp.asarray(m.Sigma), delta),
                            quad_inv_jax(Sigma_star, delta),
                        )
                    )
                )
                dks.append(dk)
            min_dk = min(dks)

        if min_dk > float(distance_tol):
            modes.append(
                Mode(
                    mu=jnp.asarray(mu_star),
                    Sigma=jnp.asarray(Sigma_star),
                    w=1.0,
                    anchor_mu=jnp.asarray(anchor_mu_star),
                )
            )
            new_added += 1
            if max_new_modes is not None and new_added >= int(max_new_modes):
                break

    if not modes:
        raise RuntimeError(
            f"mode_explore found no usable modes: all {n_lbfgs} LBFGS candidates "
            f"({n_skipped_nonfinite} of them non-finite) were skipped or rejected."
        )

    recompute_mode_weights(modes, logpi_single)

    # Drop modes whose relative weight is negligible (covers both float underflow
    # and genuine low-mass modes that contribute less than weight_tol of the
    # dominant mode, e.g. spurious local maxima in SUR profile posteriors).
    max_w = max(m.w for m in modes)
    n_before = len(modes)
    survive = [m.w > weight_tol * max_w for m in modes]
    surviving_old_idx = [j for j in range(n_old) if survive[j]]
    modes[:] = [m for m, keep in zip(modes, survive) if keep]
    if len(modes) == 0:
        raise RuntimeError("All modes have zero weight after recompute_mode_weights; "
                           "check that logpi_single returns finite values near modes.")
    if len(modes) < n_before:
        total = sum(m.w for m in modes)
        for m in modes:
            m.w = m.w / total

    info = {
        "n_particles": int(n),
        "n_lbfgs_particles": int(n_lbfgs),
        "dim": int(d),
        "beta_hot": float(beta_hot),
        "v": int(v),
        "alpha": float(alpha),
        "distance_tol": float(distance_tol),
        "kernel": type(kernel).__name__,
        "acc_rate_mean": float(sum(acc_rates) / len(acc_rates)) if acc_rates else 0.0,
        "n_modes_after": int(len(modes)),
        "n_new_modes_added": int(new_added),
        "n_skipped_nonfinite": int(n_skipped_nonfinite),
        "surviving_old_idx": surviving_old_idx,
        "mode_weights": [m.w for m in modes],
        "mode_mus": [m.mu.tolist() if hasattr(m.mu, "tolist") else list(m.mu) for m in modes],
        "anchor_mus": [
            np.asarray(
                m.anchor_mu if getattr(m, "anchor_mu", None) is not None else m.mu
            ).tolist()
            for m in modes
        ],
    }
    return modes, info


def filter_negligible_modes(modes: List[Mode], weight_tol: float = 1e-6) -> List[Mode]:
    """Drop modes whose relative weight is negligible -- the same rule
    `mode_explore` applies internally after its own `recompute_mode_weights`
    call (`weight <= weight_tol * max_weight`) -- and renormalize the
    survivors' weights to sum to 1. Useful after externally reweighting a
    mode list (e.g. after merging or canonicalizing modes and calling
    `recompute_mode_weights`), where merging/canonicalizing can leave behind
    near-massless duplicate basins that would otherwise still consume
    flow-training/MCMC budget for no benefit. Returns `modes` unchanged (not
    emptied) if every mode would be filtered out."""
    if not modes:
        return []
    max_w = max(m.w for m in modes)
    kept = [m for m in modes if m.w > float(weight_tol) * max_w]
    if not kept:
        return list(modes)
    total = sum(m.w for m in kept)
    for m in kept:
        m.w = m.w / total
    return kept


def multi_round_mode_explore(
    key: Array,
    particles_fn: Callable[[Array, int], Array],
    logpi_single: LogPiSingle,
    beta_hot: float,
    n: int,
    v: int,
    n_rounds: int,
    mh: Union[str, Kernel] = "rwmh",
    mh_kwargs: Optional[Mapping[str, Any]] = None,
    modes: Optional[List[Mode]] = None,
    grad_logpi_single: Optional[GradLogPiSingle] = None,
    hess_logpi_single: Optional[HessLogPiSingle] = None,
    verbose: bool = False,
    **mode_explore_kwargs: Any,
) -> Tuple[List[Mode], List[Dict[str, Any]]]:
    """Repeated diffuse-restart `mode_explore` (mirrors `alps.py`'s
    `discover_modes_alps`, generalized to any particle-sampling function
    instead of a fixed `search_scale * randn`).

    A single `mode_explore` call's MH warm-up trajectory, run from one batch
    of particles, tends to gravitate toward whichever basin(s) are easiest
    to reach from that batch's random start -- for a target with several
    well-separated basins (e.g. a GMM posterior's genuinely distinct
    data-clustering configurations, on top of any discrete label-permutation
    symmetry), one round routinely misses most of them even when its own
    `max_new_modes` cap isn't binding. Each round here draws a *fresh*,
    independent particle batch via `particles_fn(key, n)` and calls
    `mode_explore` seeded with the modes accumulated so far, so later rounds
    both search for genuinely new basins and re-confirm/dedup existing ones
    (via `mode_explore`'s own Mahalanobis grouping rule) rather than
    reinventing them.

    Returns the accumulated mode list plus one info dict per round (each
    `mode_explore`'s own info dict, with `"round"` and
    `"n_modes_before_round"` added).

    Target evaluations are JIT-wrapped once per invocation and reused across
    rounds, including direct density checks and mode-weight evaluations.
    Callers do not need to JIT-wrap their target functions themselves.
    """
    # The compiled MH/optimizer/Hessian paths below do not cover the direct
    # logpi calls in candidate validation and recompute_mode_weights. Cache
    # those too: targets that construct a lax.scan body otherwise recreate
    # that body on every eager evaluation. Wrap outside the round loop so
    # every phase uses the same callable.
    logpi_single = jax.jit(logpi_single)
    if grad_logpi_single is not None:
        grad_logpi_single = jax.jit(grad_logpi_single)

    # Build the JIT-compiled MH step, LBFGS solver, and hessian function ONCE
    # here and reuse them across every round below. `mode_explore` otherwise
    # rebuilds all three from fresh closures on each call; since JAX's (and
    # jaxopt's) compilation cache is keyed on function/solver identity rather
    # than just input shapes, a fresh object every round means a full XLA
    # recompile every round even though n/d/beta_hot never change across
    # rounds. Defaults below mirror `mode_explore`'s own LBFGS defaults.
    kernel = _resolve_kernel(mh, mh_kwargs)
    _mh_step_batch = _build_mh_step_batch(kernel, logpi_single, beta_hot, grad_logpi_single)
    _lbfgs_solver, _lbfgs_obj_and_grad = _build_lbfgs_solver(
        logpi_single, grad_logpi_single,
        optimize_steps=mode_explore_kwargs.get("optimize_steps", 200),
        lbfgs_tol=mode_explore_kwargs.get("lbfgs_tol", 1e-5),
        lbfgs_history_size=mode_explore_kwargs.get("lbfgs_history_size", 10),
        lbfgs_use_grad_clip=mode_explore_kwargs.get("lbfgs_use_grad_clip", False),
        lbfgs_grad_clip=mode_explore_kwargs.get("lbfgs_grad_clip", 10.0),
    )
    _hess_fn = _build_hess_fn(logpi_single, hess_logpi_single)

    modes = [] if modes is None else list(modes)
    info_per_round: List[Dict[str, Any]] = []
    for r in range(int(n_rounds)):
        key, k_particles, k_me = jax.random.split(key, 3)
        particles = particles_fn(k_particles, int(n))
        n_before = len(modes)
        modes, info = mode_explore(
            particles=particles,
            logpi_single=logpi_single,
            grad_logpi_single=grad_logpi_single,
            hess_logpi_single=hess_logpi_single,
            beta_hot=beta_hot,
            modes=modes,
            v=v,
            mh=kernel,
            mh_kwargs=mh_kwargs,
            key=k_me,
            _mh_step_batch=_mh_step_batch,
            _lbfgs_solver=_lbfgs_solver,
            _lbfgs_obj_and_grad=_lbfgs_obj_and_grad,
            _hess_fn=_hess_fn,
            **mode_explore_kwargs,
        )
        info["round"] = int(r)
        info["n_modes_before_round"] = int(n_before)
        info_per_round.append(info)
        if verbose:
            print(f"[multi_round_mode_explore] round {r:02d} | modes {n_before} -> {len(modes)} "
                  f"(+{len(modes) - n_before})")
    return modes, info_per_round
