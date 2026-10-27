"""experiments/transport_stage.py

CAT, CNF, GAT and GMM stage shared by the `run_<target>.py` benchmark runs.

CAT is the budget standard. It runs first, on the discovered modes, at a fixed
per-target configuration (`CAT_CONFIG`); its realized *transport* cost (oracle calls
minus the shared mode discovery) is the budget every other method is matched to:

  - the baselines in the run scripts (`budget.matched_baseline_config`);
  - the CNF (`CNF_CONFIG`, on a uniform grid with as many points as CAT's final grid,
    `batch_size` solved for the budget);
  - GAT (CAT's configuration with its J fields folded into one -- J x CAT's per-mode
    parameters and batch size -- on the Laplace mixture of all discovered modes,
    `n_train` solved for the budget).

Costs are weighted as value + d * value_grad (the only Hessian calls are in mode
discovery). Each method's oracle cost returned here includes the shared mode
discovery; its wall time is the transport time only -- the caller adds the discovery
time.
"""
from __future__ import annotations

import time
from typing import Any, Dict, Optional

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr

import common  # noqa: F401  (puts the project root on sys.path)
from src.mode_explore.explore import Mode
from src.samplers._mode_base import build_mode_base
from src.samplers.cat import _ess_betas, _gauss_logpdf, cat
from src.samplers.cnf import cnf, cnf_cost
from src.samplers.gat import field_param_count, gat_fit
from src.utils import budget as ob

VC_LABELS = ("CAT", "CNF", "GAT")

# Evaluation hooks for `run_dynamics.py` (empty otherwise): per-label callbacks used when
# none is passed explicitly, and the last entry (result object included) of every run.
CALLBACKS: Dict[str, Any] = {}
LAST_ENTRIES: Dict[str, Any] = {}

# ---------------- configurations ----------------
# Vector-field parametrization shared by CAT, GAT and the CNF, so they differ only in
# how the field is trained.
NET = dict(hidden=64, depth=3, activation="silu", spline_knots=8, rk4_substeps=2)
# CAT: continuity residual + velocity-norm penalty, learned log-normalizer rate,
# ESS-adaptive grid (target ESS / delta_max from `ADAPTIVE_GRID`).
CAT_BASE = dict(n_refresh=20, train_steps_per_refresh=250, batch_size=512, n_validation=256,
                validate_every=5, select_best=True, lr=1e-3, grad_clip=5.0, velocity_l2_weight=1e-3,
                replay_refreshes=20, particle_chunk=2048, center_h=True, **NET)
# CNF: the same field trained by reverse KL to the endpoint, on a uniform grid with as
# many points as CAT's final grid (same solver cost); `batch_size` spends the budget.
CNF_BASE = dict(n_refresh=20, train_steps_per_refresh=250, n_validation=256, validate_every=5,
                select_best=True, lr=1e-3, grad_clip=5.0, **NET)

# Per target: CAT's settings (`n_train` fixed: CAT runs as given and sets the budget;
# `final_calibration=None` estimates the endpoint weights from all n_snapshot particles,
# otherwise from `n_calibrate` fresh draws per mode; `min_mode_weight` drops discovered
# modes lighter than that before transport), and the CNF's field width.
CAT_CONFIG: Dict[str, Dict[str, Any]] = {
    "skewnormal": dict(CAT_BASE, n_refresh=10, train_steps_per_refresh=100, batch_size=256,
                       final_calibration=None, n_train=1024),
    "sur": dict(CAT_BASE, train_steps_per_refresh=100, min_mode_weight=0.05, n_train=768),
    "iohmm": dict(CAT_BASE, hidden=32, n_train=2560),
    "lda": dict(CAT_BASE, n_train=6144),
}
CNF_CONFIG: Dict[str, Dict[str, Any]] = {
    "skewnormal": dict(CNF_BASE), "sur": dict(CNF_BASE), "iohmm": dict(CNF_BASE, hidden=32),
    "lda": dict(CNF_BASE),
}
# (adaptive_target_ess, adaptive_delta_max) of the ESS-adaptive grids of CAT and GAT
ADAPTIVE_GRID = {"skewnormal": (0.7, 0.3), "sur": (0.3, 0.3), "iohmm": (0.3, 0.3), "lda": (0.7, 0.3)}


# ---------------- budget helpers ----------------
def weighted(d: int, cost: ob.OracleCost) -> int:
    """value + d * value_grad (no Hessian calls outside mode discovery)."""
    return cost.weighted(d, grad_weight=d, hessian_grad_units=0)


def prune_modes(modes, min_weight):
    """Drop discovered modes with weight < `min_weight` (renormalizing the rest):
    near-empty spurious modes get poorly fitted maps whose endpoint weights
    log C_j are then badly overestimated."""
    if not min_weight:
        return modes
    keep = [m for m in modes if m.w >= min_weight] or [max(modes, key=lambda m: m.w)]
    tot = sum(m.w for m in keep)
    return [Mode(mu=m.mu, Sigma=m.Sigma, w=m.w / tot, anchor_mu=m.anchor_mu) for m in keep]


def cat_modes(target: str, modes):
    """The modes CAT (and the GMM baseline) uses for `target`."""
    return prune_modes(list(modes), CAT_CONFIG[target].get("min_mode_weight", 0.0))


def _cat_pilot_grid(logpi, modes, seed: int, target_ess: float, delta_max: float, n_pilot: int = 1024):
    """Grid of CAT's first (identity-flow) refresh: draws x_0 ~ b_j, h at those draws,
    `_ess_betas` at the target ESS. Charged as J * n_pilot values."""
    mu = jnp.stack([jnp.asarray(m.anchor_mu if m.anchor_mu is not None else m.mu) for m in modes])
    chol = jnp.linalg.cholesky(jnp.stack([jnp.asarray(m.Sigma) for m in modes]))
    loga = jnp.log(jnp.asarray([float(m.w) for m in modes]))
    J = len(modes)
    eps = jr.normal(jr.PRNGKey(seed + 31), (J, n_pilot, mu.shape[1]))
    x0 = mu[:, None, :] + jnp.einsum("jab,jnb->jna", chol, eps)
    lb = jnp.stack([_gauss_logpdf(x0.reshape(-1, mu.shape[1]), mu[k], chol[k]) for k in range(J)], -1)
    h = (logpi(x0.reshape(-1, mu.shape[1])) - jax.scipy.special.logsumexp(lb + loga, -1)).reshape(J, n_pilot)
    grid = _ess_betas(np.zeros(1), np.zeros((1, J, n_pilot)), np.asarray(h)[None], target_ess,
                      delta_max, 200, 2)
    return grid, ob.OracleCost(value=J * n_pilot)


def _gat_pilot_grid(logpi, modes, seed: int, target_ess: float, delta_max: float, n_pilot: int = 1024):
    """Grid of GAT's first (identity-flow) refresh: x_0 ~ b (the Laplace mixture),
    h = log pi - log b, `_ess_betas` at the target ESS. Charged as n_pilot values."""
    base = build_mode_base(modes, 1.0, jnp.float64)
    x0 = base.sample(jr.PRNGKey(seed + 31), n_pilot)
    h = logpi(x0) - base.log_prob(x0)
    grid = _ess_betas(np.zeros(1), np.zeros((1, 1, n_pilot)), np.asarray(h)[None, None],
                      target_ess, delta_max, 200, 2)
    return grid, ob.OracleCost(value=n_pilot)


def _cnf_cost(kw, n_modes):
    v, g = cnf_cost(n=kw["n"], batch_size=kw["batch_size"], n_validation=kw["n_validation"],
                    n_modes=n_modes, n_refresh=kw["n_refresh"],
                    train_steps_per_refresh=kw["train_steps_per_refresh"],
                    validate_every=kw["validate_every"], final_calibration=kw.get("final_calibration"))
    return ob.OracleCost(value=v, value_grad=g)


def _gat_cost(kw, n_betas):
    """Nominal GAT cost: CAT's with one field (J = 1) and all n final draws evaluated."""
    return ob.cat_cost(
        n=kw["n"], n_train=kw["n_train"], n_validation=kw["n_validation"], n_modes=1,
        n_betas=n_betas, n_refresh=kw["n_refresh"], validate_every=kw.get("validate_every", 1),
        final_calibration=None, adaptive_grid=True)


def resolve_cat(target, *, logpi, modes, seed, n, n_calibrate):
    """(cat kwargs, pilot cost): `CAT_CONFIG[target]` with the adaptive-grid settings; the
    pilot's J * 1024 evaluations are added to CAT's cost."""
    kw = dict(CAT_CONFIG[target])
    kw.setdefault("final_calibration", n_calibrate)
    kw.pop("min_mode_weight", None)       # applied by cat_modes
    ess, dmax = ADAPTIVE_GRID[target]
    kw.update(adaptive_target_ess=ess, adaptive_delta_max=dmax)
    _, pilot = _cat_pilot_grid(logpi, modes, seed, ess, dmax)
    return dict(n=n, **kw), pilot


def resolve_cnf(target, *, n, budget, n_modes, d, n_cat_betas):
    """(betas, cnf kwargs): a uniform grid with CAT's number of points and the largest
    `batch_size` whose nominal cost fits `budget`."""
    betas = np.linspace(0.0, 1.0, n_cat_betas)
    kw = dict(n=n, **CNF_CONFIG[target])
    lo = 0
    while weighted(d, _cnf_cost(dict(kw, batch_size=lo + 1), n_modes)) <= budget:
        lo += 1
    if lo == 0:
        raise ValueError(f"budget {budget:.3e} too small for batch_size=1")
    kw["batch_size"] = lo
    return betas, kw


def resolve_gat(target, *, logpi, modes, n_cat_modes, d, seed, n, budget):
    """(gat_fit kwargs, pilot cost) for GAT at CAT's configuration: the same training
    schedule, optimizer, loss and grid rule, with CAT's J = `n_cat_modes` fields folded into
    one -- J times CAT's per-mode parameter count (`target_params`) and J times its batch
    size -- and `n_train` solved for `budget`. GAT transports from the Laplace mixture of
    all `modes`, as the baselines' `modes=` option does."""
    spec = dict(CAT_CONFIG[target])
    for k in ("min_mode_weight", "final_calibration"):
        spec.pop(k, None)
    j = int(n_cat_modes)
    spec["target_params"] = j * field_param_count(
        d, spec.get("hidden", 64), depth=spec.get("depth", 3), spline_knots=spec.get("spline_knots", 0))
    spec["batch_size"] = j * spec["batch_size"]
    ess, dmax = ADAPTIVE_GRID[target]
    spec.update(adaptive_target_ess=ess, adaptive_delta_max=dmax)
    grid, pilot = _gat_pilot_grid(logpi, modes, seed, ess, dmax)
    budget = budget - weighted(d, pilot)
    kw = dict(n=n, **spec)
    lo, hi = 0, 8
    fits = lambda v: weighted(d, _gat_cost(dict(kw, n_train=v), len(grid))) <= budget
    while fits(hi):
        lo, hi = hi, 2 * hi
    while hi - lo > 8:
        mid = max((lo + hi) // 2 // 8 * 8, lo + 8)
        lo, hi = (mid, hi) if fits(mid) else (lo, mid)
    if lo == 0:
        raise ValueError(f"budget {budget:.3e} too small for GAT n_train=8")
    kw["n_train"] = lo
    return kw, pilot


# ---------------- stage ----------------
def _guarded(label, fn, log_prefix, mode_discovery_cost):
    t0 = time.time()
    try:
        res, cost, info = fn()
    except Exception as e:
        print(f"{log_prefix} [{label}] FAILED after {time.time() - t0:.1f}s: {e!r}")
        return None
    wall = time.time() - t0
    return dict(res=res, wall_time_s=wall, cost=mode_discovery_cost + cost, transport_cost=cost, info=info)


def run_cat(
    target: str, *, seed: int, dtype, d: int, logpi, logpi_single, grad_logpi_single,
    modes, mode_discovery_cost: ob.OracleCost, n_snapshot: int, n_calibrate: int, log_prefix: str,
    callback=None,
) -> Optional[Dict[str, Any]]:
    """Run CAT at `CAT_CONFIG[target]` on `modes`.

    Returns None on failure, else dict(res, wall_time_s, cost, transport_cost, info):
    `wall_time_s` is the transport time, `cost` includes `mode_discovery_cost`
    and `transport_cost` (the budget standard) does not.
    """
    callback = callback or CALLBACKS.get("CAT")

    def _run():
        modes_j = cat_modes(target, modes)
        kw, pilot_cost = resolve_cat(target, logpi=logpi, modes=modes_j, seed=seed, n=n_snapshot,
                                     n_calibrate=n_calibrate)
        print(f"{log_prefix} [CAT] {len(modes_j)}/{len(modes)} modes | kwargs {kw}")
        res = cat(
            key=jax.random.PRNGKey(seed), logpi=logpi, logpi_single=logpi_single,
            grad_logpi_single=grad_logpi_single, modes=modes_j, betas=None, dtype=dtype,
            callback=callback, **kw,
        )
        cost = ob.OracleCost(value=res.n_target_evals, value_grad=res.n_target_grad_evals) + pilot_cost
        return res, cost, dict(kwargs=kw, n_modes=len(modes_j), n_betas=len(res.betas_used),
                               phase_times=getattr(res, "phase_times", None))

    entry = LAST_ENTRIES["CAT"] = _guarded("CAT", _run, log_prefix, mode_discovery_cost)
    if entry is not None:
        # Evaluation only (outside the timed transport, not in the cost; a failure here never fails
        # CAT): ESS / n of pi / q over the final mixture draws, and the selected validation round's
        # per-mode endpoint ESS / n and E_q |v|^2 along the grid.
        try:
            res = entry["res"]
            x = res.final_snapshot
            logp = jnp.concatenate([logpi(x[i:i + 2048]) for i in range(0, x.shape[0], 2048)])
            lw = np.asarray(logp, np.float64) - np.asarray(res.final_logq, np.float64)
            lw = lw[np.isfinite(lw)]
            w = np.exp(lw - lw.max())
            sel = next((i for i, g in enumerate(res.diagnostics) if g["refresh"] == res.selected_refresh), -1)
            entry["info"].update(
                final_ess=float(w.sum() ** 2 / np.sum(w * w) / lw.size),
                val_endpoint_ess=np.asarray(res.ess_ratio_history[sel])[:, -1].tolist(),
                val_kinetic_energy=np.asarray(res.kinetic_energy_history[sel]).tolist())
        except Exception as e:
            print(f"{log_prefix} [CAT] ESS evaluation failed: {e!r}")
        print(f"{log_prefix} [CAT] transport wall time: {entry['wall_time_s']:.1f}s | transport cost "
              f"{weighted(d, entry['transport_cost']):.4e} (value + {d} x grad) | "
              f"final ESS {entry['info'].get('final_ess', float('nan')):.3f} | final weights "
              f"{np.round(np.asarray(entry['res'].final_weights), 3).tolist()}")
    return entry


def run_cnf_gat(
    target: str, *, seed: int, dtype, d: int, logpi, logpi_single, grad_logpi_single,
    modes, cat_entry: Dict[str, Any], mode_discovery_cost: ob.OracleCost,
    n_snapshot: int, n_calibrate: int, log_prefix: str, skip_methods: frozenset = frozenset(),
    callbacks: Optional[Dict[str, Any]] = None,
) -> Dict[str, Optional[Dict[str, Any]]]:
    """Run the CNF and GAT on `modes` at CAT's realized transport cost.

    Returns {label: None (skipped/failed) or an entry as `run_cat`'s}. Each method
    is guarded independently. `callbacks`: {label: per-validation callback} (evaluation only).
    """
    callbacks = callbacks or CALLBACKS
    budget = weighted(d, cat_entry["transport_cost"])
    print(f"{log_prefix} CNF/GAT: {len(modes)} modes | n_calibrate={n_calibrate} | "
          f"n_snapshot={n_snapshot} | CAT transport budget {budget:.4e} (value + {d} x grad)")

    def _cnf():
        betas, kw = resolve_cnf(target, n=n_snapshot, budget=budget, n_modes=len(modes), d=d,
                                n_cat_betas=len(cat_entry["res"].betas_used))
        kw["final_calibration"] = n_calibrate
        print(f"{log_prefix} [CNF] kwargs {kw}")
        res = cnf(
            key=jax.random.PRNGKey(seed), logpi=logpi, modes=list(modes), betas=jnp.asarray(betas),
            dtype=dtype, callback=callbacks.get("CNF"), **kw,
        )
        cost = ob.OracleCost(value=res.n_target_evals, value_grad=res.n_target_grad_evals)
        return res, cost, dict(kwargs=kw, n_modes=len(modes), params_per_mode=res.n_params_per_mode,
                               phase_times=res.phase_times)

    def _gat():
        n_cat = cat_entry["info"]["n_modes"]
        kw, pilot_cost = resolve_gat(target, logpi=logpi, modes=list(modes), n_cat_modes=n_cat, d=d,
                                     seed=seed, n=n_snapshot, budget=budget)
        print(f"{log_prefix} [GAT] CAT's config (J={n_cat} folded) | kwargs {kw}")
        res = gat_fit(
            jax.random.PRNGKey(seed), logpi, logpi_single, d, modes=list(modes),
            grad_logpi_single=grad_logpi_single, betas=None, dtype=dtype, callback=callbacks.get("GAT"), **kw,
        )
        cost = ob.OracleCost(value=res.n_target_evals, value_grad=res.n_target_grad_evals) + pilot_cost
        return res, cost, dict(kwargs=kw, n_modes=len(modes), n_params=res.n_params,
                               n_betas=len(res.betas_used), final_ess=res.final_ess, log_Z=res.log_Z,
                               log_Z_rate=res.log_Z_rate, phase_times=res.phase_times)

    out: Dict[str, Optional[Dict[str, Any]]] = {}
    for label, fn in (("CNF", _cnf), ("GAT", _gat)):
        if label in skip_methods:
            print(f"{log_prefix} [{label}] skipped (--skip-methods)")
            out[label] = None
            continue
        entry = out[label] = LAST_ENTRIES[label] = _guarded(label, fn, log_prefix, mode_discovery_cost)
        if entry is None:
            continue
        res = entry["res"]
        print(f"{log_prefix} [{label}] transport wall time: {entry['wall_time_s']:.1f}s | "
              f"cost / CAT transport {weighted(d, entry['transport_cost']) / budget:.3f} | "
              + (f"final weights {np.round(np.asarray(res.final_weights), 3).tolist()}"
                 if hasattr(res, "final_weights") else f"final ESS {res.final_ess:.3f}"))
    return out


def run_gmm(target: str, *, seed: int, modes, n_snapshot: int, dtype, mvn_jitter0: float = 1e-10) -> Dict[str, Any]:
    """The initial Laplace mixture b = sum_j a_j N(mu_j, Sigma_j) that CAT transports
    (CAT's modes for `target`, i.e. after its `min_mode_weight` pruning if any, with their
    anchors, covariances and Laplace weights a_j from mode discovery), sampled directly:
    the "GMM" baseline. No oracle calls beyond the shared discovery.

    Returns dict(x, logq, wall_time_s, weights, means): `n_snapshot` draws and their exact
    log b, and b's normalized weights a_j and means (CAT's modes, in discovery order).
    """
    from jax.scipy.special import logsumexp
    from src.mode_explore.density import modes_to_arrays
    from src.utils.numeric import chol_jitter_jax, sym_jax

    t0 = time.time()
    modes = cat_modes(target, modes)
    mu, Sigma, logw = modes_to_arrays(modes, dtype=dtype, anchor_means=True)
    L = chol_jitter_jax(sym_jax(Sigma), jitter0=mvn_jitter0, max_tries=12)          # (J, d, d)
    loga = logw - logsumexp(logw)
    k_comp, k_eps = jax.random.split(jax.random.PRNGKey(seed))
    comp = jax.random.categorical(k_comp, loga, shape=(int(n_snapshot),))
    eps = jax.random.normal(k_eps, (int(n_snapshot), mu.shape[1]), dtype=dtype)
    x = jnp.zeros_like(eps)
    for j in range(mu.shape[0]):     # per mode: gathering L[comp] is (n, d, d), too big at lda's d
        x = jnp.where((comp == j)[:, None], mu[j] + eps @ L[j].T, x)

    def comp_logpdf(mu_j, L_j):
        y = jax.scipy.linalg.solve_triangular(L_j, (x - mu_j).T, lower=True).T
        return (-0.5 * jnp.sum(y * y, -1) - jnp.sum(jnp.log(jnp.diag(L_j)))
                - 0.5 * mu.shape[1] * jnp.log(2 * jnp.pi))

    logq = logsumexp(jax.vmap(comp_logpdf)(mu, L) + loga[:, None], axis=0)
    x, logq = np.asarray(x), np.asarray(logq)
    return dict(x=x, logq=logq, wall_time_s=time.time() - t0,
                weights=np.exp(np.asarray(loga)).tolist(), means=np.asarray(mu).tolist())


def summary_extras(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Extra summary.json fields for a CAT/CNF/GAT entry."""
    res = entry["res"]
    return dict(
        transport_wall_time_s=entry["wall_time_s"],
        final_weights=np.asarray(res.final_weights).tolist() if hasattr(res, "final_weights") else None,
        selected_refresh=res.selected_refresh,
        kwargs=entry["info"]["kwargs"],
        n_modes_trained=entry["info"]["n_modes"],
        phase_times=entry["info"].get("phase_times"),
        **{k: entry["info"][k] for k in ("n_params", "n_betas", "final_ess", "val_endpoint_ess",
                                          "val_kinetic_energy", "log_Z", "log_Z_rate") if k in entry["info"]},
    )
