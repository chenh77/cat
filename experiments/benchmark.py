"""experiments/benchmark.py

Pieces shared by the `run_<target>.py` benchmark runs: the shared mode discovery,
guarded method calls, and assembling and saving one seed's results.

Every run saves, under `experiments/results/benchmark/<target>/seed_<seed>/`:
  - config.json    -- resolved run config (hyperparameters, matched baseline budgets,
                      mode-discovery and CAT transport costs, failed methods)
  - summary.json   -- per (successful) method: wall time (including the shared mode
                      discovery), n_samples, oracle-cost triplet (log-target / value+grad
                      / Hessian calls, including discovery), transport-only weighted cost
                      value + d * value_grad and its ratio to CAT's; CAT, the CNF and GAT
                      also their kwargs and diagnostics; GMM the Laplace weights and means
  - rung_ess.json  -- per method: the per-rung ESS of the methods with an annealing
                      ladder (CAT: validation ESS per mode and checkpoint; TemperFlow,
                      AFT); null for the others
  - samples.npz    -- final samples per method, and log q of the samples of the methods
                      with an exact density (CAT, CNF, GAT, GMM, FAB, TemperFlow)
"""
from __future__ import annotations

import contextlib
import os
import time
from typing import Any, Callable, Dict

import numpy as np
import jax
import jax.numpy as jnp

from common import npz_key, write_json
from src.mode_explore.density import StandardNormal
from src.mode_explore.explore import multi_round_mode_explore
from src.utils import budget as ob
from transport_stage import VC_LABELS, summary_extras

METHOD_LABELS = ["CAT", "GMM", "APT", "APT-DEO", "ALPS", "FAB", "TemperFlow", "AFT", "CNF", "GAT"]
# Methods with a per-rung ESS diagnostic.
RUNG_LADDER_METHODS = ("CAT", "TemperFlow", "AFT")
# Baselines whose final samples come with an exact log q (saved as logq_<label>).
EXACT_Q_BASELINES = ("FAB", "TemperFlow")
# Their final samples come from `.samples` (no flow snapshot).
VIA_SAMPLES = ("APT", "APT-DEO", "ALPS", "FAB", "AFT")
# Non-adaptive baselines (exact repeat count, no realized-cost shortfall, unlike
# TemperFlow's and AFT's adaptive stopping) get no budget margin: a true 1x match.
FIXED_ITERATION_MARGINS = {"apt": 1.0, "alps": 1.0, "fab": 1.0}


def beta_schedule_geom(K: int, beta1: float):
    return [0.0] + list(np.geomspace(beta1, 1.0, int(K)))


def make_logpi(target, dtype):
    """(logpi_single, grad_logpi_single, logpi) of a target with a batched `log_prob`."""
    def logpi_single(x):
        x = jnp.asarray(x, dtype=dtype)
        return target.log_prob(x[None, :])[0]
    return logpi_single, jax.grad(logpi_single), jax.vmap(logpi_single)


def discover_modes(build_target: Callable[[], Any], d: int, cfg: Dict[str, Any], *, seed: int, dtype,
                   on_cpu: bool = True, **explore_kwargs):
    """Shared mode discovery (`multi_round_mode_explore` from N(0, I) particles).

    `on_cpu`: run on a CPU-committed copy of the target (`build_target()`), then move the
    modes to the default device. Discovery's L-BFGS + Hessian phase is a sequential,
    dispatch-latency-bound per-particle loop with host syncs, far slower on a GPU without
    fast float64 than on the CPU, for negligible FLOPs.
    Returns (modes, wall time)."""
    base = StandardNormal(d=d)
    ctx = jax.default_device(jax.devices("cpu")[0]) if on_cpu else contextlib.nullcontext()
    with ctx:
        logpi_single, grad_logpi_single, _ = make_logpi(build_target(), dtype)
        t0 = time.time()
        modes, _ = multi_round_mode_explore(
            jax.random.PRNGKey(seed), lambda k, n_: base.sample(k, n_, dtype=dtype), logpi_single,
            beta_hot=cfg["beta_hot"], n=cfg["n"], v=cfg["v"], n_rounds=cfg["n_explore_rounds"], mh=cfg["mh"],
            mh_kwargs={"step_size": cfg["step_size"]}, grad_logpi_single=grad_logpi_single,
            optimize_steps=cfg["optimize_steps"], max_new_modes=cfg["max_new_modes"],
            max_lbfgs_particles=cfg["max_lbfgs_particles"], restart_prob=cfg["restart_prob"],
            restart_scale=cfg["restart_scale"], verbose=True, **explore_kwargs,
        )
        t_discover = time.time() - t0
    if on_cpu:
        for m in modes:
            m.mu = jax.device_put(m.mu)
            m.Sigma = jax.device_put(m.Sigma)
            if m.anchor_mu is not None:
                m.anchor_mu = jax.device_put(m.anchor_mu)
    return modes, t_discover


def discovery_cost(d: int, cfg: Dict[str, Any], n_modes: int, cov_method: str) -> ob.OracleCost:
    return ob.mode_explore_cost(
        dim=d, n=cfg["n"], v=cfg["v"], optimize_steps=cfg["optimize_steps"],
        max_lbfgs_particles=cfg["max_lbfgs_particles"], rounds=cfg["n_explore_rounds"],
        kernel=cfg["mh"], cov_method=cov_method, final_modes=n_modes,
    )


def matched_budgets(cat_transport_cost, *, d, cfg, n_temperatures, fixed_kwargs):
    """Every baseline's repeat count / budget matched to CAT's transport cost, weighted as
    value + d * value_grad. ALPS and AFT are particle methods whose population doubles as
    their final sample count, so they are matched and run at `n_snapshot` draws."""
    n, n_snapshot = cfg["n"], cfg["n_snapshot"]
    return ob.matched_baseline_config(
        cat_transport_cost, dim=d, n_temperatures=n_temperatures, n_chains=n_snapshot,
        n=n, n_train=cfg["n_train"], n_probe=n, fab_batch_size=n // 4, aft_n=n_snapshot,
        margin=cfg["budget_margin"], method_margins=FIXED_ITERATION_MARGINS,
        grad_weight=d, hessian_grad_units=0, **fixed_kwargs,
    )


def guarded(tag: str, skip_methods: frozenset):
    """`run(label, fn, *args, **kwargs) -> (result or None, wall time)`: skips `label` if
    asked, and records a failure instead of raising, so one baseline's numerical failure
    does not discard the seed's other results."""
    def run(label: str, fn, *args, **kwargs):
        if label in skip_methods:
            print(f"[{tag}] [{label}] skipped (--skip-methods)")
            return None, 0.0
        t0 = time.time()
        try:
            res = fn(*args, **kwargs)
            wall = time.time() - t0
            print(f"[{tag}] [{label}] wall time: {wall:.1f}s")
            return res, wall
        except Exception as e:
            wall = time.time() - t0
            print(f"[{tag}] [{label}] FAILED after {wall:.1f}s: {e!r}")
            return None, wall
    return run


def save_run(*, tag, out_dir, cfg, seed, d, betas, apt_betas, budget_cfg, fixed_kwargs,
             mode_discovery_cost, n_modes_discovered, cat_transport_cost, candidates, vc, skip_methods,
             dtype, extra_npz=None):
    """Assemble one seed's results and save them (see the module docstring).

    `candidates`: {label: (result or None, wall time incl. discovery)} of every method;
    `vc`: the CAT/CNF/GAT entries of `transport_stage`."""
    n, n_snapshot = cfg["n"], cfg["n_snapshot"]
    all_results = {k: v for k, v in candidates.items() if v[0] is not None}
    failed_methods = [k for k, v in candidates.items() if v[0] is None and k not in skip_methods]
    print(f"[{tag}] succeeded: {list(all_results)}")
    if failed_methods:
        print(f"[{tag}] failed (excluded from summary): {failed_methods}")

    def samples_of(label, res):
        if label == "GMM":
            return res["x"]
        if label in VC_LABELS:
            return res.final_snapshot
        return res.samples if label in VIA_SAMPLES else res.snapshots[-1]

    xs_all = {label: np.asarray(jnp.asarray(samples_of(label, res), dtype=dtype))
              for label, (res, _) in all_results.items()}
    # FAB and TemperFlow are flows with an exact density: save log q of their samples
    logq_all = {
        label: np.concatenate([np.asarray(res.q_log_prob(jnp.asarray(xs_all[label][i:i + 1024])))
                               for i in range(0, xs_all[label].shape[0], 1024)])
        for label, (res, _) in all_results.items() if label in EXACT_Q_BASELINES
    }
    if "GMM" in all_results:
        logq_all["GMM"] = all_results["GMM"][0]["logq"]

    fk = fixed_kwargs
    oracle_costs = ob.realized_costs(
        results={k: v[0] for k, v in all_results.items() if k not in VC_LABELS and k != "GMM"},
        config=budget_cfg, n_temperatures=len(apt_betas), n=n, n_train=cfg["n_train"], n_probe=n,
        fab_batch_size=n // 4, fab_n_intermediate_distributions=fk["fab_n_intermediate_distributions"],
        fab_hmc_n_outer_steps=fk["fab_hmc_n_outer_steps"], fab_hmc_n_inner_steps=fk["fab_hmc_n_inner_steps"],
        fab_n_updates=fk["fab_n_updates"], aft_mala_steps=fk["aft_mala_steps"],
        alps_n_chains=n_snapshot, aft_n=n_snapshot,
    )
    # every method uses the discovered modes: charge the shared discovery to all
    # (the CAT/CNF/GAT costs from transport_stage already include it)
    oracle_costs = {k: c + mode_discovery_cost for k, c in oracle_costs.items()}
    if "GMM" in all_results:
        oracle_costs["GMM"] = mode_discovery_cost      # no transport
    oracle_costs.update({l: e["cost"] for l, e in vc.items() if e})

    cat_weighted = ob.transport_weighted(cat_transport_cost, d)
    summary: Dict[str, Any] = {}
    rung_ess: Dict[str, Any] = {}
    for label, (res, wall_time) in all_results.items():
        cost = oracle_costs[label]
        weighted = ob.transport_weighted(cost - mode_discovery_cost, d)
        summary[label] = {
            "wall_time_s": wall_time,
            "n_samples": int(xs_all[label].shape[0]),
            "log_target_calls": cost.value,
            "value_grad_calls": cost.value_grad,
            "hessian_calls": cost.hessian,
            "weighted_cost_1_d": weighted,
            "cost_ratio_to_cat": weighted / cat_weighted,
        }
        if label in VC_LABELS:
            summary[label].update(summary_extras(vc[label]))
        elif label == "GMM":     # b's weights and means, in discovery order (CAT's modes)
            summary[label].update(mode_weights=res["weights"], mode_means=res["means"])

        if label == "CAT":      # last validation round's ESS per mode and checkpoint, (J, K)
            ess = np.asarray(res.ess_ratio_history[-1]) if res.ess_ratio_history else np.zeros((0, 0))
            rung_ess[label] = dict(
                betas=[float(b) for b in res.betas_used[1:]],
                ess_ratio_per_mode=ess[:, 1:].tolist(),
            ) if ess.ndim == 2 and ess.shape[-1] == len(res.betas_used) else None
        elif label in RUNG_LADDER_METHODS:
            rung_ess[label] = dict(
                betas=[float(b) for b in res.betas_used[1:]],
                ess_ratio=[float(v) for v in res.ess_ratio_history],
            )
        else:
            rung_ess[label] = None

    os.makedirs(out_dir, exist_ok=True)
    resolved_cfg = dict(cfg)
    resolved_cfg.update(dict(
        target=tag, seed=seed, betas=betas, apt_betas=apt_betas,
        budget_cfg=budget_cfg, mode_discovery_cost=mode_discovery_cost.triplet(),
        cat_transport_cost=cat_transport_cost.triplet(),
        budget_basis="CAT transport cost (shared mode discovery excluded)",
        cost_convention="transport only: value=1, value_grad=dim",
        n_modes_discovered=n_modes_discovered, fixed_iteration_margins=FIXED_ITERATION_MARGINS,
        failed_methods=failed_methods, skipped_methods=sorted(skip_methods),
    ))
    write_json(os.path.join(out_dir, "config.json"), resolved_cfg)
    write_json(os.path.join(out_dir, "summary.json"), summary)
    write_json(os.path.join(out_dir, "rung_ess.json"), rung_ess)

    samples_npz = {npz_key(label): xs_all[label] for label in all_results}
    samples_npz.update({f"logq_{l}": np.asarray(e["res"].final_logq) for l, e in vc.items() if e})
    samples_npz.update({f"logq_{npz_key(label)}": v for label, v in logq_all.items()})
    samples_npz.update(extra_npz or {})
    np.savez(os.path.join(out_dir, "samples.npz"), **samples_npz)
    print(f"[{tag}] saved config.json, summary.json, rung_ess.json, samples.npz to {out_dir}")

    present = [l for l in METHOD_LABELS if l in all_results]
    header = f"{'metric':>18} | " + " | ".join(f"{l:>16}" for l in present)
    print(header)
    print("-" * len(header))
    for k in ["wall_time_s", "n_samples", "log_target_calls", "value_grad_calls",
              "hessian_calls", "weighted_cost_1_d", "cost_ratio_to_cat"]:
        def fmt(v):
            if v is None:
                return f"{'None':>16}"
            if isinstance(v, float):
                return f"{v:16.4g}"
            if isinstance(v, int):
                return f"{v:16,d}"
            return f"{str(v):>16}"
        print(f"{k:>18} | " + " | ".join(fmt(summary[l].get(k)) for l in present))
    return dict(summary=summary, rung_ess=rung_ess, out_dir=out_dir)


def add_run_args(parser, target: str):
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--out-dir", default=None,
                        help=f"Defaults to experiments/results/benchmark/{target}/seed_<seed>/")
    parser.add_argument("--skip-methods", default="",
                        help="Comma-separated method labels to skip (CAT always runs: it sets the budget).")


def parse_skip(spec: str) -> frozenset:
    return frozenset(m.strip() for m in spec.split(",") if m.strip())
