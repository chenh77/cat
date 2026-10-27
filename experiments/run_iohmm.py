#!/usr/bin/env python
"""experiments/run_iohmm.py

Benchmark run on posteriordb's `iohmm_reg_simulated-iohmm_reg` (input-output HMM
regression, T = 500, K = 3 states, M = 4-dimensional input, d = 29 unconstrained;
`src.targets.IohmmReg`). The posterior is exactly K! = 6-fold label-symmetric
(`IohmmReg.permute_components`), so it has 6 modes of equal weight; the run also saves the
6 canonical mode centers (`mode_centers`: every label permutation of the discovered mode
with the highest log density) for the evaluation.

Methods, budget matching and outputs as in `run_skewnormal.py` (see `benchmark.py`).

Usage:
  python experiments/run_iohmm.py --seed 2026
"""
from __future__ import annotations

import argparse
import itertools
import os
from typing import Any, Dict

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import common
import benchmark as bm
from src.samplers.apt import run_adaptive_pt
from src.samplers.alps import alps_sample
from src.samplers.fab import fab_train
from src.samplers.temperflow import temperflow_fit
from src.samplers.aft import aft_fit
from transport_stage import run_cat, run_cnf_gat, run_gmm

TARGET = "iohmm"
CONFIG: Dict[str, Any] = dict(
    # sample sizes and the fixed ladder of APT
    n=512, n_snapshot=2000, K_betas=6, beta1=1e-3, budget_margin=1.0,
    # mode discovery
    beta_hot=0.5, mh="mala", step_size=0.2, v=40, optimize_steps=2000, max_new_modes=6,
    max_lbfgs_particles=200, restart_prob=0.05, restart_scale=6.0, eig_floor=1e-4, n_explore_rounds=8,
    # discrete-flow baselines
    n_train=256, n_coupling=6, hidden=64, depth=3, lr=1e-3,
    fab_lr=1e-5,
)
BASELINE_FIXED_KWARGS: Dict[str, Any] = dict(
    pilot_temperflow_stages=44, pilot_aft_stages=6,
    fab_n_intermediate_distributions=4, fab_hmc_n_outer_steps=3, fab_hmc_n_inner_steps=5,
    fab_n_updates=4, aft_mala_steps=25,
)


def permutation_mode_centers(target, anchor_mu: np.ndarray) -> np.ndarray:
    """All K! label-permuted copies of `anchor_mu`, via the exact `log_prob`-preserving
    `permute_components` map."""
    perms = list(itertools.permutations(range(target.K)))
    anchor = jnp.asarray(anchor_mu)
    return np.asarray(jnp.stack([target.permute_components(anchor, jnp.array(p)) for p in perms]))


def run_experiment(seed: int, out_dir: str, skip_methods: frozenset = frozenset()) -> Dict[str, Any]:
    cfg = dict(CONFIG)
    tag = TARGET
    dtype = jnp.float64
    n, n_snapshot = cfg["n"], cfg["n_snapshot"]
    betas = bm.beta_schedule_geom(K=cfg["K_betas"], beta1=cfg["beta1"])
    apt_betas = betas[1:]

    target = common.build_iohmm()
    d = target.dim
    logpi_single, grad_logpi_single, logpi = bm.make_logpi(target, dtype)
    print(f"[{tag}] d={d}, T={target.T}, K={target.K}, M={target.M}, seed={seed}, "
          f"betas={[f'{b:.2e}' for b in betas]}")

    # ---- mode discovery (shared by every method): expect K! = 6 modes
    modes, t_discover = bm.discover_modes(common.build_iohmm, d, cfg, seed=seed, dtype=dtype,
                                          eig_floor=cfg["eig_floor"])
    print(f"[{tag}] modes discovered: {len(modes)} (expected K!=6) | discovery wall time: {t_discover:.1f}s")
    mode_discovery_cost = bm.discovery_cost(d, cfg, len(modes), "hessian")

    # ---- CAT: the budget standard
    cat_entry = run_cat(TARGET, seed=seed, dtype=dtype, d=d, logpi=logpi, logpi_single=logpi_single,
                        grad_logpi_single=grad_logpi_single, modes=modes,
                        mode_discovery_cost=mode_discovery_cost, n_snapshot=n_snapshot, n_calibrate=n,
                        log_prefix=f"[{tag}]")
    if cat_entry is None:
        raise RuntimeError("CAT failed: no budget to match the other methods to")
    cat_transport_cost = cat_entry["transport_cost"]
    budget_cfg = bm.matched_budgets(cat_transport_cost, d=d, cfg=cfg, n_temperatures=len(apt_betas),
                                    fixed_kwargs=BASELINE_FIXED_KWARGS)
    print(f"[{tag}] matched baseline budgets (CAT transport, value + d*grad): {budget_cfg}")

    # canonical K! permutation-mode centers, anchored at the discovered mode with the highest
    # log density (any discovered mode works: permute_components preserves log_prob exactly)
    logpi_at_modes = np.asarray([float(logpi_single(m.mu)) for m in modes])
    mode_centers = permutation_mode_centers(target, np.asarray(modes[int(np.argmax(logpi_at_modes))].mu))

    # ---- baselines, each at its matched budget
    run = bm.guarded(tag, skip_methods)
    res_gmm, _ = run("GMM", run_gmm, TARGET, seed=seed, modes=modes, n_snapshot=n_snapshot, dtype=dtype)
    t_gmm = res_gmm["wall_time_s"] if res_gmm is not None else 0.0
    apt_kw = dict(n_iterations=budget_cfg["apt_iterations"],
                  burnin=int(0.3 * budget_cfg["apt_iterations"]),
                  thinning=max(1, int(0.7 * budget_cfg["apt_iterations"]) // 2500),
                  initial_step_size=0.3, x0_scale=3.0, verbose=True)
    res_apt, t_apt = run("APT", run_adaptive_pt, jax.random.PRNGKey(seed), logpi_single, d, betas=apt_betas,
                         use_deo=False, modes=modes, **apt_kw)
    res_apt_deo, t_apt_deo = run("APT-DEO", run_adaptive_pt, jax.random.PRNGKey(seed), logpi_single, d,
                                 betas=apt_betas, use_deo=True, modes=modes, **apt_kw)
    res_alps, t_alps = run("ALPS", alps_sample, jax.random.PRNGKey(seed), logpi, logpi_single, d=d, modes=modes,
                           n_samples=n_snapshot, value_budget=budget_cfg["alps_value_budget"], verbose=True)
    res_fab, t_fab = run(
        "FAB", fab_train, jax.random.PRNGKey(seed), logpi, logpi_single, d, modes=modes,
        n_rounds=budget_cfg["fab_rounds"], n_ais=n, n_intermediate_distributions=4,
        hmc_n_outer_steps=3, hmc_n_inner_steps=5, hmc_init_step_size=0.3,
        buffer_size=8192, batch_size=n // 4, n_updates_per_smc_forward_pass=4,
        n_coupling=cfg["n_coupling"], hidden=cfg["hidden"], depth=cfg["depth"], lr=cfg["fab_lr"],
        n_snapshot=n_snapshot, verbose=True)
    res_tf, t_tf = run(
        "TemperFlow", temperflow_fit, jax.random.PRNGKey(seed), logpi, logpi_single, d, modes=modes,
        beta0=0.05, s_max=2.0, max_stages=60, n_probe=n, n=n, n_train=cfg["n_train"], n_snapshot=n_snapshot,
        train_steps=budget_cfg["temperflow_train_steps"], n_coupling=cfg["n_coupling"],
        hidden=cfg["hidden"], depth=cfg["depth"], lr=cfg["lr"], verbose=True)
    # TemperFlow returns normally even when its first stage fails (non-finite snapshot)
    if res_tf is not None and not np.all(np.isfinite(np.asarray(res_tf.snapshots[-1]))):
        print(f"[{tag}] [TemperFlow] FAILED: final snapshot is non-finite.")
        res_tf = None
    res_aft, t_aft = run(
        "AFT", aft_fit, jax.random.PRNGKey(seed), logpi, logpi_single, d, modes=modes,
        n=n_snapshot, target_ess=0.5, max_stages=40, train_steps=budget_cfg["aft_train_steps"],
        n_coupling=cfg["n_coupling"], hidden=cfg["hidden"], depth=cfg["depth"], lr=cfg["lr"],
        mala_step_size=0.1, mala_steps=25, schedule_min_growth=1.3, verbose=True)

    # ---- the CNF and GAT at CAT's transport cost
    vc = {"CAT": cat_entry, **run_cnf_gat(
        TARGET, seed=seed, dtype=dtype, d=d, logpi=logpi, logpi_single=logpi_single,
        grad_logpi_single=grad_logpi_single, modes=modes, cat_entry=cat_entry,
        mode_discovery_cost=mode_discovery_cost, n_snapshot=n_snapshot, n_calibrate=n,
        log_prefix=f"[{tag}]", skip_methods=skip_methods)}

    candidates = {
        "GMM": (res_gmm, t_gmm + t_discover), "APT": (res_apt, t_apt + t_discover),
        "APT-DEO": (res_apt_deo, t_apt_deo + t_discover), "ALPS": (res_alps, t_alps + t_discover),
        "FAB": (res_fab, t_fab + t_discover), "TemperFlow": (res_tf, t_tf + t_discover),
        "AFT": (res_aft, t_aft + t_discover),
        **{l: (e["res"], e["wall_time_s"] + t_discover) if e else (None, 0.0) for l, e in vc.items()},
    }
    return bm.save_run(
        tag=tag, out_dir=out_dir, cfg=cfg, seed=seed, d=d, betas=betas, apt_betas=apt_betas,
        budget_cfg=budget_cfg, fixed_kwargs=BASELINE_FIXED_KWARGS, mode_discovery_cost=mode_discovery_cost,
        n_modes_discovered=len(modes), cat_transport_cost=cat_transport_cost, candidates=candidates, vc=vc,
        skip_methods=skip_methods, dtype=dtype, extra_npz=dict(mode_centers=mode_centers))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    bm.add_run_args(parser, TARGET)
    args = parser.parse_args()
    out_dir = args.out_dir or os.path.join(common.benchmark_dir(TARGET), f"seed_{args.seed}")
    run_experiment(seed=args.seed, out_dir=out_dir, skip_methods=bm.parse_skip(args.skip_methods))


if __name__ == "__main__":
    main()
