#!/usr/bin/env python
"""experiments/run_lda.py

Benchmark run on posteriordb's `three_men1-ldaK2` (latent Dirichlet allocation with K = 2
topics, V = 249 words, M = 6 documents, d = 502 unconstrained, prior-Gaussian coordinates;
`src.targets.LdaK2`). The posterior is exactly 2-fold topic-label symmetric
(`LdaK2.topic_swap`), so it has two modes of equal weight; mode discovery usually finds both
in one round, and several diffuse-restart rounds make that reliable.

Methods, budget matching and outputs as in `run_skewnormal.py` (see `benchmark.py`).

Usage:
  python experiments/run_lda.py --seed 2026
"""
from __future__ import annotations

import argparse
import os
from typing import Any, Dict

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

TARGET = "lda"
CONFIG: Dict[str, Any] = dict(
    # sample sizes and the fixed ladder of APT
    n=256, n_snapshot=5000, K_betas=6, beta1=1e-2, budget_margin=1.0,
    # mode discovery
    beta_hot=0.5, mh="mala", step_size=0.1, v=25, optimize_steps=800, max_new_modes=4,
    max_lbfgs_particles=80, n_explore_rounds=4, restart_prob=0.05, restart_scale=4.0, eig_floor=1e-3,
    # discrete-flow baselines
    n_train=160, n_coupling=6, hidden=128, depth=3, lr=1e-3,
    fab_lr=1e-4,
    # endpoint calibration sample size per mode of CAT and the CNF
    cat_n_calibrate=4096,
)
BASELINE_FIXED_KWARGS: Dict[str, Any] = dict(
    pilot_temperflow_stages=6, pilot_aft_stages=7,
    fab_n_intermediate_distributions=4, fab_hmc_n_outer_steps=3, fab_hmc_n_inner_steps=4,
    fab_n_updates=4, aft_mala_steps=25,
)


def run_experiment(seed: int, out_dir: str, skip_methods: frozenset = frozenset()) -> Dict[str, Any]:
    cfg = dict(CONFIG)
    tag = TARGET
    dtype = jnp.float64
    n, n_snapshot = cfg["n"], cfg["n_snapshot"]
    betas = bm.beta_schedule_geom(K=cfg["K_betas"], beta1=cfg["beta1"])
    apt_betas = betas[1:]

    target = common.build_lda()
    d = target.dim
    logpi_single, grad_logpi_single, logpi = bm.make_logpi(target, dtype)
    print(f"[{tag}] d={d}, V={target.V}, M={target.M}, N={target.N} "
          f"({target.N_unique_doc_word} unique doc-word pairs), seed={seed}, betas={[f'{b:.2e}' for b in betas]}")

    # ---- mode discovery (shared by every method): expect 2 modes, one per topic-swap basin
    modes, t_discover = bm.discover_modes(
        lambda: target, d, cfg, seed=seed, dtype=dtype, on_cpu=False,
        alpha=0.05, distance_tol=None, cov_ridge=1e-6, eig_floor=cfg["eig_floor"], eig_cap=1e7,
        cov_method="hessian", cov_inflate=1.0)
    print(f"[{tag}] modes discovered: {len(modes)} (expected 2) | discovery wall time: {t_discover:.1f}s")
    mode_discovery_cost = bm.discovery_cost(d, cfg, len(modes), "hessian")

    # ---- CAT: the budget standard
    n_cal = cfg["cat_n_calibrate"]
    cat_entry = run_cat(TARGET, seed=seed, dtype=dtype, d=d, logpi=logpi, logpi_single=logpi_single,
                        grad_logpi_single=grad_logpi_single, modes=modes,
                        mode_discovery_cost=mode_discovery_cost, n_snapshot=n_snapshot, n_calibrate=n_cal,
                        log_prefix=f"[{tag}]")
    if cat_entry is None:
        raise RuntimeError("CAT failed: no budget to match the other methods to")
    cat_transport_cost = cat_entry["transport_cost"]
    budget_cfg = bm.matched_budgets(cat_transport_cost, d=d, cfg=cfg, n_temperatures=len(apt_betas),
                                    fixed_kwargs=BASELINE_FIXED_KWARGS)
    print(f"[{tag}] matched baseline budgets (CAT transport, value + d*grad): {budget_cfg}")

    # ---- baselines, each at its matched budget
    run = bm.guarded(tag, skip_methods)
    res_gmm, _ = run("GMM", run_gmm, TARGET, seed=seed, modes=modes, n_snapshot=n_snapshot, dtype=dtype)
    t_gmm = res_gmm["wall_time_s"] if res_gmm is not None else 0.0
    apt_kw = dict(n_iterations=budget_cfg["apt_iterations"],
                  burnin=int(0.33 * budget_cfg["apt_iterations"]),
                  thinning=max(1, int(0.67 * budget_cfg["apt_iterations"]) // 1500),
                  initial_step_size=0.15, x0_scale=1.0, verbose=True)
    res_apt, t_apt = run("APT", run_adaptive_pt, jax.random.PRNGKey(seed), logpi_single, d, betas=apt_betas,
                         use_deo=False, modes=modes, **apt_kw)
    res_apt_deo, t_apt_deo = run("APT-DEO", run_adaptive_pt, jax.random.PRNGKey(seed), logpi_single, d,
                                 betas=apt_betas, use_deo=True, modes=modes, **apt_kw)
    res_alps, t_alps = run("ALPS", alps_sample, jax.random.PRNGKey(seed), logpi, logpi_single, d=d, modes=modes,
                           grad_logpi_single=grad_logpi_single, n_samples=n_snapshot,
                           value_budget=budget_cfg["alps_value_budget"], verbose=True)
    res_fab, t_fab = run(
        "FAB", fab_train, jax.random.PRNGKey(seed), logpi, logpi_single, d, modes=modes,
        n_rounds=budget_cfg["fab_rounds"], n_ais=n, n_intermediate_distributions=4,
        hmc_n_outer_steps=3, hmc_n_inner_steps=4, hmc_init_step_size=0.15,
        buffer_size=4096, batch_size=n // 4, n_updates_per_smc_forward_pass=4,
        n_coupling=cfg["n_coupling"], hidden=cfg["hidden"], depth=cfg["depth"], lr=cfg["fab_lr"],
        n_snapshot=n_snapshot, verbose=True)
    res_tf, t_tf = run(
        "TemperFlow", temperflow_fit, jax.random.PRNGKey(seed), logpi, logpi_single, d, modes=modes,
        max_stages=60, n_probe=n, n=n, n_train=cfg["n_train"], n_snapshot=n_snapshot,
        train_steps=budget_cfg["temperflow_train_steps"], n_coupling=cfg["n_coupling"],
        hidden=cfg["hidden"], depth=cfg["depth"], lr=cfg["lr"], verbose=True)
    res_aft, t_aft = run(
        "AFT", aft_fit, jax.random.PRNGKey(seed), logpi, logpi_single, d, modes=modes,
        n=n_snapshot, target_ess=0.5, max_stages=40, train_steps=budget_cfg["aft_train_steps"],
        n_coupling=cfg["n_coupling"], hidden=cfg["hidden"], depth=cfg["depth"], lr=cfg["lr"],
        mala_step_size=0.1, mala_steps=25, schedule_min_growth=1.3, verbose=True)

    # ---- the CNF and GAT at CAT's transport cost
    vc = {"CAT": cat_entry, **run_cnf_gat(
        TARGET, seed=seed, dtype=dtype, d=d, logpi=logpi, logpi_single=logpi_single,
        grad_logpi_single=grad_logpi_single, modes=modes, cat_entry=cat_entry,
        mode_discovery_cost=mode_discovery_cost, n_snapshot=n_snapshot, n_calibrate=n_cal,
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
        skip_methods=skip_methods, dtype=dtype)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    bm.add_run_args(parser, TARGET)
    args = parser.parse_args()
    out_dir = args.out_dir or os.path.join(common.benchmark_dir(TARGET), f"seed_{args.seed}")
    run_experiment(seed=args.seed, out_dir=out_dir, skip_methods=bm.parse_skip(args.skip_methods))


if __name__ == "__main__":
    main()
