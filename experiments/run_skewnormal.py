#!/usr/bin/env python
"""experiments/run_skewnormal.py

Benchmark run on the two-mode skew-normal mixture (`src.targets.build_two_mode`): d = 5,
mode 1 N(-c 1, I), mode 2 a product of skew-normals at +c 1 with alpha = 4, centers 10
apart, true weights (0.8, 0.2). The Laplace weights of the discovered modes are far from
the true ones (about 0.4 / 0.6), so the weight calibration matters.

Methods: CAT and the baselines GMM (CAT's initial Laplace mixture b, sampled directly),
APT, APT-DEO, ALPS, FAB, TemperFlow, AFT, CNF and GAT (skip any with --skip-methods), all
using the modes of one shared mode discovery: CAT and the CNF transport one field per mode,
ALPS leaps between the modes, and APT/APT-DEO, FAB, TemperFlow, AFT and GAT start from
their Laplace mixture (`modes=`). Every method's wall time includes the discovery time.

Budget: CAT runs first at a fixed configuration (`transport_stage.CAT_CONFIG`); its
realized transport cost (oracle calls minus the shared discovery), weighted as
value + d * value_grad, is the budget every other method is matched to. TemperFlow and AFT
size their training steps from a pilot stage count (`BASELINE_FIXED_KWARGS`), so their
realized cost can miss the budget either way; summary.json records each method's ratio.

This script only runs the samplers and saves their draws (see `benchmark.py` for the
outputs); `eval_summary.py` scores them.

Usage:
  python experiments/run_skewnormal.py --seed 2026
"""
from __future__ import annotations

import argparse
import os
from typing import Any, Dict

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import common
import benchmark as bm
from src.targets import SkewNormalMixture, build_two_mode
from src.samplers.apt import run_adaptive_pt
from src.samplers.alps import alps_sample
from src.samplers.fab import fab_train
from src.samplers.temperflow import temperflow_fit
from src.samplers.aft import aft_fit
from transport_stage import run_cat, run_cnf_gat, run_gmm

TARGET = "skewnormal"
CONFIG: Dict[str, Any] = dict(
    # target
    d=5, sep=10.0, alpha=4.0, w2=0.2,
    # sample sizes and the fixed ladder of APT
    n=256, n_snapshot=4096, K_betas=6, beta1=1e-3, budget_margin=1.0,
    # mode discovery
    beta_hot=1e-2, mh="ula", step_size=5.0, v=60, optimize_steps=60, max_new_modes=20,
    max_lbfgs_particles=100, restart_prob=0.1, restart_scale=20.0, n_explore_rounds=10,
    # discrete-flow baselines
    n_train=128, n_coupling=2, hidden=64, depth=2, lr=1e-2,
    fab_lr=1e-4, tf_lr=1e-3,
)
# Pilot stage counts (and AFT's resampled stages) that seed the cost matching of the
# adaptive baselines, and the baselines' own fixed hyperparameters.
BASELINE_FIXED_KWARGS: Dict[str, Any] = dict(
    pilot_temperflow_stages=3, pilot_aft_stages=6, pilot_aft_resampled=2,
    fab_n_intermediate_distributions=4, fab_hmc_n_outer_steps=3, fab_hmc_n_inner_steps=5,
    fab_n_updates=4, aft_mala_steps=25,
)


def build_target(cfg=CONFIG, dtype=jnp.float64) -> SkewNormalMixture:
    mus, sigmas, weights, alphas = build_two_mode(d=cfg["d"], sep=cfg["sep"], alpha=cfg["alpha"],
                                                  w2=cfg["w2"], dtype=dtype)
    return SkewNormalMixture(mus=mus, sigmas=sigmas, weights=weights, alpha=alphas)


def run_experiment(seed: int, out_dir: str, skip_methods: frozenset = frozenset()) -> Dict[str, Any]:
    cfg = dict(CONFIG)
    tag = TARGET
    dtype = jnp.float64
    d, n, n_snapshot = cfg["d"], cfg["n"], cfg["n_snapshot"]
    betas = bm.beta_schedule_geom(K=cfg["K_betas"], beta1=cfg["beta1"])
    apt_betas = betas[1:]

    target = build_target(cfg, dtype)
    logpi_single, grad_logpi_single, logpi = bm.make_logpi(target, dtype)
    print(f"[{tag}] d={d}, n_modes(true)={target.K}, seed={seed}, betas={[f'{b:.2e}' for b in betas]}")

    # ---- mode discovery (shared by every method)
    modes, t_discover = bm.discover_modes(lambda: build_target(cfg, dtype), d, cfg, seed=seed, dtype=dtype,
                                          cov_method="profile_max", cov_inflate=1.0)
    print(f"[{tag}] modes discovered: {len(modes)} | discovery wall time: {t_discover:.1f}s")
    mode_discovery_cost = bm.discovery_cost(d, cfg, len(modes), "profile_max")

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

    # ---- baselines, each at its matched budget
    run = bm.guarded(tag, skip_methods)
    res_gmm, _ = run("GMM", run_gmm, TARGET, seed=seed, modes=modes, n_snapshot=n_snapshot, dtype=dtype)
    t_gmm = res_gmm["wall_time_s"] if res_gmm is not None else 0.0
    apt_kw = dict(n_iterations=budget_cfg["apt_iterations"],
                  burnin=int(0.375 * budget_cfg["apt_iterations"]),
                  thinning=max(1, int(0.625 * budget_cfg["apt_iterations"]) // 2500),
                  initial_step_size=0.5, x0_scale=5.0, verbose=True)
    res_apt, t_apt = run("APT", run_adaptive_pt, jax.random.PRNGKey(seed), logpi_single, d, betas=apt_betas,
                         use_deo=False, modes=modes, **apt_kw)
    res_apt_deo, t_apt_deo = run("APT-DEO", run_adaptive_pt, jax.random.PRNGKey(seed), logpi_single, d,
                                 betas=apt_betas, use_deo=True, modes=modes, **apt_kw)
    res_alps, t_alps = run("ALPS", alps_sample, jax.random.PRNGKey(seed), logpi, logpi_single, d=d, modes=modes,
                           n_samples=n_snapshot, value_budget=budget_cfg["alps_value_budget"], verbose=True)
    res_fab, t_fab = run(
        "FAB", fab_train, jax.random.PRNGKey(seed), logpi, logpi_single, d, modes=modes,
        n_rounds=budget_cfg["fab_rounds"], n_ais=n, n_intermediate_distributions=4,
        hmc_n_outer_steps=3, hmc_n_inner_steps=5, hmc_init_step_size=1.0,
        buffer_size=8192, batch_size=n // 4, n_updates_per_smc_forward_pass=4,
        n_coupling=cfg["n_coupling"], hidden=cfg["hidden"], depth=cfg["depth"], lr=cfg["fab_lr"],
        n_snapshot=n_snapshot, verbose=True)
    res_tf, t_tf = run(
        "TemperFlow", temperflow_fit, jax.random.PRNGKey(seed), logpi, logpi_single, d, modes=modes,
        max_stages=50, n_probe=n, n=n, n_train=cfg["n_train"], n_snapshot=n_snapshot,
        train_steps=budget_cfg["temperflow_train_steps"], n_coupling=cfg["n_coupling"],
        hidden=cfg["hidden"], depth=cfg["depth"], lr=cfg["tf_lr"], verbose=True)
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
        skip_methods=skip_methods, dtype=dtype,
        extra_npz=dict(true_mus=np.asarray(target.mus), true_sigmas=np.asarray(target.sigmas),
                       true_weights=np.asarray(target.weights), true_alphas=np.asarray(target.alphas)))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    bm.add_run_args(parser, TARGET)
    args = parser.parse_args()
    out_dir = args.out_dir or os.path.join(common.benchmark_dir(TARGET), f"seed_{args.seed}")
    run_experiment(seed=args.seed, out_dir=out_dir, skip_methods=bm.parse_skip(args.skip_methods))


if __name__ == "__main__":
    main()
