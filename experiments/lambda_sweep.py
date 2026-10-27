#!/usr/bin/env python
"""experiments/lambda_sweep.py

Sensitivity of CAT to the velocity penalty lambda (`velocity_l2_weight`, 1e-3 in
`transport_stage.CAT_BASE`): the residual loss plus lambda * E|v|^2.

  run: CAT only, exactly as in the benchmark (the target's own
      `run_<target>.run_experiment`: same mode discovery and `CAT_CONFIG`), with lambda
      overridden and every other method skipped. One output directory per (lambda, seed),
      `experiments/results/lambda_sweep/<target>/lam_<lambda>/seed_<seed>/`, with the
      standard benchmark outputs (`benchmark.py`); runs whose summary.json exists are
      skipped. The default seeds 4026-4030 are disjoint from the benchmark's 2026-2125.
  table: per target and lambda, mean +/- std over seeds of CAT's final ESS / n (pi / q
      on its final mixture draws, `final_ess` in summary.json); also printed: the mean
      kinetic energy E_q |v|^2 of the selected validation round and the transport cost
      relative to lambda = 1e-3 on the same seed. The LaTeX table is printed and written
      to `experiments/figures/lambda_table.tex`.

Usage:
  python experiments/lambda_sweep.py run --target sur                       # all lambdas x seeds
  python experiments/lambda_sweep.py run --target lda --lambdas 1e-3 --seeds 4026
  python experiments/lambda_sweep.py table
"""
from __future__ import annotations

import argparse
import glob
import importlib
import json
import os

import numpy as np

import common

LAMBDAS = ("0", "1e-4", "3e-4", "1e-3", "3e-3", "1e-2", "1e-1")
SEEDS = tuple(range(4026, 4031))
DEFAULT_LAMBDA = 1e-3
TARGETS = ["skewnormal", "sur", "iohmm", "lda"]      # table columns
TARGET_TITLE = {"skewnormal": "Skew-normal mixture", "sur": "SUR", "iohmm": "IOHMM", "lda": "LDA"}
SWEEP_DIR = os.path.join(common.RESULTS_DIR, "lambda_sweep")


def run_dir(target: str, lam: str, seed: int) -> str:
    return os.path.join(SWEEP_DIR, target, f"lam_{lam}", f"seed_{seed}")


# ---------------- run ----------------
def run(target: str, lambdas, seeds):
    import jax
    jax.config.update("jax_enable_x64", True)
    import benchmark
    import transport_stage as ts

    run_mod = importlib.import_module(f"run_{target}")
    skip = frozenset(benchmark.METHOD_LABELS) - {"CAT"}
    base = dict(ts.CAT_CONFIG[target])
    for lam in lambdas:
        ts.CAT_CONFIG[target] = dict(base, velocity_l2_weight=float(lam))
        for seed in seeds:
            out_dir = run_dir(target, lam, seed)
            if os.path.exists(os.path.join(out_dir, "summary.json")):
                print(f"[lambda_sweep] {target} lambda={lam} seed={seed}: done, skipped")
                continue
            print(f"[lambda_sweep] {target} lambda={lam} seed={seed} -> {out_dir}", flush=True)
            run_mod.run_experiment(seed=seed, out_dir=out_dir, skip_methods=skip)
    ts.CAT_CONFIG[target] = base


# ---------------- table ----------------
def load(target: str):
    """{lambda: {seed: dict(ess, cost, ke)}} of the finished runs of `target`."""
    rows = {}
    for p in glob.glob(os.path.join(SWEEP_DIR, target, "lam_*", "seed_*", "summary.json")):
        parts = p.split(os.sep)
        lam, seed = float(parts[-3][len("lam_"):]), int(parts[-2][len("seed_"):])
        with open(p) as f:
            cat = json.load(f).get("CAT")
        if cat is None or "final_ess" not in cat:
            continue
        ke = cat.get("val_kinetic_energy")
        rows.setdefault(lam, {})[seed] = dict(ess=cat["final_ess"], cost=cat["weighted_cost_1_d"],
                                              ke=float(np.mean(ke)) if ke else np.nan)
    return rows


def mean_std(a):
    a = np.asarray(a, float)
    return a.mean(), a.std(ddof=1) if a.size > 1 else 0.0


def lambda_tex(lam: float) -> str:
    if lam == 0:
        return "0"
    e = int(np.floor(np.log10(lam) + 1e-9))
    m = lam / 10.0 ** e
    return f"$10^{{{e}}}$" if np.isclose(m, 1) else f"${m:g} \\times 10^{{{e}}}$"


def latex(results: dict) -> str:
    targets = [t for t in TARGETS if t in results]
    lams = sorted({lam for t in targets for lam in results[t]})
    n_seeds = max(len(s) for t in targets for s in results[t].values())
    lines = [r"\begin{table}[t]", r"\centering", r"\small",
             r"\caption{Sensitivity of CAT to the regularization coefficient $\lambda$. Entries report "
             r"mean $\pm$ std of relative ESS over " + str(n_seeds) + r" independent runs. "
             r"The default value is shown in bold.}",
             r"\label{tab:lambda}",
             r"\begin{tabular}{l" + "c" * len(targets) + "}", r"\toprule",
             r"$\lambda$ & " + " & ".join(TARGET_TITLE[t] for t in targets) + r" \\", r"\midrule"]
    for lam in lams:
        cells = []
        for t in targets:
            seeds = results[t].get(lam)
            if not seeds:
                cells.append("--")
                continue
            m, s = mean_std([r["ess"] for r in seeds.values()])
            cell = f"{m:.3f} $\\pm$ {s:.3f}" + (f"$^{{({len(seeds)})}}$" if len(seeds) < n_seeds else "")
            cells.append(f"\\textbf{{{cell}}}" if np.isclose(lam, DEFAULT_LAMBDA) else cell)
        name = lambda_tex(lam)
        if np.isclose(lam, DEFAULT_LAMBDA):
            name = rf"$\mathbf{{{name.strip('$')}}}$"
        lines.append(f"{name} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def table(targets, out: str):
    results = {t: load(t) for t in targets}
    results = {t: r for t, r in results.items() if r}
    if not results:
        raise FileNotFoundError(f"no finished runs under {SWEEP_DIR}")
    for t, by_lam in results.items():
        ref = next((s for lam, s in by_lam.items() if np.isclose(lam, DEFAULT_LAMBDA)), {})
        print(f"\n[{t}]\n{'lambda':>8} {'n':>2} {'ESS/n':>15} {'E|v|^2':>8} {'cost/ref':>15}")
        for lam in sorted(by_lam):
            seeds = by_lam[lam]
            ess = mean_std([r["ess"] for r in seeds.values()])
            ke = np.mean([r["ke"] for r in seeds.values()])
            rel = [r["cost"] / ref[s]["cost"] for s, r in seeds.items() if s in ref]
            rel_s = "{:.3f}+/-{:.3f}".format(*mean_std(rel)) if rel else "-"
            print(f"{lam:>8g} {len(seeds):>2} {ess[0]:>9.3f}+/-{ess[1]:.3f} {ke:>8.3g} {rel_s:>15}")
    tex = latex(results)
    print(tex)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        f.write(tex + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run", help="CAT runs over lambdas x seeds on one target")
    p.add_argument("--target", required=True, choices=common.TARGETS)
    p.add_argument("--lambdas", default=",".join(LAMBDAS))
    p.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    p = sub.add_parser("table", help="ESS / n table over the finished runs")
    p.add_argument("--targets", default=",".join(TARGETS))
    p.add_argument("--out", default=os.path.join(common.FIGURES_DIR, "lambda_table.tex"))
    args = ap.parse_args()
    if args.cmd == "run":
        run(args.target, args.lambdas.split(","), [int(s) for s in args.seeds.split(",")])
    else:
        table(args.targets.split(","), args.out)


if __name__ == "__main__":
    main()
