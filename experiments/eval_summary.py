#!/usr/bin/env python
"""experiments/eval_summary.py

Methods x targets x metrics summary over the benchmark runs in
`experiments/results/benchmark/<target>/seed_<seed>/` (from `run_<target>.py`).

Methods: CAT and the baselines GMM (CAT's initial Laplace mixture b, sampled directly),
FAB, TemperFlow, APT, APT-DEO, ALPS, AFT (GAT is also scored, console only). Targets:
skewnormal, sur, iohmm, lda. Metrics per (target, method), mean +/- std over the seeds
where the method produced draws (superscript: that count, when fewer than all seeds):
  - ESS / n and Var(log w), w = pi / q, for the methods with an exact density of their
    draws (CAT, GMM, FAB, TemperFlow; "--" otherwise). Diagnostic only: every method is
    scored on its unweighted draws.
  - between-mode share of Var(log w) (methods with an exact density): 1 - sum_k p_k
    Var(log w | mode k) / Var(log w), p_k the share of draws in mode k -- the part of the
    importance-weight variance due to wrong mode masses rather than wrong mode shapes.
  - mode recovery rate: fraction of ALL seeds on which every mode has >= 1 (in-basin)
    draw; a failed run counts as not recovered.
  - mode TV: 0.5 * (sum_k |phat_k - w_k| + p_off) for skewnormal, iohmm and lda (not
    sur, whose mode weights are not known exactly).
  - weight TV (CAT and GMM only; on sur against the NUTS reference's basin weights): the
    same TV computed from the mixture weights themselves instead of the draws, free of
    sampling noise: CAT's calibrated weights vs b's Laplace weights, each discovered mode
    mapped to the true mode of its mean. This isolates CAT's weight calibration.
  - SW2 to the reference (folded into one basin on iohmm and lda, see `target_eval.py`),
    the same reference subset and projections for every method within a seed; also as
    median [IQR] over the seeds, since a few seeds with draws far in the tails can
    dominate the mean.
  - wall time (s) from summary.json (including the shared mode discovery).

Per-seed values are cached in `experiments/results/benchmark/eval_summary_<target>.json`
(reused unless --recompute or a seed directory changed after the cache); the LaTeX table
is printed and written to `experiments/figures/summary_table.tex`.

Usage:
  python experiments/eval_summary.py                       # all four targets
  python experiments/eval_summary.py --targets lda --recompute
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import jax

import common
from common import npz_key, seed_dirs
from target_eval import TargetEval, is_metrics

METHODS = ["CAT", "GMM", "FAB", "TemperFlow", "APT", "APT-DEO", "ALPS", "AFT"]   # table columns
SCORED = METHODS + ["GAT"]      # also scored (console only)
ABBREV = {"CAT": "CAT", "GMM": "GMM", "FAB": "FAB", "TemperFlow": "TF", "APT": "APT", "APT-DEO": "APT-DEO",
          "ALPS": "ALPS", "AFT": "AFT", "GAT": "GAT"}
TARGETS = ["sur", "skewnormal", "iohmm", "lda"]
TARGET_TITLE = {"sur": "SUR", "skewnormal": "Skew-normal mixture", "iohmm": "IOHMM", "lda": "LDA"}
CACHE_VERSION = 1          # bump when the per-seed metrics change
NUM_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}
METRICS = ("ess", "logw_var", "logw_between", "recovered", "mode_tv", "weight_tv", "sw2", "wall_time_s")


def _load_npz(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def _load_summary(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def weight_tv(te: TargetEval, weights, means) -> float:
    """TV between mixture weights (mapped onto the true modes by their means) and the
    true mode weights; mass on modes whose mean is off-mode counts as error."""
    lab = te.assign(np.asarray(means, np.float64))
    w = np.bincount(lab, weights=np.asarray(weights, np.float64), minlength=te.n_modes + 1)
    w = w / w.sum()
    return 0.5 * float(np.sum(np.abs(w[:-1] - te.true_weights)) + w[-1])


def score_seed(te: TargetEval, seed_dir: str, key) -> dict:
    """{method: {metric: value}} for the methods present in this seed directory."""
    npz = _load_npz(os.path.join(seed_dir, "samples.npz"))
    summ = _load_summary(os.path.join(seed_dir, "summary.json"))
    out = {}
    for m in SCORED:
        k = npz_key(m)
        if k not in npz:
            continue
        x = np.asarray(npz[k], np.float64)
        row = {"wall_time_s": summ.get(m, {}).get("wall_time_s"), "n": int(x.shape[0])}
        if f"logq_{k}" in npz:
            ok = np.all(np.isfinite(x), -1)
            lw = te.log_prob(x[ok]) - np.asarray(npz[f"logq_{k}"])[ok]
            row["ess"], row["logw_var"] = is_metrics(lw, np.zeros_like(lw))
        lab = te.assign(x)                  # one (sur: costly) assignment for every mode metric
        if "ess" in row:
            lo, fin = lab[ok], np.isfinite(lw)
            lo, lwf = lo[fin], lw[fin]
            within = sum(np.mean(lo == j) * np.var(lwf[lo == j]) for j in np.unique(lo) if np.sum(lo == j) > 1)
            row["logw_between"] = float(1.0 - within / np.var(lwf)) if np.var(lwf) > 0 else 0.0
        row["recovered"] = float(np.all(np.bincount(lab, minlength=te.n_modes + 1)[:te.n_modes] > 0))
        if te.true_weights is not None and te.name != "sur":
            ph = np.bincount(lab, minlength=te.n_modes + 1) / lab.size        # = te.mode_tv(x)
            row["mode_tv"] = 0.5 * float(np.sum(np.abs(ph[:-1] - te.true_weights)) + ph[-1])
        row["sw2"] = te.sw2(x, key)
        out[m] = row
    means = summ.get("GMM", {}).get("mode_means")
    if means is not None:
        for m, wkey in (("GMM", "mode_weights"), ("CAT", "final_weights")):
            w = summ.get(m, {}).get(wkey)
            if m in out and w is not None and len(w) == len(means):
                out[m]["weight_tv"] = weight_tv(te, w, means)
    return out


def compute(target: str, recompute: bool) -> tuple[dict, int]:
    rdir = common.benchmark_dir(target)
    cache_path = os.path.join(os.path.dirname(rdir), f"eval_summary_{target}.json")
    cache = {}
    if os.path.exists(cache_path) and not recompute:
        with open(cache_path) as f:
            cache = json.load(f)
    dirs = seed_dirs(rdir)
    te = None
    for i, sd in enumerate(dirs):
        name = os.path.basename(sd)
        mtime = max(os.path.getmtime(os.path.join(sd, f)) for f in ("samples.npz", "summary.json"))
        if name in cache and cache[name]["mtime"] >= mtime and cache[name].get("version", 1) >= CACHE_VERSION:
            continue
        if te is None:
            te = TargetEval.load(target, rdir)
        t0 = time.time()
        cache[name] = dict(mtime=mtime, version=CACHE_VERSION, scores=score_seed(te, sd, jax.random.PRNGKey(i)))
        print(f"[eval_summary/{target}] {name}: {sorted(cache[name]['scores'])} ({time.time() - t0:.0f}s)",
              flush=True)
        if i % 10 == 9:
            with open(cache_path, "w") as f:
                json.dump(cache, f)
    with open(cache_path, "w") as f:
        json.dump(cache, f)
    return {k: v["scores"] for k, v in cache.items() if k in {os.path.basename(d) for d in dirs}}, len(dirs)


def aggregate(per_seed: dict, n_seeds: int) -> dict:
    """{method: {metric: (mean, std, count)}}; recovered over all seeds (missing = 0)."""
    agg = {}
    for m in SCORED:
        rows = [s[m] for s in per_seed.values() if m in s]
        a = {}
        for met in METRICS:
            v = np.asarray([r[met] for r in rows if r.get(met) is not None], np.float64)
            if met == "recovered":
                a[met] = (float(v.sum() / n_seeds), 0.0, n_seeds) if rows else None
            else:
                a[met] = (float(np.nanmean(v)), float(np.nanstd(v)), int(v.size)) if v.size else None
            if met == "sw2" and v.size:   # median [IQR]: a few seeds far in the tails dominate the mean
                q25, q50, q75 = np.nanquantile(v, [0.25, 0.5, 0.75])
                a["sw2_med"] = (float(q50), (float(q25), float(q75)), int(v.size))
        a["n_ran"] = len(rows)
        agg[m] = a
    return agg


ROWS = [  # (metric, label, digits, better)
    ("ess", "ESS$/n$ $\\uparrow$", 3, max),
    ("logw_var", "Var$(\\log w)$ $\\downarrow$", 2, min),
    ("logw_between", "between-mode share", 2, None),
    ("recovered", "Recovery $\\uparrow$", 2, max),
    ("mode_tv", "Mode TV $\\downarrow$", 3, min),
    ("weight_tv", "Weight TV $\\downarrow$", 3, min),
    ("sw2", "SW2 $\\downarrow$", 3, min),
    ("sw2_med", "SW2 median [IQR] $\\downarrow$", 3, min),
    ("wall_time_s", "Time (s)", 0, None),
]


def latex(results: dict) -> str:
    n_seeds = max(n for _, n in results.values())
    lines = [r"\begin{table}[t]", r"\centering", r"\small",
             r"\caption{CAT and baselines on " + NUM_WORDS.get(len(results), str(len(results)))
             + r" targets, mean $\pm$ std over " + str(n_seeds) + r" seeds "
             r"(superscript: number of seeds a method completed, when fewer). "
             r"ESS$/n$ and Var$(\log w)$ of $w=\pi/q$ for the methods with an exact density (--: none); "
             r"between-mode share: the fraction of Var$(\log w)$ explained by which mode a draw is in (wrong mode masses). "
             r"Recovery: fraction of seeds with every mode visited (failed runs count as misses). "
             r"Mode TV: total variation between the empirical and true mode weights (off-mode draws count as error). "
             r"Weight TV: the same between a mixture's own weights and the true ones (CAT: calibrated $\hat w$; GMM: Laplace weights of $b$; for SUR the true weights are the NUTS reference's basin weights). "
             r"SW2: sliced 2-Wasserstein distance to the reference (IOHMM and LDA folded into one basin), "
             r"as mean $\pm$ std and as median [interquartile range]. "
             r"Time: wall clock including the shared mode discovery. All methods are scored on unweighted draws.}",
             r"\label{tab:summary}",
             r"\begin{tabular}{ll" + "c" * len(METHODS) + "}", r"\toprule",
             "Target & Metric & " + " & ".join(ABBREV[m] for m in METHODS) + r" \\"]
    for target, (agg, n_seeds) in results.items():
        lines.append(r"\midrule")
        first = True
        for met, label, dig, better in ROWS:
            vals = {m: agg[m][met] for m in METHODS if agg[m].get(met) is not None}
            if not vals:
                continue
            best = better(round(v[0], dig) for v in vals.values()) if better else None
            cells = []
            for m in METHODS:
                if m not in vals:
                    cells.append("--")
                    continue
                mean, std, cnt = vals[m]
                if met == "recovered":
                    s = f"{mean:.{dig}f}"
                elif met == "sw2_med":    # (median, (q25, q75), count)
                    s = f"{mean:.{dig}f} [{std[0]:.{dig}f}, {std[1]:.{dig}f}]"
                else:
                    s = f"{mean:.{dig}f} $\\pm$ {std:.{dig}f}"
                if met != "recovered" and cnt < n_seeds:
                    s += f"$^{{({cnt})}}$"
                if best is not None and round(mean, dig) == best:
                    s = f"\\textbf{{{s}}}"
                cells.append(s)
            name = TARGET_TITLE[target] if first else ""
            first = False
            lines.append(f"{name} & {label} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--targets", default=",".join(TARGETS))
    ap.add_argument("--recompute", action="store_true")
    ap.add_argument("--out", default=os.path.join(common.FIGURES_DIR, "summary_table.tex"))
    args = ap.parse_args()
    results = {}
    for t in args.targets.split(","):
        per_seed, n_seeds = compute(t, args.recompute)
        results[t] = (aggregate(per_seed, n_seeds), n_seeds)
        print(f"\n[{t}] n_seeds={n_seeds}")
        for m in SCORED:
            a = results[t][0][m]
            print(f"  {ABBREV[m]:>8} ran {a['n_ran']:3d} |" + " | ".join(
                f"{met} {a[met][0]:.4g}±{a[met][1]:.3g}" for met in METRICS if a.get(met)))
    tex = latex(results)
    print(tex)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        f.write(tex + "\n")


if __name__ == "__main__":
    main()
