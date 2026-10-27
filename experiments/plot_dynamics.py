#!/usr/bin/env python
"""experiments/plot_dynamics.py

Figures from `run_dynamics.py` (`experiments/results/dynamics/<target>/seed_<seed>/`),
written to `experiments/figures/dynamics/`.

  training --targets iohmm,lda: CAT's training dynamics, one row per target: training
      loss per optimizer step, and ESS / n and ELBO = E_q[log pi~ - log q] of the current
      model after every refresh, on one iteration axis (optimizer steps). The open marker
      at iteration 0 is b itself (the GMM baseline's draws of the same seed in the
      benchmark results).
  wallclock --target lda: ESS / n, SW2 and ELBO of CAT, GAT and the CNF vs transport wall
      time (lda: minus JAX compile time, with an inset of the first 250 s; b at t = 0
      joined to every method's first refresh).

Usage:
  python experiments/plot_dynamics.py training --targets iohmm,lda
  python experiments/plot_dynamics.py wallclock --target lda
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch

import common
from common import RC_PRINT, TITLE_COLOR, save_figure

CAT_COLOR = "#006199"
SPEED_COLORS = {"CAT": "#006199", "GAT": "#E0301E", "CNF": "#2E8B57"}
SPEED_MARKERS = {"CAT": "o", "GAT": "s", "CNF": "^"}
TARGET_LABELS = {"sur": "SUR", "lda": "LDA", "iohmm": "IOHMM", "skewnormal": "Skew-normal"}
# wallclock insets (axes fractions x, y, w, h): the empty region right of CAT/GAT
INSET_BOUNDS = {"ess": [0.4, 0.13, 0.57, 0.43], "sw2": [0.4, 0.4, 0.57, 0.5], "elbo": [0.4, 0.2, 0.57, 0.46]}
# per target: wallclock options (lda: the CNF runs ~8x longer, so CAT/GAT are zoomed in)
WALLCLOCK_KW = {"lda": dict(inset=250, minus_compile=True)}


def dynamics_dir(target, seed):
    return os.path.join(common.RESULTS_DIR, "dynamics", target, f"seed_{seed}")


def _elbo(r, log_Z_ref):
    """ELBO = E_q[log pi~ - log q] of one record."""
    return -r["kl_offset"] if "kl_offset" in r else log_Z_ref - r["kl"]


def init_point(dyn, d):
    """Scores of b (the GMM baseline: the common untrained start of CAT, GAT and the CNF)
    at iteration 0 / time 0, on the same log Z_ref and SW2 key as the curves; computed
    once from the GMM draws of the benchmark run of the same seed and cached in
    dynamics.json."""
    init = dyn.get("init")
    if init is not None and "sw2" in init:
        return init
    t, seed = dyn["target"], dyn["seed"]
    f = os.path.join(common.benchmark_dir(t), f"seed_{seed}", "samples.npz")
    z = np.load(f) if os.path.exists(f) else {}
    if "GMM" not in z or "logq_GMM" not in z:
        print(f"[plot_dynamics] no GMM draws in {f}: no iteration-0 point for {t}")
        return None
    import jax
    jax.config.update("jax_enable_x64", True)
    from target_eval import TargetEval, is_metrics
    te = TargetEval.load(t)
    x, lq = np.asarray(z["GMM"]), np.asarray(z["logq_GMM"])
    lp = te.log_prob(x)
    ok = np.isfinite(lp) & np.isfinite(lq)
    ess, lwv = is_metrics(lp[ok], lq[ok])
    r = lq[ok] - lp[ok]
    dyn["init"] = dict(iteration=0, ess=ess, logw_var=lwv, kl_offset=float(r.mean()),
                       kl=float(r.mean() + dyn["log_Z_ref"]),
                       kl_se=float(r.std() / np.sqrt(ok.sum())), mode_weights=te.mode_weights(x).tolist(),
                       sw2=te.sw2(x[ok], jax.random.PRNGKey(777), n_sw=2048),   # as run_dynamics' callbacks
                       source=os.path.relpath(f, common.EXPERIMENTS_DIR))
    with open(os.path.join(d, "dynamics.json"), "w") as fh:
        json.dump(dyn, fh, indent=1, default=float)
    return dyn["init"]


def _segment_log_mean(loss, breaks, w=25):
    """Running geometric mean of the per-step loss over a centred window of `w` steps,
    clipped at the refresh steps `breaks` so the jumps at a refresh are not smeared."""
    ll = np.log(np.maximum(loss, 1e-300))
    out = np.empty_like(ll)
    edges = [0] + [int(b) for b in breaks if 0 < b < ll.size] + [ll.size]
    for a, b in zip(edges[:-1], edges[1:]):
        c = np.concatenate([[0.0], np.cumsum(ll[a:b])])
        i = np.arange(b - a)
        lo, hi = np.maximum(i - w // 2, 0), np.minimum(i + w // 2 + 1, b - a)
        out[a:b] = (c[hi] - c[lo]) / (hi - lo)
    return np.exp(out)


def _training_row(axes, dyn, init=None, c=CAT_COLOR):
    """One row (CAT, one run) on three axes: loss per step, ESS and ELBO per refresh, one
    iteration axis; `init` (if given) is b at iteration 0, before any training (open marker)."""
    loss = np.asarray(dyn["train_loss"]["CAT"])
    recs = dyn["curves"]["CAT"]
    if init is not None:
        recs = [init] + recs
    it = np.asarray([r["iteration"] for r in recs])
    elbo = np.asarray([_elbo(r, dyn["log_Z_ref"]) for r in recs])
    trained = elbo[1:] if init is not None else elbo
    ax = axes[0]
    steps = np.arange(1, loss.size + 1)
    ax.plot(steps, loss, color=c, lw=0.35, alpha=0.3)
    ax.plot(steps, _segment_log_mean(loss, it[it > 0]), color=c, lw=0.9)
    ax.set_yscale("log")
    ax = axes[1]
    ax.plot(it, [r["ess"] for r in recs], "o-", color=c, ms=2.2, lw=0.9)
    ax.set_ylim(0, 1.02)
    ax = axes[2]
    se = np.asarray([r.get("kl_se", 0.0) for r in recs])
    far = init is not None and trained.max() - elbo[0] > 20 * np.ptp(trained)
    k = 1 if far else 0          # b far below: drawn separately at the bottom edge
    ax.plot(it[k:], elbo[k:], "o-", color=c, ms=2.2, lw=0.9)
    ax.fill_between(it[k:], elbo[k:] - se[k:], elbo[k:] + se[k:], color=c, alpha=0.2, lw=0)
    ax.ticklabel_format(axis="y", useOffset=False)
    b_kw = dict(ms=4.2, mfc="white", mec=c, mew=0.9, zorder=5, clip_on=False)
    if init is not None:
        axes[1].plot(0, init["ess"], "o", **b_kw)
        lo, hi = trained.min(), trained.max()
        if far:
            # keep the axis on the trained range, mark b at the bottom edge with its
            # value and join it to the first refresh with a dashed segment
            pad = 0.1 * (hi - lo)
            y0 = lo - 2 * pad
            ax.set_ylim(y0, hi + pad)
            ax.plot(it[:2], [y0, elbo[1]], "--", color=c, lw=0.7, clip_on=False)
            ax.plot(0, y0, "v", **b_kw)
            ax.annotate(f"$b$: {elbo[0]:.2f}", (0, y0), xytext=(12, 1),
                        textcoords="offset points", fontsize=6.5, va="bottom", ha="left", color=c)
        else:
            ax.plot(0, elbo[0], "o", **b_kw)
    for ax in axes:
        ax.set_xlim(-0.02 * loss.size, loss.size)


def plot_training_curves(rows, path, width=6.3):
    """One row per (label, dyn, init) in `rows`; label None: no row label. Drawn at the
    printed width (inches): fonts are 7 pt (ticks), 8 pt (legend, x label), 9 pt (titles)."""
    c = CAT_COLOR
    n = len(rows)
    labelled = any(r[0] for r in rows)
    with plt.rc_context(RC_PRINT):
        # fixed layout in inches: margins left/right/bottom/top, gaps between panels / rows;
        # the three panels share what is left of `width`, each w x w/phi (golden ratio)
        gap, vgap, L, R, B, T = 0.55, 0.12, 0.33 + 0.2 * labelled, 0.05, 0.38, 0.5
        w = (width - L - R - 2 * gap) / 3
        h = w / ((1 + 5 ** 0.5) / 2)
        W, H = width, B + n * h + (n - 1) * vgap + T
        fig = plt.figure(figsize=(W, H))
        xc = (L + 1.5 * w + gap) / W                     # centre of the panel row
        for r, (label, dyn, init) in enumerate(rows):
            y = B + (n - 1 - r) * (h + vgap)
            axes = [fig.add_axes([(L + i * (w + gap)) / W, y / H, w / W, h / H]) for i in range(3)]
            _training_row(axes, dyn, init, c)
            if r == 0:
                for ax, title in zip(axes, ("Training loss", "ESS/$N$", "ELBO")):
                    ax.set_title(title, fontsize=9, color="k", pad=4)
            if r < n - 1:
                for ax in axes:
                    ax.tick_params(labelbottom=False)
            if label:
                fig.text(0.04 / W, (y + h / 2) / H, label, rotation=90, ha="left", va="center",
                         fontsize=9, fontweight="bold", color=TITLE_COLOR)
        fig.supxlabel("Iteration (optimizer steps)", x=xc, y=0.02 / H, va="bottom", fontsize=8)
        hs = [Line2D([], [], color=c, lw=0.35, alpha=0.5), Line2D([], [], color=c, lw=0.9),
              Line2D([], [], color=c, lw=0.9, marker="o", ms=2.8)]
        lab = ["Loss per step", "Running mean (25 steps)", "After each refresh"]
        if any(r[2] is not None for r in rows):
            hs.append(Line2D([], [], ls="", marker="o", ms=4.5, mfc="white", mec=c, mew=0.9))
            lab.append("$b$ (untrained)")
        fig.legend(hs, lab, loc="upper center", bbox_to_anchor=(xc, 1.0), ncol=len(hs), frameon=False,
                   fontsize=8, handlelength=1.6, handletextpad=0.4, columnspacing=1.2)
        save_figure(fig, path)


def plot_wallclock(dyn, path, init=None, width=6.3, inset=None, minus_compile=False):
    """ESS / n, SW2 and ELBO vs transport wall time, in the layout of plot_training_curves.
    `init` (if given): the common base b at t = 0 (open black marker). If the run recorded
    `train_start` (the time of the first training step: setup and JIT compilation before
    it), each method stays at b (solid, flat, ending in a tick) until then, and a dashed
    segment joins it to the first refresh. `inset` (seconds): every panel gets an inset of
    its curves over [0, inset], placed in `INSET_BOUNDS`. `minus_compile`: time is
    transport wall time minus the JAX compile time so far; b sits at t = 0 and is joined
    to every method's first refresh by a solid segment (no flat / dashed segments)."""
    x0 = 0.0
    with plt.rc_context(RC_PRINT):
        gap, L, R, B, T = 0.55, 0.33, 0.05, 0.38, 0.5     # as plot_training_curves
        w = (width - L - R - 2 * gap) / 3
        h = w / ((1 + 5 ** 0.5) / 2)
        W, H = width, B + h + T
        fig = plt.figure(figsize=(W, H))
        axes = [fig.add_axes([(L + i * (w + gap)) / W, B / H, w / W, h / H]) for i in range(3)]
        xc = (L + 1.5 * w + gap) / W
        keys = ("ess", "sw2", "elbo")
        # each panel: the main axes, plus its inset (drawn with the same calls)
        panels = [[ax] + ([ax.inset_axes(INSET_BOUNDS[k])] if inset else []) for ax, k in zip(axes, keys)]
        val = lambda r, k: _elbo(r, dyn["log_Z_ref"]) if k == "elbo" else r[k]
        ts = {m: (dyn["curves"][m][0].get("train_start") or 0.0) for m in dyn["methods"] if dyn["curves"].get(m)}
        hs, lab = [], []
        for m in dyn["methods"]:
            recs = dyn["curves"].get(m, [])
            if not recs:
                continue
            c, mk = SPEED_COLORS[m], SPEED_MARKERS[m]
            t = np.asarray([r["elapsed"] - (r.get("compile_s") or 0.0) * minus_compile for r in recs])
            if minus_compile and init is not None:
                # b at t = 0 heads every curve, joined to the first refresh by a solid line
                recs, t = [init] + recs, np.concatenate([[0.0], t])
            for p_, key in zip(panels[:2], keys[:2]):
                for ax in p_:
                    ax.plot(t, [r[key] for r in recs], "-", marker=mk, ms=2.4, lw=0.9, color=c)
            elbo = np.asarray([_elbo(r, dyn["log_Z_ref"]) for r in recs])
            se = np.asarray([r.get("kl_se", 0.0) for r in recs])
            for ax in panels[2]:
                ax.plot(t, elbo, "-", marker=mk, ms=2.4, lw=0.9, color=c)
                ax.fill_between(t, elbo - se, elbo + se, color=c, alpha=0.2, lw=0)
            hs.append(Line2D([], [], color=c, lw=0.9, marker=mk, ms=3.2))
            lab.append("RKL-CNF" if m == "CNF" else m)
            if init is None or minus_compile:
                continue
            # the earliest-starting method's flat segment on top, so every end shows
            z = 3 + sorted(ts.values(), reverse=True).index(ts[m]) * 0.1
            for p_, key in zip(panels, keys):
                y0 = val(init, key)
                for ax in p_:
                    if ts[m] > 0:
                        ax.plot([x0, ts[m]], [y0, y0], "-", lw=0.9, color=c, zorder=z)
                        ax.plot(ts[m], y0, "|", ms=5, mew=0.9, color=c, zorder=z)
                    ax.plot([max(ts[m], x0), t[0]], [y0, val(recs[0], key)], "--", lw=0.7, color=c)
        for ax in panels[2]:
            ax.ticklabel_format(axis="y", useOffset=False)
        if init is not None:
            b_kw = dict(ms=4.2, mfc="white", mec="k", mew=0.9, zorder=5, clip_on=False, ls="none")
            for p_, key in zip(panels, keys):
                p_[0].plot(x0, val(init, key), "o", **b_kw)
                p_[0].set_xlim(left=x0)
                if inset:
                    p_[1].plot(0, val(init, key), "o", **{**b_kw, "ms": 3.2, "clip_on": key == "elbo"})
            hs.append(Line2D([], [], ls="", marker="o", ms=4.5, mfc="white", mec="k", mew=0.9))
            lab.append("$b$ (untrained)")
        for ax, title in zip(axes, ("ESS/$N$", "SW2", "ELBO")):
            ax.set_title(title, fontsize=9, color="k", pad=4)
        axes[0].set_ylim(0, 1.02)
        if inset:
            for k, (ax, ins) in enumerate(panels):
                # y range of the trained points inside the window, 10% padding; for the
                # ELBO b is far below, so it is left out (it shows on the main panel)
                ys = np.concatenate([np.asarray(ln.get_ydata(), float)[
                    (np.asarray(ln.get_xdata(), float) <= inset) & (np.asarray(ln.get_xdata(), float) > 0)]
                                     for ln in ins.get_lines() if ln.get_linestyle() == "-"
                                     and ln.get_marker() in SPEED_MARKERS.values()])
                if init is not None and keys[k] != "elbo":
                    ys = np.append(ys, val(init, keys[k]))
                lo, hi = ys.min(), ys.max()
                ins.set_xlim(0, inset)
                ins.set_ylim(*((-0.03, 1.05) if k == 0 else (lo - 0.1 * (hi - lo), hi + 0.1 * (hi - lo))))
                ins.yaxis.set_major_locator(plt.FixedLocator([0, 0.5, 1]) if k == 0
                                            else plt.MaxNLocator(2, integer=keys[k] == "elbo"))
                ins.tick_params(labelsize=6, pad=1, length=1.5)
                ins.grid(False)
                ind = ax.indicate_inset_zoom(ins, edgecolor="0.4", lw=0.5)
                if not any(cp.get_visible() for cp in ind.connectors):
                    # the zoom box spans the full panel height (ESS), where matplotlib draws no
                    # connectors: join its right edge to the inset's left edge instead
                    y0, y1 = ins.get_ylim()
                    for ya, yi in ((y0, 0), (y1, 1)):
                        fig.add_artist(ConnectionPatch((inset, ya), (0, yi), coordsA=ax.transData,
                                                       coordsB=ins.transAxes, color="0.4", lw=0.5))
        fig.supxlabel("Wall-clock time (s)", x=xc, y=0.02 / H, va="bottom", fontsize=8)
        fig.legend(hs, lab, loc="upper center", bbox_to_anchor=(xc, 1.0), ncol=len(hs), frameon=False,
                   fontsize=8, handlelength=1.6, handletextpad=0.4, columnspacing=1.2)
        save_figure(fig, path)


def _load(target, seed):
    d = dynamics_dir(target, seed)
    with open(os.path.join(d, "dynamics.json")) as f:
        dyn = json.load(f)
    return dyn, init_point(dyn, d)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="figure", required=True)
    p = sub.add_parser("training", help="CAT's training curves, one row per target")
    p.add_argument("--targets", default="iohmm,lda")
    p = sub.add_parser("wallclock", help="CAT, GAT and the CNF against wall-clock time")
    p.add_argument("--target", default="lda", choices=common.TARGETS)
    for p in sub.choices.values():
        p.add_argument("--seed", type=int, default=2026)
        p.add_argument("--figures-dir", default=os.path.join(common.FIGURES_DIR, "dynamics"))
    args = ap.parse_args()
    if args.figure == "training":
        targets = args.targets.split(",")
        rows = [(TARGET_LABELS[t], *_load(t, args.seed)) for t in targets]
        plot_training_curves(rows, os.path.join(args.figures_dir, "training_curves_" + "_".join(targets)))
    else:
        dyn, init = _load(args.target, args.seed)
        plot_wallclock(dyn, os.path.join(args.figures_dir, f"wallclock_{args.target}"), init,
                       **WALLCLOCK_KW.get(args.target, {}))


if __name__ == "__main__":
    main()
