#!/usr/bin/env python
"""experiments/plot_skewnormal.py

First-coordinate projection on the skew-normal mixture: a grid of histograms of the
marginal x_1 of each method's final draws (one panel per method, from one benchmark run of
`run_skewnormal.py`) against the oracle density of x_1 (a fine histogram of 200000 exact
target draws). The skewed mode is a product of iid skew-normals, so a single coordinate is
its most skewed direction, and the two modes stay well separated in x_1; the panels show
how well each method captures both the skewness and the mode weights.

Usage:
  python experiments/plot_skewnormal.py                       # seed 2026, 8 methods
  python experiments/plot_skewnormal.py --seed 2027 --methods CAT,GMM,FAB
Output: experiments/figures/skewnormal/x1_hist_grid.{png,pdf}
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import common
from common import RC_PRINT, TITLE_COLOR, grid_shape, npz_key, save_figure

METHODS = ["CAT", "GMM", "FAB", "TemperFlow", "APT", "APT-DEO", "ALPS", "AFT"]
ABBREV = {"TemperFlow": "TF"}
SAMPLE_COLOR = "#006199"
ORACLE_COLOR = "#9E2B25"


def oracle_draws(n: int = 200_000) -> np.ndarray:
    """Exact draws from the skew-normal mixture."""
    import jax
    jax.config.update("jax_enable_x64", True)
    from run_skewnormal import build_target
    return np.asarray(build_target().sample(jax.random.PRNGKey(12345), n)[0])


def plot_x1_hist_grid(samples, methods, oracle, path, n_bins=180, width=6.3):
    """Histograms of the marginal u = x_1 per method vs the oracle density of u, drawn at
    the printed width `width` (inches)."""
    proj = lambda x: np.asarray(x)[:, 0]
    u_ref = proj(oracle)
    lo, hi = np.quantile(u_ref, [0.001, 0.999])
    pad = 0.05 * (hi - lo)
    edges = np.linspace(lo - pad, hi + pad, n_bins + 1)
    fine = np.linspace(edges[0], edges[-1], 4 * n_bins + 1)
    ref_dens, _ = np.histogram(u_ref, bins=fine, density=True)
    ref_dens = ref_dens * np.mean((u_ref >= fine[0]) & (u_ref <= fine[-1]))
    ref_mid = 0.5 * (fine[1:] + fine[:-1])
    ymax = 1.15 * ref_dens.max()

    n_rows, n_cols = grid_shape(len(methods))
    with plt.rc_context(RC_PRINT):
        # fixed margins left/right/bottom/top and gaps between panels / rows, in inches;
        # each panel w x w/phi (golden ratio)
        gap, vgap, L, R, B, T = 0.12, 0.3, 0.5, 0.05, 0.38, 0.5
        w = (width - L - R - (n_cols - 1) * gap) / n_cols
        h = w / ((1 + 5 ** 0.5) / 2)
        W, H = width, B + n_rows * h + (n_rows - 1) * vgap + T
        fig = plt.figure(figsize=(W, H))
        xc = (L + (n_cols * w + (n_cols - 1) * gap) / 2) / W     # centre of the panel grid
        axes = np.empty((n_rows, n_cols), dtype=object)
        for row in range(n_rows):
            for col in range(n_cols):
                y = B + (n_rows - 1 - row) * (h + vgap)
                axes[row, col] = fig.add_axes([(L + col * (w + gap)) / W, y / H, w / W, h / H])
        for ax in axes.flat[len(methods):]:
            ax.set_visible(False)
        for i, label in enumerate(methods):
            row, col = divmod(i, n_cols)
            ax = axes[row, col]
            if col > 0:
                ax.tick_params(labelleft=False)
            if row < n_rows - 1 and i + n_cols < len(methods):
                ax.tick_params(labelbottom=False)
            x = samples.get(label)
            if x is None:
                ax.text(0.5, 0.5, "no samples", transform=ax.transAxes, ha="center", va="center",
                        color=TITLE_COLOR)
            else:
                u = proj(x[np.all(np.isfinite(x), -1)])
                # density over all draws: mass outside the plotted range is not renormalized away
                cnt, _ = np.histogram(u, bins=edges)
                ax.bar(edges[:-1], cnt / (u.size * np.diff(edges)), width=np.diff(edges), align="edge",
                       color=SAMPLE_COLOR, alpha=0.55, linewidth=0)
            ax.plot(ref_mid, ref_dens, color=ORACLE_COLOR, lw=0.9)
            ax.set_title(ABBREV.get(label, label), fontsize=9, fontweight="bold", color=TITLE_COLOR, pad=4)
            ax.set_xlim(edges[0], edges[-1])
            ax.set_ylim(0, ymax)
        fig.supxlabel(r"$x_1$", x=xc, y=0.02 / H, va="bottom", fontsize=8)
        fig.text(0.04 / W, (B + (H - B - T) / 2) / H, "Density", rotation=90, ha="left", va="center",
                 fontsize=8)
        fig.legend(handles=[Patch(color=SAMPLE_COLOR, alpha=0.55, label="Samples"),
                            Line2D([0], [0], color=ORACLE_COLOR, lw=0.9, label="Oracle")],
                   loc="upper center", bbox_to_anchor=(xc, 1.0), ncol=2, frameon=False, fontsize=8,
                   handlelength=1.6, handletextpad=0.4, columnspacing=1.2)
        save_figure(fig, path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--figures-dir", default=os.path.join(common.FIGURES_DIR, "skewnormal"))
    args = ap.parse_args()
    methods = args.methods.split(",")
    with np.load(os.path.join(common.benchmark_dir("skewnormal"), f"seed_{args.seed}", "samples.npz")) as z:
        samples = {m: z[npz_key(m)] for m in methods if npz_key(m) in z.files}
    plot_x1_hist_grid(samples, methods, oracle_draws(), os.path.join(args.figures_dir, "x1_hist_grid"))


if __name__ == "__main__":
    main()
