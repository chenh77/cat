#!/usr/bin/env python
"""experiments/skewt_paths.py

Transport paths of CAT and GAT on the d = 10 two-mode skew-t mixture
(`src.targets.SkewTMixture`, true weights 0.8 / 0.2, Laplace mixture b exactly 50 / 50),
at separations gamma = 1 and 4.

For each (method, gamma), one fit (seed 0; CAT's settings of the skew-normal benchmark;
GAT gets CAT's J = 2 fields folded into one: 2x parameters, 2x batch, 2x trajectories)
and fresh trajectories projected onto the separation axis z_1 = x . R e_1:
  - empirical tracks: the same particles RK4-integrated through the learned field on a
    101-point beta grid;
  - oracle tracks: quantile maps of the z_1 marginal along each method's reference path,
    estimated by importance weighting exact draws (defensive mixture proposals): for GAT the
    global path b^{1-beta} pi^beta, for CAT the component paths a_j b_j e^{beta h},
    h = log pi - log b;
  - the cumulative length L(beta) = int_0^beta s(beta') dbeta' of the path, with
    s(beta)^2 = sum_j w_j E|v_j(beta, X_beta)|^2 (GAT: one field, w = 1; CAT: per component,
    mixed by the calibrated weights).

Usage:
  python experiments/skewt_paths.py --task 0      # 0..3: (method, gamma) in TASKS
  python experiments/skewt_paths.py --plot
Output: experiments/results/skewt_paths/{traj,speed}_{method}_d10_s{gamma}.npz,
experiments/figures/skewt_paths/paths_skewt_d10.{png,pdf}
"""
from __future__ import annotations

import argparse
import itertools
import os
import time

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jr

import common
from common import RC_PRINT, TITLE_COLOR, save_figure
from src.samplers.cat import cat
from src.samplers.gat import field_param_count, gat_fit
from src.targets import SkewTMixture

D = 10
METHODS = ["CAT", "GAT"]
SEPARATIONS = [1.0, 4.0]
TASKS = list(itertools.product(METHODS, SEPARATIONS))
SEED = 0
N_PATH = 8192
N_IS = 200_000
LEVELS = (np.arange(24) + 0.5) / 24          # oracle quantile levels
OUT_DIR = os.path.join(common.RESULTS_DIR, "skewt_paths")
FIG_DIR = os.path.join(common.FIGURES_DIR, "skewt_paths")
FINE_BETAS = np.linspace(0, 1, 101)
N_SPEED = 2048
SPEED_BETAS = np.linspace(0, 1, 101)

# CAT's settings on the skew-normal benchmark (`transport_stage.CAT_CONFIG["skewnormal"]`
# with n_train / batch size per mode fixed); GAT folds CAT's J fields into one.
BASE_KW = dict(n_refresh=10, train_steps_per_refresh=100, n_validation=256, validate_every=5,
               select_best=True, lr=1e-3, grad_clip=5.0, velocity_l2_weight=1e-3, replay_refreshes=20,
               center_h=True, hidden=64, depth=3, activation="silu", spline_knots=8, rk4_substeps=2,
               adaptive_target_ess=0.7, adaptive_delta_max=0.3, particle_chunk=2048)
N_TRAIN_PER_MODE, BATCH_PER_MODE = 1024, 256


def fit(method, key, logpi, logpi_single, d, modes, n):
    kw = dict(BASE_KW, n=n)
    J = len(modes)
    t0 = time.time()
    if method == "GAT":
        kw.update(n_train=J * N_TRAIN_PER_MODE, batch_size=J * BATCH_PER_MODE,
                  target_params=J * field_param_count(d, 64, depth=3, spline_knots=8))
        res = gat_fit(key, logpi, logpi_single, d, modes=modes, **kw)
    else:
        kw.update(n_train=N_TRAIN_PER_MODE, batch_size=BATCH_PER_MODE)
        res = cat(key=key, logpi=logpi, logpi_single=logpi_single, modes=modes, **kw)
    return res, time.time() - t0


def b_draws(modes, n, seed):
    rng = np.random.default_rng(seed)
    w = np.array([m.w for m in modes]); w /= w.sum()
    comp = rng.choice(len(modes), size=n, p=w)
    x = np.zeros((n, modes[0].mu.shape[0]))
    for j, m in enumerate(modes):
        L = np.linalg.cholesky(np.asarray(m.Sigma))
        k = comp == j
        x[k] = np.asarray(m.mu) + rng.standard_normal((k.sum(), L.shape[0])) @ L.T
    return x


def log_bj(m, x):
    L = np.linalg.cholesky(np.asarray(m.Sigma))
    y = np.linalg.solve(L, (x - np.asarray(m.mu)).T).T
    return -0.5 * np.sum(y * y, -1) - np.sum(np.log(np.diag(L))) - 0.5 * x.shape[1] * np.log(2 * np.pi)


def log_b(modes, x):
    w = np.array([m.w for m in modes]); w /= w.sum()
    out = []
    for wj, m in zip(w, modes):
        L = np.linalg.cholesky(np.asarray(m.Sigma))
        y = np.linalg.solve(L, (x - np.asarray(m.mu)).T).T
        out.append(np.log(wj) - 0.5 * np.sum(y * y, -1) - np.sum(np.log(np.diag(L)))
                   - 0.5 * x.shape[1] * np.log(2 * np.pi))
    return np.logaddexp(*out)


def wquantile(z, w, levels):
    o = np.argsort(z)
    c = np.cumsum(w[o]); c = (c - 0.5 * w[o]) / c[-1]
    return np.interp(levels, c, z[o])


def oracle_paths(tg, modes, betas, component):
    """(K, L) quantile maps of z_1 along the global path, or (J, K, L) along the component
    paths. Importance weights from the defensive mixture q = (b + pi) / 2 (globally, or b_j
    and pi_j per component): every path density lies between its two ends, so the weights
    stay bounded where b alone degenerates."""
    e1 = tg.R[:, 0]
    if not component:
        x = np.concatenate([b_draws(modes, N_IS // 2, 1), tg.reference(N_IS // 2, seed=2)])
        lp, lb = np.asarray(tg.logpi(jnp.asarray(x))), log_b(modes, x)
        lq, z = np.logaddexp(lp, lb), x @ e1
        lws = [(1 - b) * lb + b * lp - lq for b in betas]
        return np.stack([wquantile(z, np.exp(lw - lw.max()), LEVELS) for lw in lws])
    out = []
    for j, m in enumerate(modes):
        rng = np.random.default_rng(10 + j)
        L = np.linalg.cholesky(np.asarray(m.Sigma))
        x = np.concatenate([np.asarray(m.mu) + rng.standard_normal((N_IS // 2, D)) @ L.T,
                            tg.component_draws(j, N_IS // 2, seed=20 + j)])
        lbj = log_bj(m, x)
        h = np.asarray(tg.logpi(jnp.asarray(x))) - log_b(modes, x)
        lq, z = np.logaddexp(lbj, tg.log_component(j, x)), x @ e1
        lws = [lbj + b * h - lq for b in betas]
        out.append(np.stack([wquantile(z, np.exp(lw - lw.max()), LEVELS) for lw in lws]))
    return np.stack(out)


def field(method, res, J):
    """v(x, beta): (n, d) for GAT; (J, n, d) per component for CAT."""
    if method == "GAT":
        return lambda x, b: res.velocity(x, b)
    return lambda x, b: jnp.stack([res.velocity(x[j], b, j) for j in range(J)])


def rk4_fine(v, x, grid, on_step=None):
    """RK4-integrate x through v on `grid` (one step per interval); returns positions at every grid
    point. on_step(k, v1) sees the velocity at grid point k (the last one included)."""
    xs = [x]
    for k, b in enumerate(grid):
        v1 = v(x, b)
        if on_step is not None:
            on_step(k, v1)
        if k == len(grid) - 1:
            break
        dt = grid[k + 1] - b
        v2 = v(x + 0.5 * dt * v1, b + 0.5 * dt)
        v3 = v(x + 0.5 * dt * v2, b + 0.5 * dt)
        v4 = v(x + dt * v3, b + dt)
        x = x + dt / 6 * (v1 + 2 * v2 + 2 * v3 + v4)
        xs.append(x)
    return jnp.stack(xs)


def run(task):
    method, s = TASKS[task]
    tg = SkewTMixture(D, s)
    modes = tg.modes()
    e1 = tg.R[:, 0]
    res, wall = fit(method, jr.PRNGKey(SEED), tg.logpi, tg.logpi_single, D, modes, N_PATH)
    betas = np.asarray(res.betas_used)
    xs, _ = res.rollout(jr.PRNGKey(SEED + 777), N_PATH)
    z = np.asarray(xs) @ e1                     # GAT: (K, n); CAT: (K, J, n), every component's paths
    weights = np.ones(1) if method == "GAT" else np.asarray(res.final_weights)
    # the same particles through the learned field on a fine grid (smooth tracks for the plot)
    xf = rk4_fine(field(method, res, len(modes)), jnp.asarray(xs[0]), FINE_BETAS)
    z_fine = np.asarray(xf @ jnp.asarray(e1), dtype=np.float32)
    fine_err = float(jnp.sqrt(jnp.mean(jnp.sum((xf[-1] - jnp.asarray(xs[-1])) ** 2, -1))))
    orc = oracle_paths(tg, modes, FINE_BETAS, component=method != "GAT")
    if method == "GAT":         # fraction of particles that switch mode
        cross = float(np.mean(tg.side(np.asarray(xs[0])) != tg.side(np.asarray(xs[-1]))))
    else:   # fraction of component draws (mixed by the calibrated weights) ending in the other mode
        lab1 = np.stack([tg.side(np.asarray(xs[-1, j])) for j in range(len(modes))])  # (J, n)
        other = np.array([lab1[0].mean(), 1 - lab1[1].mean()])  # comp 0 = L, comp 1 = R
        cross = float(np.sum(weights * other))
    os.makedirs(OUT_DIR, exist_ok=True)
    np.savez(os.path.join(OUT_DIR, f"traj_{method}_d{D}_s{s:g}.npz"), betas=betas, z=z, weights=weights,
             fine_betas=FINE_BETAS, z_fine=z_fine, fine_endpoint_rms_err=fine_err,
             oracle_betas=FINE_BETAS, oracle=orc, levels=LEVELS, cross=cross, wall=wall,
             mode_z=np.array([float(np.asarray(m.mu) @ e1) for m in modes]))
    print(f"{method} gamma={s:g}: K={len(betas)} | cross {cross:.4f} | weights {np.round(weights, 3)} | "
          f"fine-grid endpoint rms vs rollout {fine_err:.2e} | {wall:.0f}s")
    save_speed(method, s, res, modes, wall)


def save_speed(method, s, res, modes, wall):
    """s(beta) of a fitted sampler at every beta of SPEED_BETAS: fresh particles
    RK4-integrated through the learned field on that grid."""
    xs, _ = res.rollout(jr.PRNGKey(SEED + 777), N_SPEED)
    e2 = []                                                           # scalar (GAT) or (J,) per beta
    x = rk4_fine(field(method, res, len(modes)), jnp.asarray(xs[0]), SPEED_BETAS,
                 lambda k, v1: e2.append(np.asarray(jnp.mean(jnp.sum(v1 * v1, -1), -1))))[-1]
    err = float(jnp.sqrt(jnp.mean(jnp.sum((x - xs[-1]) ** 2, -1))))   # vs the method's own rollout
    weights = np.ones(1) if method == "GAT" else np.asarray(res.final_weights)
    e2 = np.stack(e2).reshape(len(SPEED_BETAS), -1)
    sp = np.sqrt(e2 @ weights)
    np.savez(os.path.join(OUT_DIR, f"speed_{method}_d{D}_s{s:g}.npz"), betas=SPEED_BETAS, s=sp, e2=e2,
             weights=weights, s_integral=float(np.trapezoid(sp, SPEED_BETAS)), grid=np.asarray(res.betas_used),
             endpoint_rms_err=err, wall=wall)
    print(f"{method} gamma={s:g}: int s = {np.trapezoid(sp, SPEED_BETAS):.3f} | endpoint rms vs rollout {err:.2e}",
          flush=True)


def cum_length(f):
    """Cumulative length L(beta) = int_0^beta s(beta') dbeta' of a speed record (trapezoid)."""
    return np.concatenate([[0.0], np.cumsum(0.5 * (f["s"][1:] + f["s"][:-1]) * np.diff(f["betas"]))])


def plot(h=1.07):
    """One row per gamma: CAT's and GAT's z_1 trajectories (empirical and oracle) with the
    endpoint histograms against the exact z_1 marginal, and the cumulative length of both
    (black: CAT solid, GAT dashed, since blue / red mark the modes). Panels are laid out in
    inches (square-ish trajectory boxes of height `h`)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    GRAY = "0.55"
    MODE_COLORS = ["#006199", "#E0301E"]
    n = len(SEPARATIONS)

    def side_hist(ax, groups, ref_u, ref_p, bins):
        """Horizontal stacked density histogram of z_1 (groups: (values, total weight, color)),
        reference density as the gray dashed line."""
        vals = [g[0] for g in groups]
        wts = [np.full(len(g[0]), g[1] / (len(g[0]) * (bins[1] - bins[0]))) for g in groups]
        ax.hist(vals, bins=bins, weights=wts, stacked=True, orientation="horizontal",
                color=[g[2] for g in groups], alpha=0.55, histtype="stepfilled", lw=0)
        ax.plot(ref_p, ref_u, color=GRAY, lw=0.8, ls=(0, (3, 2)))
        ax.set_xlim(0, 1.15 * max(ref_p.max(), ax.get_xlim()[1]))
        ax.tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)
        ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(False)

    with plt.rc_context(RC_PRINT):
        gap, vgap, L, R, B, T = 0.25, 0.12, 0.62, 0.05, 0.38, 0.5   # L: row label, z_1 label and ticks
        hw, pg = 0.45, 0.08                        # histogram width, trajectory-histogram gap
        tw = 1.4 * h                               # trajectory box
        w = tw + pg + hw                           # one (trajectories | end hist) group
        m = len(METHODS)
        sg, sw = 0.55, 1.618 * h                   # gap before / width of the length column
        W, H = L + m * w + (m - 1) * gap + sg + sw + R, B + n * h + (n - 1) * vgap + T
        fig = plt.figure(figsize=(W, H))
        xc = (L + (m * w + (m - 1) * gap) / 2) / W    # centre of the trajectory columns
        sstyle = {"CAT": dict(ls="-", marker="o", mfc="k"), "GAT": dict(ls=(0, (4, 1.5)), marker="s", mfc="white")}
        sax = []
        for r, s in enumerate(SEPARATIONS):
            y = B + (n - 1 - r) * (h + vgap)
            tg = SkewTMixture(D, s)
            ylim = (-s / 2 - 7, s / 2 + 7)
            u = np.linspace(*ylim, 400)
            bins = np.linspace(*ylim, 41)
            p_end = tg.z1_density(u)                                                # exact z_1 marginal of pi
            for c, method in enumerate(METHODS):
                x0 = L + c * (w + gap)
                ax = fig.add_axes([x0 / W, y / H, tw / W, h / H])
                ax1 = fig.add_axes([(x0 + tw + pg) / W, y / H, hw / W, h / H], sharey=ax)
                f = np.load(os.path.join(OUT_DIR, f"traj_{method}_d{D}_s{s:g}.npz"))
                z, orc, ob = f["z"], f["oracle"], f["oracle_betas"]
                tb, tz = f["fine_betas"], f["z_fine"]
                if method == "GAT":     # colour by starting side
                    sideR = z[0] > 0
                    g1 = [(z[-1][sideR == k], np.mean(sideR == k), MODE_COLORS[k]) for k in (0, 1)]
                else:                   # components mixed by the calibrated weights w_j
                    wt = f["weights"]
                    g1 = [(z[-1, j], wt[j], MODE_COLORS[j]) for j in range(2)]
                side_hist(ax1, g1, u, p_end, bins)
                ax1.tick_params(left=False, labelleft=False)
                if method == "GAT":
                    for q in orc.T:
                        ax.plot(ob, q, color=GRAY, lw=0.6, ls=(0, (3, 2)), zorder=1)
                    # empirical: particles at the oracle's initial quantiles (nearest z_1 at beta = 0)
                    o = np.argsort(z[0])
                    pick = o[np.clip((LEVELS * len(o)).astype(int), 0, len(o) - 1)]
                    for i in pick:
                        ax.plot(tb, tz[:, i], color=MODE_COLORS[int(z[0, i] > 0)], lw=0.6, alpha=0.85, zorder=2)
                    note = f"{100 * float(f['cross']):.2g}% switch"
                else:
                    for j in range(orc.shape[0]):
                        for q in orc[j].T[::2]:
                            ax.plot(ob, q, color=GRAY, lw=0.6, ls=(0, (3, 2)), zorder=1)
                        o = np.argsort(z[0, j])
                        pick = o[np.clip((LEVELS[::2] * len(o)).astype(int), 0, len(o) - 1)]
                        for i in pick:
                            ax.plot(tb, tz[:, j, i], color=MODE_COLORS[j], lw=0.6, alpha=0.85, zorder=2)
                    wt = f["weights"]
                    note = f"$w$ = {wt[0]:.2f}/{wt[1]:.2f}"
                ax.text(0.97, 0.04, note, transform=ax.transAxes, ha="right", va="bottom", fontsize=6.5,
                        bbox=dict(fc="white", ec="none", alpha=0.8, pad=1))
                ax.set_xlim(0, 1)
                ax.set_ylim(*ylim)
                ax.set_yticks([-8, -4, 0, 4, 8])
                if c > 0:                                        # same y range: numbers on the left only
                    ax.tick_params(labelleft=False)
                if r == 0:
                    ax.set_title(method, fontsize=9, fontweight="bold", color=TITLE_COLOR, pad=4)
                if r < n - 1:
                    ax.tick_params(labelbottom=False)
                if c == 0:
                    ax.set_ylabel("$z_1$", labelpad=1)
            ax = fig.add_axes([(W - R - sw) / W, y / H, sw / W, h / H])
            for method in METHODS:
                f = np.load(os.path.join(OUT_DIR, f"speed_{method}_d{D}_s{s:g}.npz"))
                ax.plot(f["betas"], cum_length(f), color="k", lw=0.9, ms=3, markevery=10, label=method,
                        **sstyle[method])
            ax.set_xlim(0, 1)
            if r == 0:
                ax.set_title("Cumulative length", fontsize=9, fontweight="bold", color=TITLE_COLOR, pad=4)
            if r < n - 1:
                ax.tick_params(labelbottom=False)
            else:
                ax.set_xlabel(r"$\beta$", fontsize=8)
            sax.append(ax)
            fig.text(0.04 / W, (y + h / 2) / H, rf"$\gamma$ = {s:g}", rotation=90, ha="left", va="center",
                     fontsize=9, color="k")
        fig.supxlabel(r"$\beta$", x=xc, y=0.02 / H, va="bottom", fontsize=8)
        hs = [Line2D([], [], color=GRAY, lw=0.9, ls=(0, (3, 2))), Line2D([], [], color=MODE_COLORS[0], lw=0.9),
              Line2D([], [], color=MODE_COLORS[1], lw=0.9)]
        lab = ["Oracle", "Empirical, mode L", "Empirical, mode R (true weight 0.2)"]
        fig.legend(hs, lab, loc="upper center", bbox_to_anchor=(xc, 1.0), ncol=len(hs), frameon=False,
                   fontsize=8, handlelength=1.6, handletextpad=0.4, columnspacing=1.2)
        hs, lab = sax[0].get_legend_handles_labels()    # length legend on top of its column
        fig.legend(hs, lab, loc="upper center", bbox_to_anchor=((W - R - sw / 2) / W, 1.0), ncol=len(hs),
                   frameon=False, fontsize=8, handlelength=2.2, handletextpad=0.4, columnspacing=1.2)
        for ax in sax:                                   # one length scale across rows
            ax.set_ylim(0, 1.05 * max(a.get_ylim()[1] for a in sax))
        save_figure(fig, os.path.join(FIG_DIR, f"paths_skewt_d{D}"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", type=int, help=f"fit one (method, gamma) of {TASKS}")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()
    if args.plot:
        plot()
    elif args.task is not None:
        run(args.task)
    else:
        for task in range(len(TASKS)):
            run(task)
        plot()


if __name__ == "__main__":
    main()
