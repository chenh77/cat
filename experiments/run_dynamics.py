#!/usr/bin/env python
"""experiments/run_dynamics.py

Single-seed training diagnostics of CAT, GAT and the CNF, on top of the benchmark run:
the target's own `run_<target>.run_experiment` does mode discovery and runs the three
methods at exactly the benchmark configurations and budgets (all other baselines
skipped); evaluation callbacks (`transport_stage.CALLBACKS`) score the current model
after every refresh. Callback time and target calls are excluded from each method's wall
clock and oracle budget, so the run itself is the benchmark run.

Per refresh (`dynamics.json`, "curves"), on `--n-eval` fresh draws of the current model:
  - elapsed: the method's transport wall time so far (mode discovery excluded);
    iteration: optimizer steps so far; train_start: elapsed at the first training step
    (setup and JIT compilation before it); compile_s: JAX compile time so far
    (`src/samplers/compile_timer.py`).
  - ESS / n and Var(log w), w = pi / q; kl_offset = E_q[log q - log pi~] (so that the
    ELBO is -kl_offset and reverse KL(q || pi) = kl_offset + log Z, with log Z =
    `log_Z_ref`, CAT's final importance estimate on its final draws).
  - SW2 to the reference (`target_eval.TargetEval.sw2`, the same reference subset and
    projections at every evaluation), mode weights (and mode TV on the symmetric targets).
  CAT and the CNF (one flow per mode) are scored as their final sampler is: the mode
  weights are calibrated from the endpoint importance weights of the current fields
  (log C_j = logmeanexp(log pi_j - log q_j)), and log q of a mixture draw is
  log sum_j w_j q_j via the inverse ODE.
Per optimizer step: CAT and GAT's training loss (`train_loss`); the CNF's mean loss per
refresh.

Usage:
  python experiments/run_dynamics.py --target lda --seed 2026
  python experiments/run_dynamics.py --target iohmm --methods CAT
Output: experiments/results/dynamics/<target>/seed_<seed>/ (dynamics.json, plus the
standard benchmark outputs of the three methods).
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import time

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.scipy.special import logsumexp

import benchmark
import common
import transport_stage as ts
from src.samplers.cat import _gauss_logpdf
from target_eval import TargetEval, is_metrics

ALL_VC = ("CAT", "GAT", "CNF")


def _logmeanexp(a, axis=-1):
    return logsumexp(a, axis=axis) - np.log(a.shape[axis])


def component_logpi(te: TargetEval, x, loga, mu, chol):
    """log pi_j(x) = log a_j + log b_j(x) + log pi~(x) - log sum_k a_k b_k(x), (J, n),
    and log pi~(x) (n,)."""
    lb = jnp.stack([_gauss_logpdf(x, mu[k], chol[k]) for k in range(mu.shape[0])], 0)   # (J, n)
    lp = te.log_prob(np.asarray(x))
    return np.asarray(lb + loga[:, None] + (lp - logsumexp(lb + loga[:, None], axis=0))[None]), lp


def mixture_draws(te, st, key, n):
    """(x, log q, log pi~, log C) of n draws of the current per-mode-flow mixture,
    with the weights calibrated as the final sampler does."""
    k_roll, k_comp = jax.random.split(key)
    xs, lqs = st["rollout"](k_roll, n)
    x1, lq1 = np.asarray(xs[-1]), np.asarray(lqs[-1])                 # (J, n, d), (J, n)
    J = x1.shape[0]
    logC = np.empty(J)
    for j in range(J):
        lpj, _ = component_logpi(te, x1[j], st["loga"], st["mu"], st["chol"])
        logC[j] = _logmeanexp(lpj[j] - lq1[j])
    logw = logC - logsumexp(logC)
    comp = np.asarray(jax.random.categorical(k_comp, jnp.asarray(logw), shape=(n,)))
    x = x1[comp, np.arange(n)]
    lq_all = np.array(st["inverse"](jnp.asarray(x)))                   # (J, n), writable copy
    lq_all[comp, np.arange(n)] = lq1[comp, np.arange(n)]
    logq = np.asarray(logsumexp(lq_all + logw[:, None], axis=0))
    return x, logq, te.log_prob(x), logC


def make_callback(label, te, records, n_eval, sw_key):
    def cb(st):
        t0 = time.time()
        key = jax.random.PRNGKey(10_000 + st["refresh"])
        rec = dict(refresh=int(st["refresh"]), elapsed=float(st["elapsed"]),
                   train_start=st.get("train_start"), compile_s=st.get("compile_s"))
        if label == "GAT":
            xs, lqs = st["rollout"](key, n_eval)
            x, logq = np.asarray(xs[-1]), np.asarray(lqs[-1])
            logp = te.log_prob(x)
        else:
            x, logq, logp, logC = mixture_draws(te, st, key, n_eval)
            rec["logC"] = logC.tolist()
        ok = np.isfinite(logp) & np.isfinite(logq) & np.all(np.isfinite(x), -1)
        rec["ess"], rec["logw_var"] = is_metrics(logp[ok], logq[ok])
        rec["kl_offset"] = float(np.mean(logq[ok] - logp[ok]))
        rec["kl_se"] = float(np.std(logq[ok] - logp[ok]) / np.sqrt(ok.sum()))     # Monte Carlo s.e.
        rec["log_Z_is"] = float(_logmeanexp(logp[ok] - logq[ok]))
        rec["n_nonfinite"] = int((~ok).sum())
        rec["sw2"] = te.sw2(x[ok], sw_key, n_sw=min(2048, n_eval))
        rec["mode_weights"] = te.mode_weights(x).tolist()
        if te.symmetric:
            rec["mode_tv"] = te.mode_tv(x)
        rec["eval_time"] = time.time() - t0
        records.append(rec)
        print(f"  [dynamics/{label}] refresh {rec['refresh'] + 1} t={rec['elapsed']:.0f}s ESS {rec['ess']:.3f} "
              f"KL-logZ {rec['kl_offset']:.3f} SW2 {rec['sw2']:.4f} ({rec['eval_time']:.0f}s eval)", flush=True)
    return cb


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", required=True, choices=common.TARGETS)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--methods", default=",".join(ALL_VC))
    ap.add_argument("--n-eval", type=int, default=2048)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()
    methods = args.methods.split(",")
    out_dir = args.out_dir or os.path.join(common.RESULTS_DIR, "dynamics", args.target, f"seed_{args.seed}")
    os.makedirs(out_dir, exist_ok=True)

    te = TargetEval.load(args.target)
    sw_key = jax.random.PRNGKey(777)
    records = {m: [] for m in methods}
    ts.CALLBACKS.update({m: make_callback(m, te, records[m], args.n_eval, sw_key) for m in methods})

    run_mod = importlib.import_module(f"run_{args.target}")
    skip = frozenset(benchmark.METHOD_LABELS) - frozenset(methods) - {"CAT"}
    t0 = time.time()
    run_mod.run_experiment(seed=args.seed, out_dir=out_dir, skip_methods=skip)
    print(f"[dynamics] run finished in {time.time() - t0:.0f}s", flush=True)

    # log Z reference: CAT's final importance estimate on its final draws
    cat = ts.LAST_ENTRIES["CAT"]["res"]
    lp_cat = te.log_prob(np.asarray(cat.final_snapshot))
    log_Z_ref = float(_logmeanexp(lp_cat - np.asarray(cat.final_logq)))
    out = dict(target=args.target, seed=args.seed, methods=methods, log_Z_ref=log_Z_ref,
               log_Z_ref_note="CAT final IS estimate logmeanexp(log pi~ - log q) on its n_snapshot final draws",
               curves=records, final={}, train_loss={}, steps_per_refresh={})
    for m in methods:
        e = ts.LAST_ENTRIES.get(m)
        if e is None:
            print(f"[dynamics] {m} failed or did not run")
            continue
        res = e["res"]
        x = np.asarray(res.final_snapshot if m != "GAT" else res.samples)
        lq = np.asarray(res.final_logq)
        lp = te.log_prob(x)
        ess, lwv = is_metrics(lp, lq)
        out["final"][m] = dict(ess=ess, logw_var=lwv, kl=float(np.mean(lq - lp) + log_Z_ref),
                               sw2=te.sw2(x, sw_key), mode_weights=te.mode_weights(x).tolist(),
                               transport_wall_time_s=e["wall_time_s"], selected_refresh=res.selected_refresh,
                               wall_time_s=res.wall_time, train_start_s=getattr(res, "train_start", None),
                               compile_time_s=getattr(res, "compile_time", None))
        out["steps_per_refresh"][m] = int(e["info"]["kwargs"].get("train_steps_per_refresh", 250))
        if hasattr(res, "train_step_losses") and res.train_step_losses:
            out["train_loss"][m] = np.concatenate([np.asarray(a) for a in res.train_step_losses]).tolist()
        else:
            out["train_loss"][m] = [float(v) for v in res.train_loss_history]
    for m, recs in records.items():
        for r in recs:
            r["iteration"] = (r["refresh"] + 1) * out["steps_per_refresh"].get(m, 0)
            r["kl"] = r["kl_offset"] + log_Z_ref
    with open(os.path.join(out_dir, "dynamics.json"), "w") as f:
        json.dump(out, f, indent=1, default=float)
    print(f"[dynamics] saved {out_dir}/dynamics.json; final: "
          + json.dumps({m: {k: round(v, 4) for k, v in d.items() if isinstance(v, float)} for m, d in out["final"].items()}))


if __name__ == "__main__":
    main()
