# CAT

Code for the paper [*Transport along Local Annealing Paths: Wasserstein Geometry and Multimodal Sampling*](https://arxiv.org/abs/2611.05605).

This repository contains:

- the JAX implementation of CAT;
- the baselines CAT is compared with;
- the scripts that produce the experiments of the paper.

## Installation

Python 3.13; all computations run in float64.

```bash
pip install -r requirements.txt
# GPU (CUDA 12):
pip install "jax[cuda12]==0.9.0"
```

Run every script from the repository root. The scripts put the root on `sys.path` themselves.

## Repository layout

```
src/
  samplers/      CAT (cat.py) and the baselines: GAT (gat.py), CNF (cnf.py), APT / APT-DEO (apt.py),
                 ALPS (alps.py), FAB (fab.py, vendored fab-jax in _fabjax/), TemperFlow (temperflow.py),
                 AFT (aft.py)
  mode_explore/  mode discovery (explore.py), its MCMC kernels (proposals.py), Laplace mixtures (density.py)
  flow/          vector field of CAT / GAT / CNF (vector_field.py), RealNVP flows (nets.py, chain.py)
  targets/       skew-normal mixture, skew-t mixture, SUR, IOHMM, LDA
  utils/         evaluation metrics (eval.py), oracle-cost accounting (budget.py), linear algebra (numeric.py)
experiments/     run, evaluation and plotting scripts (below)
data/            posteriordb datasets, NUTS reference draws, SUR data (see data/README.md)
```

## Methods

Every method uses the same mode discovery. CAT runs first, and its oracle cost sets the budget for everything else. That cost counts target evaluations, weighted as value + d · gradient, and excludes the shared discovery. Every other method is matched to it.

| Label | Method | Source |
|---|---|---|
| CAT | per-mode vector fields along the component annealing paths, delayed weight calibration | `src/samplers/cat.py` |
| GMM | the Laplace mixture b itself (CAT's starting point) | `experiments/transport_stage.py` |
| GAT | one vector field along the global path b^{1-β} π^β | `src/samplers/gat.py` |
| CNF | CAT's vector fields trained by reverse KL, with no prescribed path | `src/samplers/cnf.py` |
| APT, APT-DEO | adaptive parallel tempering | `src/samplers/apt.py` |
| ALPS | annealed leap-point sampler (Tawn et al., 2021) | `src/samplers/alps.py` |
| FAB | flow annealed importance sampling bootstrap (Midgley et al., 2023) | `src/samplers/fab.py` |
| TF | TemperFlow (Qiu & Wang, 2024) | `src/samplers/temperflow.py` |
| AFT | annealed flow transport (Arbel et al., 2021) | `src/samplers/aft.py` |

## Experiments

The commands below use seed 2026. The paper's tables average over seeds 2026–2125, and all paper results were run on an NVIDIA RTX 5000 Ada GPU. Every script's `--help` documents its options and outputs. Results go to `experiments/results/` and figures to `experiments/figures/`.

**Benchmark (four targets, all methods).** Each run saves one seed's draws, costs and diagnostics to `experiments/results/benchmark/<target>/seed_<seed>/`.

```bash
python experiments/run_skewnormal.py --seed 2026
python experiments/run_sur.py --seed 2026
python experiments/run_iohmm.py --seed 2026
python experiments/run_lda.py --seed 2026
```

The figures and tables below come from these runs.

1. **CAT training dynamics on IOHMM and LDA.** Shows the training loss, and ESS/N and ELBO after every refresh, with b at iteration 0. Needs the benchmark run of the same seed for b.
   ```bash
   python experiments/run_dynamics.py --target iohmm --seed 2026
   python experiments/run_dynamics.py --target lda --seed 2026
   python experiments/plot_dynamics.py training --targets iohmm,lda
   ```
2. **Wall-clock comparison on LDA (CAT vs GAT vs CNF).** Shows ESS/N, SW2 and ELBO against wall-clock time, minus JAX compile time. Uses the dynamics run of experiment 1.
   ```bash
   python experiments/plot_dynamics.py wallclock --target lda
   ```
3. **Transport paths of CAT and GAT on the skew-t mixture.** Shows z₁ trajectories against oracle quantile paths, plus the cumulative path length.
   ```bash
   python experiments/skewt_paths.py            # all four fits, then the figure
   ```
4. **Summary table.** Covers the skew-normal mixture, SUR, IOHMM and LDA. Compares CAT, GMM, FAB, TF, APT, APT-DEO, ALPS and AFT on:
   - mode recovery, mode TV and weight TV;
   - SW2 to the reference;
   - wall-clock time;
   - ESS/N and Var(log w).

   Uses the benchmark runs of every seed in `experiments/results/benchmark/`, and writes `experiments/figures/summary_table.tex`.
   ```bash
   python experiments/eval_summary.py
   ```
5. **First-coordinate projection on the skew-normal mixture, 8 methods.** Compares x₁ histograms against the oracle density.
   ```bash
   python experiments/plot_skewnormal.py --seed 2026
   ```
6. **Sensitivity of CAT to the velocity penalty λ (appendix table).** Runs CAT only, exactly as in the benchmark, with λ ∈ {0, 10⁻⁴, 3·10⁻⁴, 10⁻³, 3·10⁻³, 10⁻², 10⁻¹} (default 10⁻³). It uses seeds 4026–4030, which are disjoint from the benchmark seeds. The table reports the relative ESS (ESS/N of the final draws), and is written to `experiments/figures/lambda_table.tex`. Runs go to `experiments/results/lambda_sweep/<target>/lam_<λ>/seed_<seed>/`; a finished run is skipped, and `--lambdas` and `--seeds` select a subset (e.g. one cluster job per run).
   ```bash
   python experiments/lambda_sweep.py run --target skewnormal     # likewise sur, iohmm, lda
   python experiments/lambda_sweep.py table
   ```

Shared experiment code:

| File | Contents |
|---|---|
| `experiments/transport_stage.py` | per-target configurations of CAT, CNF and GAT, and the budget matching |
| `experiments/benchmark.py` | mode discovery and output format shared by the run scripts |
| `experiments/target_eval.py` | per-target scoring: modes, reference draws, SW2 |
| `experiments/common.py` | paths, data loaders and plot style |

## Third-party code

- `src/samplers/_fabjax/` is vendored from [fab-jax](https://github.com/lollcat/fab-jax) (MIT license, `src/samplers/_fabjax/LICENSE`), patched for current JAX and blackjax.
- The IOHMM and LDA targets and their datasets come from [posteriordb](https://github.com/stan-dev/posteriordb).

