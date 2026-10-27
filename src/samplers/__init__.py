# samplers/__init__.py
"""CAT and the baseline samplers it is compared with.

  - `cat.py`         -- CAT: one time-conditioned vector field per discovered mode, trained
                        on the continuity-equation residual along the mode's annealing path
                        b_j^{1-beta} pi_j^beta, with delayed weight calibration.
  - `gat.py`         -- GAT: the global-path counterpart of CAT, one vector field along
                        b^{1-beta} pi^beta from the Laplace mixture b.
  - `cnf.py`         -- CNF: CAT's vector-field parametrization per mode, trained by reverse
                        KL to the endpoint pi_j through the ODE solver (no prescribed path).
  - `apt.py`         -- adaptive parallel tempering (APT) and its deterministic even-odd
                        variant (APT-DEO), with a MALA replica kernel (`mala.py`).
  - `alps.py`        -- ALPS: mode-informed leap-point MCMC (Tawn et al., 2021).
  - `fab.py`         -- FAB (Midgley et al., 2023), running the vendored fab-jax
                        implementation in `_fabjax/` (MIT license, see `_fabjax/LICENSE`).
  - `temperflow.py`  -- TemperFlow (Qiu & Wang, 2024): one global flow along an adaptive
                        temperature schedule.
  - `aft.py`         -- AFT: annealed flow transport (Arbel et al., 2021).

Every baseline can start from the Laplace mixture of the discovered modes (`modes=`,
`_mode_base.py`), so all methods share one mode discovery (`src.mode_explore`).
"""
