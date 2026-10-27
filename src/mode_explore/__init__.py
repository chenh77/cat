# mode_explore/__init__.py
"""Mode discovery shared by every sampler.

  - `explore.py`   -- ModeExplore: a hot-chain MCMC warm-up followed by quasi-Newton
                      (L-BFGS) mode finding and Laplace covariances; `multi_round_mode_explore`
                      repeats it with fresh particles. Returns `Mode(mu, Sigma, w)` records.
  - `proposals.py` -- MCMC kernels of the hot chain (RWMH, MALA, independence MH, ULA).
  - `density.py`   -- the Laplace mixture b = sum_j w_j N(mu_j, Sigma_j) of the discovered
                      modes (`modes_to_arrays`, batched mixture log densities) and the
                      standard-normal base distribution (`StandardNormal`).
"""
