# src/samplers/_fabjax/__init__.py
"""Vendored (and patched) subset of `github.com/lollcat/fab-jax` (Laurence
Midgley, MIT licensed -- see `LICENSE` in this directory), the official JAX
implementation of Midgley et al. 2022's FAB (Flow Annealed Importance
Sampling Bootstrap). Used by `..fab.py`.

**What's vendored, and what's trimmed.** Only the with-buffer training path:
  - `flow/`     -- the distrax spline/RealNVP-coupling flow (`build_flow`,
                   `FlowDistConfig`) and its flax-MLP conditioner
                   (`utils/nets.py`).
  - `sampling/` -- the SMC/AIS ladder (`smc.py`, `base.py`, `resampling.py`,
                   `point_is_valid.py`) and the HMC transition operator
                   (`mcmc/hmc.py`, `mcmc/blackjax_hmc_rewrite.py`).
  - `buffer/`   -- the prioritised replay buffer (`prioritised_buffer.py`;
                   an earlier, narrower investigation vendored just this one
                   file as `_fab_buffer.py` before the rest of fab-jax was
                   confirmed patchable -- removed now that this fuller
                   vendor supersedes it, see `fab.py`'s docstring).
  - `train/`    -- `fab_with_buffer.py` (the buffer-based training loop) and
                   `utils/optimize.py` (`IgnoreNanOptState`, used by it).
Not vendored (upstream's own `__init__.py`s are trimmed here to match, see
each subpackage's `__init__.py`): the no-buffer training variant
(`fab_without_buffer.py`), the Metropolis transition operator
(`mcmc/metropolis.py`), rejection sampling, wandb-based logging, pickle
checkpointing, matplotlib plotting, and upstream's own built-in test problems
(`targets/`) -- this repo supplies its own target via an arbitrary `logpi`.

**Patches applied for compatibility with this repo's current JAX/blackjax**
(fab-jax is a few years old; none of this is a algorithmic change, all of it
mechanical API drift -- confirmed by running the patched with-buffer training
loop end-to-end against a toy Gaussian target and recovering the correct
mean/std before this was wired into `fab.py`):
  - `jax.tree_map` -> `jax.tree_util.tree_map` (renamed upstream in JAX;
    every call site in this vendored subset).
  - `sampling/base.py`: a `NamedTuple` field default of `jnp.array(None)`
    raises under current JAX (`None is not a valid value for jnp.array`);
    changed to a plain `None` default.
  - `sampling/mcmc/blackjax_hmc_rewrite.py` (the file needing the most
    patching, since it reaches into blackjax's own internals):
      - `from blackjax.types import PyTree` -- no longer exists; `PyTree`
        (only ever used as a type hint here, never at runtime) is stubbed
        to `object`. Same for `from blackjax.mcmc.integrators import
        EuclideanKineticEnergy`, also type-hint-only and removed upstream.
      - `blackjax.mcmc.metrics.gaussian_euclidean` used to return a plain
        3-tuple; it now returns a `Metric` NamedTuple with 4 fields
        (`sample_momentum`, `kinetic_energy`, ...) -- unpacking updated to
        use the named fields instead of positional tuple unpacking.
      - `blackjax.mcmc.proposal.proposal_generator` no longer takes a
        `divergence_threshold` argument, and its `update` function now
        returns a single `Proposal(state, energy, weight, sum_log_p_accept)`
        instead of `(new_proposal, is_diverging)`. `is_diverging` is
        reconstructed here the same way blackjax's own current
        `mcmc.hmc.hmc_proposal` does it (`-weight > divergence_threshold`).
      - `blackjax.mcmc.proposal.static_binomial_sampling` gained a
        `log_p_accept` positional argument (supplied here from
        `new_proposal.sum_log_p_accept`) and its returned info is now a
        nested 3-tuple (`do_accept, p_accept, None`) rather than being
        spliceable directly into the caller's unpacking -- call site and
        unpacking updated to match.

If this repo's JAX/blackjax versions move again in the future, this is the
file most likely to need another round of the same kind of patching.
"""
