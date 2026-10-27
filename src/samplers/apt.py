# apt.py
"""Adaptive parallel tempering (APT) and its deterministic even-odd variant (APT-DEO),
a pure MCMC sampler over an inverse-temperature ladder pi^beta, with no learned
transport.

`adaptive_parallel_tempering` runs one MALA replica per temperature level (`mala.py`) with
Miasojedow-Moulines-Vihola diminishing adaptation of the ladder's log-gaps and of each
rung's MALA step size during burn-in, frozen afterwards. `use_deo` selects the swap scheme:
  - `use_deo=True`  (APT-DEO): deterministic even-odd sweeps, an even-parity and an
    odd-parity swap pass every iteration;
  - `use_deo=False` (APT): one randomly chosen adjacent pair proposed per iteration.

`run_adaptive_pt` is a wrapper with the call signature of the other baselines
(`logpi_single`, an increasing `betas` ladder, a small result dataclass): it reverses the
ladder into the decreasing-from-1 `initial_inverse_temperature_schedule` the sampler
expects, runs one long chain, and keeps the post-burn-in, thinned cold-replica
states as `samples`.

**Mode-informed variant** (`run_adaptive_pt(..., modes=...)`): the sampler is unchanged
(replica beta targets pi^beta); the modes only initialize the chains, one state per rung
drawn from the Laplace mixture b of the discovered modes (`_mode_base.LaplaceMixtureBase`)
instead of N(0, x0_scale^2 I).

The output samples are correlated MCMC draws targeting pi, with no tractable density.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, Optional

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit, lax, random, vmap

from ._mode_base import build_mode_base
from .mala import run_mala_steps

Array = jnp.ndarray
LogDensity = Callable[[Array], Array]   # (d,) -> scalar, i.e. this repo's logpi_single


def _inverse_temperatures_from_rho(rho: Array) -> Array:
    ratios = jnp.exp(-jnp.exp(rho))
    return jnp.concatenate(
        [jnp.ones((1,), dtype=rho.dtype), jnp.cumprod(ratios)]
    )


@partial(jit, static_argnames=("log_density", "num_mala_steps", "ref_log_density"))
def _run_mala_replica_updates(
    log_density: LogDensity,
    states: Array,
    base_logps: Array,
    inverse_temperatures: Array,
    step_sizes: Array,
    num_mala_steps: int,
    rng_key: Array,
    ref_log_density: Optional[LogDensity] = None,
) -> tuple[Array, Array, Array]:
    replica_keys = random.split(rng_key, states.shape[0])

    if ref_log_density is not None:
        # Reference-based bridge: `base_logps` holds h = logpi - log b and
        # replica beta targets log b + beta*h.
        def one_replica_ref(x, h, beta, step_size, key):
            def bridge(p):
                return (1.0 - beta) * ref_log_density(p) + beta * log_density(p)

            x_next, logp_next, accept_rate = run_mala_steps(
                bridge, x, step_size=step_size, num_steps=num_mala_steps, rng_key=key,
                initial_logp=ref_log_density(x) + beta * h, log_density_scale=1.0,
            )
            # h recomputed directly: recovering it as (logp_next - ref)/beta
            # gives NaN once the adapted ladder drives a beta to exactly 0.
            del logp_next
            return x_next, log_density(x_next) - ref_log_density(x_next), accept_rate

        return vmap(one_replica_ref)(
            states, base_logps, inverse_temperatures, step_sizes, replica_keys,
        )

    def one_replica(x, base_logp, beta, step_size, key):
        x_next, scaled_logp_next, accept_rate = run_mala_steps(
            log_density,
            x,
            step_size=step_size,
            num_steps=num_mala_steps,
            rng_key=key,
            initial_logp=beta * base_logp,
            log_density_scale=beta,
        )
        return x_next, scaled_logp_next / beta, accept_rate

    return vmap(one_replica)(
        states, base_logps, inverse_temperatures, step_sizes, replica_keys,
    )


def _swap_sweep_with_pair_stats(
    states: Array,
    base_logps: Array,
    inverse_temperatures: Array,
    rng_key: Array,
    start_index: int,
) -> tuple[Array, Array, Array, Array]:
    num_pairs = inverse_temperatures.shape[0] - 1
    pair_indices = jnp.arange(start_index, num_pairs, 2)
    left_indices = pair_indices
    right_indices = pair_indices + 1

    beta_left = inverse_temperatures[left_indices]
    beta_right = inverse_temperatures[right_indices]
    logp_left = base_logps[left_indices]
    logp_right = base_logps[right_indices]

    log_alpha = (beta_left - beta_right) * (logp_right - logp_left)
    accepted = jnp.log(random.uniform(rng_key, shape=pair_indices.shape)) < log_alpha

    state_mask = accepted.reshape((accepted.shape[0],) + (1,) * (states.ndim - 1))
    left_states = states[left_indices]
    right_states = states[right_indices]
    swapped_left_states = jnp.where(state_mask, right_states, left_states)
    swapped_right_states = jnp.where(state_mask, left_states, right_states)

    swapped_left_logps = jnp.where(accepted, logp_right, logp_left)
    swapped_right_logps = jnp.where(accepted, logp_left, logp_right)

    states = states.at[left_indices].set(swapped_left_states)
    states = states.at[right_indices].set(swapped_right_states)
    base_logps = base_logps.at[left_indices].set(swapped_left_logps)
    base_logps = base_logps.at[right_indices].set(swapped_right_logps)

    accepts_by_pair = jnp.zeros((num_pairs,), dtype=jnp.float32)
    attempts_by_pair = jnp.zeros((num_pairs,), dtype=jnp.float32)
    accepts_by_pair = accepts_by_pair.at[pair_indices].set(accepted.astype(jnp.float32))
    attempts_by_pair = attempts_by_pair.at[pair_indices].set(1.0)

    return states, base_logps, accepts_by_pair, attempts_by_pair


def _swap_single_random_pair(
    states: Array,
    base_logps: Array,
    inverse_temperatures: Array,
    rng_key: Array,
) -> tuple[Array, Array, Array, Array]:
    num_pairs = inverse_temperatures.shape[0] - 1
    pair_key, accept_key = random.split(rng_key)
    pair_idx = random.randint(pair_key, shape=(), minval=0, maxval=num_pairs)

    beta_left = inverse_temperatures[pair_idx]
    beta_right = inverse_temperatures[pair_idx + 1]
    logp_left = base_logps[pair_idx]
    logp_right = base_logps[pair_idx + 1]

    log_alpha = (beta_left - beta_right) * (logp_right - logp_left)
    accepted = jnp.log(random.uniform(accept_key)) < log_alpha

    left_state = states[pair_idx]
    right_state = states[pair_idx + 1]
    new_left_state = jnp.where(accepted, right_state, left_state)
    new_right_state = jnp.where(accepted, left_state, right_state)
    new_left_logp = jnp.where(accepted, logp_right, logp_left)
    new_right_logp = jnp.where(accepted, logp_left, logp_right)

    states = states.at[pair_idx].set(new_left_state)
    states = states.at[pair_idx + 1].set(new_right_state)
    base_logps = base_logps.at[pair_idx].set(new_left_logp)
    base_logps = base_logps.at[pair_idx + 1].set(new_right_logp)

    swap_accepts = jnp.zeros((num_pairs,), dtype=jnp.float32).at[pair_idx].set(
        accepted.astype(jnp.float32)
    )
    swap_attempts = jnp.zeros((num_pairs,), dtype=jnp.float32).at[pair_idx].set(1.0)

    return states, base_logps, swap_accepts, swap_attempts


@partial(
    jit,
    static_argnames=("log_density", "num_iterations", "num_mala_steps", "burnin", "use_deo",
                     "keep_start", "keep_every", "ref_log_density", "ref_sampler"),
)
def _scan_adaptive_parallel_tempering(
    log_density: LogDensity,
    initial_states: Array,
    initial_base_logps: Array,
    initial_rho: Array,
    initial_log_step_sizes: Array,
    num_iterations: int,
    burnin: int,
    num_mala_steps: int,
    rng_key: Array,
    target_swap_acceptance: float,
    target_mala_acceptance: float,
    gamma_temperature_base: float,
    gamma_mala_base: float,
    gamma_temperature_power: float,
    gamma_mala_power: float,
    rho_min: float,
    rho_max: float,
    log_step_size_min: float,
    log_step_size_max: float,
    use_deo: bool = True,
    keep_start: int = 0,
    keep_every: int = 1,
    ref_log_density: Optional[LogDensity] = None,
    ref_sampler: Optional[Callable[[Array], Array]] = None,
) -> tuple[Array, Array, Array, Array, Array, Array, Array, Array, Array]:
    keys = random.split(rng_key, num_iterations)
    iteration_indices = jnp.arange(num_iterations)
    # Only the kept cold states (iterations keep_start, keep_start + keep_every, ...) are
    # written into a preallocated buffer inside the scan. Stacking every iteration's
    # cold state costs num_iterations * d memory (~9 GB at d=3075, 3.8e5 iterations).
    n_keep = len(range(int(keep_start), int(num_iterations), int(keep_every)))
    keep_buffer = jnp.zeros((max(n_keep, 1),) + initial_states.shape[1:], dtype=initial_states.dtype)

    def one_iteration(carry, scan_input):
        states, base_logps, rho, log_step_sizes, keep_buffer = carry
        key, iteration_index = scan_input

        inverse_temperatures = _inverse_temperatures_from_rho(rho)
        step_sizes = jnp.exp(log_step_sizes)

        if ref_log_density is not None:
            key, refresh_key = random.split(key)
        if use_deo:
            move_key, even_swap_key, odd_swap_key = random.split(key, 3)
        else:
            move_key, swap_key = random.split(key, 2)

        states, base_logps, mala_accept_rates = _run_mala_replica_updates(
            log_density, states, base_logps, inverse_temperatures, step_sizes,
            num_mala_steps, move_key, ref_log_density,
        )

        if ref_log_density is not None:
            # Independence-MH refresh of the hottest replica with proposal b:
            # target b*exp(beta*h), so the ratio is exp(beta*(h(y) - h(x))).
            sample_key, accept_key = random.split(refresh_key)
            y = ref_sampler(sample_key)
            h_y = log_density(y) - ref_log_density(y)
            beta_hot = inverse_temperatures[-1]
            refresh_accept = jnp.log(random.uniform(accept_key)) < beta_hot * (h_y - base_logps[-1])
            states = states.at[-1].set(jnp.where(refresh_accept, y, states[-1]))
            base_logps = base_logps.at[-1].set(jnp.where(refresh_accept, h_y, base_logps[-1]))
        else:
            refresh_accept = jnp.asarray(False)

        if use_deo:
            states, base_logps, even_accepts, even_attempts = _swap_sweep_with_pair_stats(
                states, base_logps, inverse_temperatures, even_swap_key, start_index=0,
            )
            states, base_logps, odd_accepts, odd_attempts = _swap_sweep_with_pair_stats(
                states, base_logps, inverse_temperatures, odd_swap_key, start_index=1,
            )
            swap_accepts = even_accepts + odd_accepts
            swap_attempts = even_attempts + odd_attempts
        else:
            states, base_logps, swap_accepts, swap_attempts = _swap_single_random_pair(
                states, base_logps, inverse_temperatures, swap_key,
            )
        swap_accept_rates = jnp.where(
            swap_attempts > 0.0, swap_accepts / swap_attempts, 0.0,
        )

        step_number = iteration_index + 1
        gamma_temperature = gamma_temperature_base * (
            step_number.astype(rho.dtype) ** (-gamma_temperature_power)
        )
        gamma_mala = gamma_mala_base * (
            step_number.astype(log_step_sizes.dtype) ** (-gamma_mala_power)
        )

        adapted_rho = jnp.where(
            swap_attempts > 0.0,
            jnp.clip(
                rho + gamma_temperature * (swap_accept_rates - target_swap_acceptance),
                rho_min, rho_max,
            ),
            rho,
        )
        adapted_log_step_sizes = jnp.clip(
            log_step_sizes + gamma_mala * (mala_accept_rates - target_mala_acceptance),
            log_step_size_min, log_step_size_max,
        )
        should_adapt = iteration_index < burnin
        rho = jnp.where(should_adapt, adapted_rho, rho)
        log_step_sizes = jnp.where(should_adapt, adapted_log_step_sizes, log_step_sizes)
        inverse_temperatures = _inverse_temperatures_from_rho(rho)
        step_sizes = jnp.exp(log_step_sizes)

        rel = iteration_index - keep_start
        is_kept = (rel >= 0) & (rel % keep_every == 0)
        slot = jnp.clip(rel // keep_every, 0, keep_buffer.shape[0] - 1)
        # single-row in-place update (a full-buffer select would copy the buffer every iteration)
        keep_buffer = keep_buffer.at[slot].set(jnp.where(is_kept, states[0], keep_buffer[slot]))

        return (states, base_logps, rho, log_step_sizes, keep_buffer), (
            swap_accepts, swap_attempts, mala_accept_rates,
            inverse_temperatures, step_sizes, refresh_accept,
        )

    (_, _, final_rho, final_log_step_sizes, keep_buffer), scan_outputs = lax.scan(
        one_iteration,
        (initial_states, initial_base_logps, initial_rho, initial_log_step_sizes, keep_buffer),
        (keys, iteration_indices),
    )
    (
        swap_accepts, swap_attempts, mala_accept_rates,
        inverse_temperature_history, step_size_history, refresh_accepts,
    ) = scan_outputs
    cold_states = keep_buffer[:n_keep]
    final_inverse_temperatures = _inverse_temperatures_from_rho(final_rho)
    final_step_sizes = jnp.exp(final_log_step_sizes)

    return (
        cold_states, swap_accepts, swap_attempts, mala_accept_rates,
        inverse_temperature_history, step_size_history,
        final_inverse_temperatures, final_step_sizes, refresh_accepts,
    )


def _validate_bounds(name: str, bounds: tuple[float, float]) -> tuple[float, float]:
    if len(bounds) != 2:
        raise ValueError(f"{name} must contain exactly two values.")
    lower, upper = float(bounds[0]), float(bounds[1])
    if lower >= upper:
        raise ValueError(f"{name} lower bound must be smaller than upper bound.")
    return lower, upper


def adaptive_parallel_tempering(
    log_density: LogDensity,
    initial_states: Array,
    initial_inverse_temperature_schedule: Sequence[float],
    num_iterations: int,
    burnin: int = 0,
    thinning: int = 1,
    initial_step_size: float | Sequence[float] = 0.1,
    num_mala_steps: int = 1,
    rng_key: Array | None = None,
    target_swap_acceptance: float = 0.234,
    target_mala_acceptance: float = 0.57,
    gamma_temperature_base: float = 1.0,
    gamma_mala_base: float = 1.0,
    gamma_temperature_power: float = 0.7,
    gamma_mala_power: float = 0.7,
    rho_bounds: tuple[float, float] = (-20.0, 5.0),
    log_step_size_bounds: tuple[float, float] = (-20.0, 5.0),
    store_history: bool = False,
    use_deo: bool = True,
    discard_burnin: bool = True,
    ref_log_density: Optional[LogDensity] = None,
    ref_sampler: Optional[Callable[[Array], Array]] = None,
) -> tuple[Array, dict[str, Array]]:
    """Run adaptive parallel tempering with MALA local transitions.

    Adaptation of the inverse-temperature ladder and MALA step sizes is
    performed only during burnin. After burnin, the final tuning parameters
    are frozen and used for sampling.

    Args:
        log_density: Callable returning the target log density at a single
            point (this repo's `logpi_single` convention).
        initial_states: Starting states, one per inverse-temperature level
            (shape `(len(initial_inverse_temperature_schedule), d)`).
        initial_inverse_temperature_schedule: Strictly decreasing inverse
            temperatures with first entry equal to ``1.0``.
        num_iterations: Total number of replica-exchange iterations.
        burnin: Number of initial iterations to discard and use for adaptation.
        thinning: Keep every ``thinning``-th post-burnin iteration.
        initial_step_size: Initial scalar MALA step size, or one positive step
            size per inverse-temperature level.
        num_mala_steps: Number of MALA steps per replica before swap sweeps.
        rng_key: Optional JAX PRNG key.
        target_swap_acceptance: Target acceptance probability for adjacent swaps.
        target_mala_acceptance: Target MALA acceptance probability.
        gamma_temperature_base: Multiplier for temperature adaptation rates.
        gamma_mala_base: Multiplier for MALA step-size adaptation rates.
        gamma_temperature_power: Decay power for temperature adaptation rates.
        gamma_mala_power: Decay power for MALA adaptation rates.
        rho_bounds: Projection interval for temperature log-gap parameters.
        log_step_size_bounds: Projection interval for MALA log step sizes.
        store_history: Include tuning histories in diagnostics when true.
        use_deo: If ``True`` (default), use deterministic even-odd (DEO) swap
            sweeps -- both an even and an odd pass every iteration ("APT-DEO").
            If ``False``, use vanilla PT: one randomly chosen adjacent pair is
            proposed per iteration ("APT").
        discard_burnin: If ``True`` (default), the first ``burnin`` samples are
            dropped from the returned array. Set to ``False`` to keep the full
            chain (including the adaptation phase) while still adapting during
            the first ``burnin`` iterations.
        ref_log_density: Optional reference log density ``log b`` (single
            point). If given, replica ``beta`` targets
            ``(1-beta) log b + beta log_density`` instead of
            ``beta log_density``, and ``ref_sampler`` (``key -> (d,)`` exact
            draw from ``b``) must be given too; it refreshes the hottest
            replica by independence MH every iteration.

    Returns:
        samples: Samples from the cold replica after burnin and thinning.
        diagnostics: Final tuning parameters and acceptance diagnostics.
    """
    inverse_temperatures = [float(beta) for beta in initial_inverse_temperature_schedule]
    if len(inverse_temperatures) < 2:
        raise ValueError("initial_inverse_temperature_schedule must contain at least two levels.")
    if not jnp.isclose(inverse_temperatures[0], 1.0):
        raise ValueError("initial_inverse_temperature_schedule must start with 1.0.")
    if any(beta <= 0.0 for beta in inverse_temperatures):
        raise ValueError("inverse temperatures must be positive.")
    if any(
        beta_next >= beta
        for beta, beta_next in zip(inverse_temperatures[:-1], inverse_temperatures[1:])
    ):
        raise ValueError("initial_inverse_temperature_schedule must be strictly decreasing.")
    if num_iterations <= 0:
        raise ValueError("num_iterations must be positive.")
    if burnin < 0 or burnin >= num_iterations:
        raise ValueError("burnin must satisfy 0 <= burnin < num_iterations.")
    if thinning <= 0:
        raise ValueError("thinning must be positive.")
    if num_mala_steps <= 0:
        raise ValueError("num_mala_steps must be positive.")
    if not 0.0 < target_swap_acceptance < 1.0:
        raise ValueError("target_swap_acceptance must lie in (0, 1).")
    if not 0.0 < target_mala_acceptance < 1.0:
        raise ValueError("target_mala_acceptance must lie in (0, 1).")
    if gamma_temperature_base <= 0.0:
        raise ValueError("gamma_temperature_base must be positive.")
    if gamma_mala_base <= 0.0:
        raise ValueError("gamma_mala_base must be positive.")
    if gamma_temperature_power <= 0.0:
        raise ValueError("gamma_temperature_power must be positive.")
    if gamma_mala_power <= 0.0:
        raise ValueError("gamma_mala_power must be positive.")

    if (ref_log_density is None) != (ref_sampler is None):
        raise ValueError("ref_log_density and ref_sampler must be given together.")

    rho_min, rho_max = _validate_bounds("rho_bounds", rho_bounds)
    log_step_size_min, log_step_size_max = _validate_bounds("log_step_size_bounds", log_step_size_bounds)

    if rng_key is None:
        rng_key = random.PRNGKey(0)

    x0 = jnp.asarray(initial_states)
    if x0.ndim == 0:
        raise ValueError("initial_states must have a leading replica dimension.")
    num_temperatures = len(inverse_temperatures)
    if x0.shape[0] != num_temperatures:
        raise ValueError("initial_states must provide one state per inverse-temperature level.")

    tuning_dtype = jnp.result_type(x0, jnp.float32)
    step_sizes = jnp.asarray(initial_step_size, dtype=tuning_dtype)
    if step_sizes.ndim == 0:
        step_sizes = jnp.full((num_temperatures,), step_sizes, dtype=tuning_dtype)
    elif step_sizes.shape != (num_temperatures,):
        raise ValueError(
            "initial_step_size must be a scalar or provide one value per inverse-temperature level."
        )
    if bool(jnp.any(step_sizes <= 0.0)):
        raise ValueError("initial_step_size values must be positive.")

    inverse_temperatures_array = jnp.asarray(inverse_temperatures, dtype=tuning_dtype)
    ratios = inverse_temperatures_array[1:] / inverse_temperatures_array[:-1]
    initial_rho = jnp.log(-jnp.log(ratios))
    initial_rho = jnp.clip(initial_rho, rho_min, rho_max)
    initial_log_step_sizes = jnp.clip(jnp.log(step_sizes), log_step_size_min, log_step_size_max)

    base_logp0 = vmap(log_density)(x0)
    if ref_log_density is not None:
        base_logp0 = base_logp0 - vmap(ref_log_density)(x0)
    (
        cold_states, swap_accepts, swap_attempts, mala_accept_rates,
        inverse_temperature_history, step_size_history,
        final_inverse_temperatures, final_step_sizes, refresh_accepts,
    ) = _scan_adaptive_parallel_tempering(
        log_density, x0, base_logp0, initial_rho, initial_log_step_sizes,
        num_iterations, burnin, num_mala_steps, rng_key,
        target_swap_acceptance, target_mala_acceptance,
        gamma_temperature_base, gamma_mala_base, gamma_temperature_power, gamma_mala_power,
        rho_min, rho_max, log_step_size_min, log_step_size_max, use_deo,
        int(burnin if discard_burnin else 0), int(thinning),
        ref_log_density, ref_sampler,
    )

    samples = cold_states  # already the kept states: iterations start, start + thinning, ...
    total_swap_accepts = jnp.sum(swap_accepts)
    total_swap_attempts = jnp.sum(swap_attempts)
    pair_swap_attempts = jnp.sum(swap_attempts, axis=0)
    pair_swap_accepts = jnp.sum(swap_accepts, axis=0)

    diagnostics = {
        "final_inverse_temperatures": final_inverse_temperatures,
        "final_step_sizes": final_step_sizes,
        "swap_acceptance_rate": total_swap_accepts / total_swap_attempts,
        "swap_accepts_by_pair": pair_swap_accepts,
        "swap_attempts_by_pair": pair_swap_attempts,
        "swap_acceptance_rate_by_pair": jnp.where(
            pair_swap_attempts > 0.0, pair_swap_accepts / pair_swap_attempts, 0.0,
        ),
        "mala_acceptance_rate_by_temperature": jnp.mean(mala_accept_rates, axis=0),
    }
    if ref_log_density is not None:
        diagnostics["ref_refresh_acceptance_rate"] = jnp.mean(refresh_accepts)
    if store_history:
        diagnostics["inverse_temperature_history"] = inverse_temperature_history
        diagnostics["step_size_history"] = step_size_history

    return samples, diagnostics


# ---------------------------------------------------------------------------
# Convenience wrapper with the other baselines' call signature.
# ---------------------------------------------------------------------------

@dataclass
class APTRunResult:
    samples: Array                 # (n_kept, d) -- post-burnin, thinned cold-replica states
    use_deo: bool
    betas: Sequence[float]         # this repo's hot->cold betas, as passed in
    diagnostics: Dict[str, Any]    # `adaptive_parallel_tempering`'s diagnostics dict (jnp arrays -> python where noted)


def run_adaptive_pt(
    key: Array,
    logpi_single: LogDensity,
    d: int,
    *,
    betas: Sequence[float],
    use_deo: bool = True,
    n_iterations: int = 6000,
    burnin: int = 2000,
    thinning: int = 2,
    initial_step_size: float = 0.3,
    num_mala_steps: int = 1,
    x0_scale: float = 3.0,
    target_swap_acceptance: float = 0.234,
    target_mala_acceptance: float = 0.57,
    modes: Optional[Sequence[Any]] = None,
    mode_base_inflate: float = 1.0,
    dtype=jnp.float64,
    verbose: bool = False,
) -> APTRunResult:
    """Wrapper: reverses the hot->cold increasing `betas` ladder into the
    strictly-decreasing-from-1.0 schedule `adaptive_parallel_tempering`
    expects, draws diffuse per-level initial states, and runs one long
    single-chain trajectory (as the algorithm intends -- one walker
    per rung, not a population of chains).

    `modes` (a ModeExplore inventory) draws the initial states from the
    Laplace mixture `b` instead of `N(0, x0_scale^2 I)`; the ladder is
    still pi^beta -- see module docstring.
    """
    betas = [float(b) for b in betas]
    inv_temp_schedule = list(reversed(betas))
    K = len(inv_temp_schedule)

    key_x0, key_run = jax.random.split(key)
    mode_base = build_mode_base(modes, mode_base_inflate, dtype)
    if mode_base is None:
        x0 = float(x0_scale) * jax.random.normal(key_x0, (K, int(d)), dtype=dtype)
    else:
        if mode_base.dim != int(d):
            raise ValueError(f"modes have dimension {mode_base.dim}, expected d={d}.")
        x0 = mode_base.sample(key_x0, K)

    samples, diagnostics = adaptive_parallel_tempering(
        logpi_single, x0, inv_temp_schedule,
        num_iterations=int(n_iterations), burnin=int(burnin), thinning=int(thinning),
        initial_step_size=float(initial_step_size), num_mala_steps=int(num_mala_steps),
        rng_key=key_run, target_swap_acceptance=target_swap_acceptance,
        target_mala_acceptance=target_mala_acceptance, use_deo=use_deo,
    )

    diagnostics_py: Dict[str, Any] = {
        "swap_acceptance_rate": float(diagnostics["swap_acceptance_rate"]),
        "swap_acceptance_rate_by_pair": [float(x) for x in diagnostics["swap_acceptance_rate_by_pair"]],
        "mala_acceptance_rate_by_temperature": [float(x) for x in diagnostics["mala_acceptance_rate_by_temperature"]],
        "final_inverse_temperatures": [float(x) for x in diagnostics["final_inverse_temperatures"]],
        "final_step_sizes": [float(x) for x in diagnostics["final_step_sizes"]],
        "n_kept": int(samples.shape[0]),
    }
    if mode_base is not None:
        diagnostics_py["n_modes"] = mode_base.n_modes
    if verbose:
        tag = "adaptive_pt_deo" if use_deo else "adaptive_pt"
        print(
            f"[{tag}] rungs={K} iters={n_iterations} kept={diagnostics_py['n_kept']} | "
            f"swap_acc={diagnostics_py['swap_acceptance_rate']:.3f} | "
            f"mala_acc_mean={sum(diagnostics_py['mala_acceptance_rate_by_temperature'])/K:.3f}"
        )

    return APTRunResult(samples=samples, use_deo=use_deo, betas=betas, diagnostics=diagnostics_py)
