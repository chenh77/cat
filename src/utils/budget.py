"""Nominal target-oracle accounting, used to match every method's budget to CAT's.

The counters are intentionally transparent rather than profiler-derived.  A
``value`` is a standalone ``log pi`` evaluation, ``value_grad`` is a gradient
or joint value/gradient evaluation (its primal evaluation is *not* counted a
second time), and ``hessian`` is one dense full-Hessian evaluation.  Counts
are scalar-particle counts, even when the implementation vmaps a batch.

The experiments weight transport costs as value + dim * value_grad
(`transport_weighted`); the only Hessian calls are in the shared mode discovery.
This is a budget-allocation rule, not a claim about universal runtimes.  Raw
triplets are always reported beside the combined number.

ModeExplore's LBFGS count is necessarily nominal: JAXopt's data-dependent
line-search evaluations are not exposed by the current result object.  We
charge one value/gradient call per configured LBFGS iteration.  The fixed
profile-search loops, MCMC kernels, training batches, and diagnostics are
counted according to the operations in this repository.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Dict, Optional


@dataclass(frozen=True)
class OracleCost:
    value: int = 0
    value_grad: int = 0
    hessian: int = 0

    def __add__(self, other: "OracleCost") -> "OracleCost":
        if not isinstance(other, OracleCost):
            return NotImplemented
        return OracleCost(
            self.value + other.value,
            self.value_grad + other.value_grad,
            self.hessian + other.hessian,
        )

    def __sub__(self, other: "OracleCost") -> "OracleCost":
        if not isinstance(other, OracleCost):
            return NotImplemented
        return OracleCost(
            self.value - other.value,
            self.value_grad - other.value_grad,
            self.hessian - other.hessian,
        )

    def __mul__(self, factor: int) -> "OracleCost":
        factor = int(factor)
        return OracleCost(
            factor * self.value,
            factor * self.value_grad,
            factor * self.hessian,
        )

    __rmul__ = __mul__

    def weighted(self, dim: int, *, grad_weight: int = 3, hessian_grad_units: int = 3) -> int:
        return int(
            self.value
            + int(grad_weight) * self.value_grad
            + int(hessian_grad_units) * int(dim) * self.hessian
        )

    def triplet(self) -> tuple[int, int, int]:
        return self.value, self.value_grad, self.hessian


def transport_weighted(cost: OracleCost, dim: int) -> int:
    """value + dim * value_grad: the run scripts' convention for transport
    costs (shared mode discovery excluded), which make no Hessian calls."""
    if cost.hessian != 0:
        raise ValueError(f"transport cost has {cost.hessian} Hessian calls; "
                         "subtract the shared mode-discovery cost first")
    return int(cost.value + int(dim) * cost.value_grad)


def mode_explore_cost(
    *,
    dim: int,
    n: int,
    v: int,
    optimize_steps: int,
    max_lbfgs_particles: int | None,
    rounds: int = 1,
    kernel: str = "mala",
    cov_method: str = "hessian",
    final_modes: int = 0,
) -> OracleCost:
    """Configured/nominal cost of one or more ModeExplore rounds."""
    dim, n, v, rounds = int(dim), int(n), int(v), int(rounds)
    n_lbfgs = n if max_lbfgs_particles is None else min(n, int(max_lbfgs_particles))
    kernel = kernel.lower()

    if kernel == "ula":
        warm = OracleCost(value_grad=n * v)
    elif kernel == "mala":
        # sampling.proposals.MALA separately evaluates value and score at
        # both the current point and proposal.
        warm = OracleCost(value=2 * n * v, value_grad=2 * n * v)
    elif kernel in ("rwmh", "indep"):
        warm = OracleCost(value=2 * n * v)
    else:
        raise ValueError(f"Unsupported ModeExplore kernel: {kernel!r}")

    # One nominal value/gradient evaluation per configured LBFGS iteration;
    # one explicit finite-value check and one dense Hessian per candidate.
    local = OracleCost(
        value=n_lbfgs,
        value_grad=n_lbfgs * int(optimize_steps),
        hessian=n_lbfgs,
    )

    if cov_method in ("profile", "profile_max"):
        # Per candidate: log pi(mu), then two d-wide vectorized searches.
        # Each side performs 1 initial + 30 expansion + 60 bisection probes.
        local += OracleCost(value=n_lbfgs * (1 + 2 * dim * (1 + 30 + 60)))
    elif cov_method != "hessian":
        raise ValueError(f"Unsupported covariance method: {cov_method!r}")

    # recompute_mode_weights evaluates each surviving mode once per round.
    weights = OracleCost(value=int(final_modes))
    return rounds * (warm + local + weights)


def cat_cost(
    *,
    n: int,
    n_train: int,
    n_validation: int,
    n_modes: int,
    n_betas: int,
    n_refresh: int,
    validate_every: int = 1,
    final_calibration: Optional[int] = None,
    adaptive_grid: bool = False,
    discovery: OracleCost = OracleCost(),
) -> OracleCost:
    """Nominal `cat` cost; mirrors its `n_target_evals` /
    `n_target_grad_evals` counters. `n_betas` counts the checkpoints
    including beta=0. Network optimizer steps and divergences are target-free.
    With `adaptive_grid` (ESS-adaptive schedule) n_betas is the *expected* grid
    size (the realized one is an output), and the first refresh, whose field is
    the identity flow, evaluates h once per particle instead of at K checkpoints.
    """
    j, k, n_train = int(n_modes), int(n_betas), int(n_train)
    n_refresh, validate_every = int(n_refresh), int(validate_every)
    n_val_rounds = sum(
        1 for r in range(n_refresh) if (r + 1) % validate_every == 0 or r == n_refresh - 1
    )
    # Per refresh: rollout values, and the score of log pi at every collocation point.
    per_refresh = OracleCost(value=k * j * n_train, value_grad=k * j * n_train)
    per_val = OracleCost(value=k * j * int(n_validation))
    # Endpoint values: all n final particles, or m per mode with final_calibration.
    m_final = int(final_calibration) if final_calibration is not None and j > 1 else int(n)
    final = OracleCost(value=j * m_final)
    total = discovery + n_refresh * per_refresh + n_val_rounds * per_val + final
    if adaptive_grid:
        total = OracleCost(value=total.value - (k - 1) * j * n_train, value_grad=total.value_grad,
                           hessian=total.hessian)
    return total


def apt_cost(*, n_temperatures: int, n_iterations: int, num_mala_steps: int = 1) -> OracleCost:
    k = int(n_temperatures)
    it = int(n_iterations)
    s = int(num_mala_steps)
    # Initial base values; per iteration/replica run_mala_steps evaluates an
    # initial score, then one value and score for each MALA proposal.
    return OracleCost(value=k + it * k * s, value_grad=it * k * (s + 1))


def alps_cost(*, n_target_evals: int, n_target_grad_evals: int = 0,
              discovery: OracleCost = OracleCost()) -> OracleCost:
    # ALPS sizes its own run from a weighted budget (its ladder is only known after
    # its beta_max pilot) and reports its log pi evaluations (`alps_eval_count`)
    # and the gradient calls of its Hessians (d per Hessian).
    return discovery + OracleCost(value=int(n_target_evals), value_grad=int(n_target_grad_evals))


def fab_cost(
    *,
    batch_size: int,
    n_rounds: int,
    n_intermediate_distributions: int,
    hmc_n_outer_steps: int,
    hmc_n_inner_steps: int,
    n_updates_per_smc_forward_pass: int,
) -> OracleCost:
    b = int(batch_size)
    per_pass = b * (
        1
        + int(n_intermediate_distributions)
        * int(hmc_n_outer_steps)
        * int(hmc_n_inner_steps)
    )
    # Buffer initialization runs n_updates+1 SMC passes; each training round
    # runs one further pass. Buffer-gradient updates themselves are target-free.
    passes = int(n_updates_per_smc_forward_pass) + 1 + int(n_rounds)
    return OracleCost(value_grad=passes * per_pass)


def temperflow_cost(*, n: int, n_train: int, n_probe: int, n_stages: int, train_steps: int,
                    topup_steps: int = 0) -> OracleCost:
    k = int(n_stages)
    return OracleCost(
        value=k * int(n) + max(0, k - 1) * int(n_probe),
        value_grad=(k * int(train_steps) + int(topup_steps)) * int(n_train),
    )


def aft_cost(
    *,
    n: int,
    n_stages: int,
    train_steps: int,
    mala_steps: int,
    mala_steps_resample: int | None = None,
    n_resampled: int = 0,
    total_retries: int = 0,
    val_frac: float = 0.2,
) -> OracleCost:
    n = int(n)
    k = int(n_stages)
    attempts = k + int(total_retries)
    # `aft.py` holds out a `val_frac` slice of the n working particles for
    # per-step early-stopping (Arbel et al.'s Algorithm 2/3 train/validation
    # split) -- mirrors that file's own n_val/n_train split so this cost
    # model matches what actually runs. Every training attempt evaluates
    # log pi(x) (value-only) and log pi(Tx) through the differentiated map
    # on the n_train split, plus one further value-only free-energy check
    # (log pi(x_val) and log pi(T(x_val)), both undifferentiated) on the
    # disjoint n_val split every step. Each accepted stage adds
    # current/checked values on the full n-particle population and a
    # run_mala_steps call with one initial plus mala_steps pairs -- at
    # `mala_steps_resample` steps (default: same as `mala_steps`, i.e. no
    # difference) for the `n_resampled` stages that resampled, and
    # `mala_steps` for the rest (see aft_fit's own per-stage split).
    n_val = max(1, min(n - 1, round(n * val_frac)))
    n_train = n - n_val
    steps = int(train_steps)
    m_resample = int(mala_steps) if mala_steps_resample is None else int(mala_steps_resample)
    n_resampled = int(n_resampled)
    n_plain = k - n_resampled
    mala_value = n_plain * n * (2 + int(mala_steps)) + n_resampled * n * (2 + m_resample)
    mala_value_grad = n_plain * n * (1 + int(mala_steps)) + n_resampled * n * (1 + m_resample)
    return OracleCost(
        value=(attempts * steps * n_train + 2 * attempts * steps * n_val
               + attempts * n + mala_value),
        value_grad=attempts * steps * n_train + mala_value_grad,
    )


def repetitions_at_least(
    target_weighted: int,
    *,
    dim: int,
    cost_for_repetitions: Callable[[int], OracleCost],
    minimum: int = 1,
    grad_weight: int = 3,
    hessian_grad_units: int = 3,
) -> int:
    """Smallest integer repeat count whose nominal cost reaches the target."""
    target_weighted = int(target_weighted)
    minimum = int(minimum)
    c0 = cost_for_repetitions(0).weighted(dim, grad_weight=grad_weight, hessian_grad_units=hessian_grad_units)
    c1 = cost_for_repetitions(1).weighted(dim, grad_weight=grad_weight, hessian_grad_units=hessian_grad_units)
    increment = c1 - c0
    if increment <= 0:
        raise ValueError("Cost must increase with repetitions.")
    return max(minimum, int(math.ceil((target_weighted - c0) / increment)))


def matched_baseline_config(
    reference: OracleCost,
    *,
    dim: int,
    n_temperatures: int,
    n_chains: int,
    n: int,
    n_train: int,
    n_probe: int,
    pilot_temperflow_stages: int,
    pilot_aft_stages: int,
    fab_batch_size: int,
    fab_n_intermediate_distributions: int,
    fab_hmc_n_outer_steps: int,
    fab_hmc_n_inner_steps: int,
    fab_n_updates: int,
    aft_mala_steps: int,
    alps_discovery: OracleCost = OracleCost(),
    aft_n: int | None = None,
    pilot_aft_resampled: int = 0,
    margin: float = 1.0,
    method_margins: Dict[str, float] | None = None,
    grad_weight: int = 3,
    hessian_grad_units: int = 3,
) -> Dict[str, int]:
    """Choose baseline repeat counts with nominal cost at least `reference`'s
    (the budget standard's cost: CAT's transport cost in the run scripts).

    Adaptive-flow stage counts are pilot values from the unbudgeted run.  The
    run scripts recompute their realized costs using the actual
    stage counts, which for target-ESS-adaptive baselines (TemperFlow, AFT)
    can land well below the pilot count and undercut the reference cost.
    `margin` (>1.0) inflates the pre-run target to build in slack against
    that shortfall; callers wanting the reference to always be the cheapest method
    should pass a margin comfortably above the largest shortfall they've
    observed (a factor of `pilot_stages / realized_stages`).

    `method_margins` overrides `margin` per baseline (keys: "apt", "alps",
    "fab", "temperflow", "aft"; any key omitted falls back
    to `margin`). Useful because the shortfall `margin` guards against is
    specific to target-ESS-adaptive methods (TemperFlow, AFT) whose realized
    stage count can undershoot their pilot -- APT/APT-DEO, ALPS, and FAB have
    no such shortfall (their repeat count is exact, not adaptive), so giving
    them the same >1 margin as TemperFlow/AFT hands them extra nominal
    budget they don't need it for, which is not really a cost-matched
    comparison for those methods. Pass e.g. `{"apt": 1.0, "alps": 1.0, "fab":
    1.0}` with `margin=1.5` to keep the shortfall slack
    only for TemperFlow/AFT.

    `aft_n` lets AFT's population size (and so its final resampled particle
    count) differ from the shared `n` used by the other baselines' internal
    batches -- e.g. to match AFT's final sample count to the reference method's, without
    changing FAB/TemperFlow's batch size. Defaults to `n`.

    `pilot_aft_resampled` is how many of AFT's pilot stages resampled: those run
    `max(4 * aft_mala_steps, 20)` MALA steps (aft_fit's default) instead of
    `aft_mala_steps`, which dominates AFT's cost at small budgets.

    `grad_weight`/`hessian_grad_units` set the same nominal conversion used
    throughout (see `OracleCost.weighted`); pass e.g. `grad_weight=dim,
    hessian_grad_units=dim` for the `1, dim, dim**2` convention.
    """
    _aft_n = int(n if aft_n is None else aft_n)
    _method_margins = method_margins or {}
    _ref_weighted = reference.weighted(dim, grad_weight=grad_weight, hessian_grad_units=hessian_grad_units)

    def _target_for(name: str) -> int:
        m = float(_method_margins.get(name, margin))
        return int(math.ceil(m * _ref_weighted))

    target = _target_for("apt")  # kept as the informational default (all-methods) target
    _rep_kwargs = dict(dim=dim, grad_weight=grad_weight, hessian_grad_units=hessian_grad_units)

    apt_iterations = repetitions_at_least(
        _target_for("apt"),
        cost_for_repetitions=lambda it: apt_cost(
            n_temperatures=n_temperatures, n_iterations=it, num_mala_steps=1
        ),
        **_rep_kwargs,
    )
    # ALPS sizes its run from the weighted budget itself (`alps_sample(value_budget=...)`).
    alps_value_budget = _target_for("alps") - alps_discovery.weighted(
        dim, grad_weight=grad_weight, hessian_grad_units=hessian_grad_units)
    fab_rounds = repetitions_at_least(
        _target_for("fab"),
        cost_for_repetitions=lambda rounds: fab_cost(
            batch_size=fab_batch_size,
            n_rounds=rounds,
            n_intermediate_distributions=fab_n_intermediate_distributions,
            hmc_n_outer_steps=fab_hmc_n_outer_steps,
            hmc_n_inner_steps=fab_hmc_n_inner_steps,
            n_updates_per_smc_forward_pass=fab_n_updates,
        ),
        **_rep_kwargs,
    )
    temperflow_train_steps = repetitions_at_least(
        _target_for("temperflow"),
        cost_for_repetitions=lambda steps: temperflow_cost(
            n=n,
            n_train=n_train,
            n_probe=n_probe,
            n_stages=pilot_temperflow_stages,
            train_steps=steps,
        ),
        **_rep_kwargs,
    )
    aft_train_steps = repetitions_at_least(
        _target_for("aft"),
        cost_for_repetitions=lambda steps: aft_cost(
            n=_aft_n,
            n_stages=pilot_aft_stages,
            train_steps=steps,
            mala_steps=aft_mala_steps,
            mala_steps_resample=max(4 * int(aft_mala_steps), 20),
            n_resampled=pilot_aft_resampled,
        ),
        **_rep_kwargs,
    )
    return {
        "target_weighted": target,
        "apt_iterations": apt_iterations,
        "alps_value_budget": alps_value_budget,
        "fab_rounds": fab_rounds,
        "temperflow_train_steps": temperflow_train_steps,
        "aft_train_steps": aft_train_steps,
    }


def realized_costs(
    *,
    results: dict,
    config: dict,
    n_temperatures: int,
    n: int,
    n_train: int,
    n_probe: int,
    fab_batch_size: int,
    fab_n_intermediate_distributions: int,
    fab_hmc_n_outer_steps: int,
    fab_hmc_n_inner_steps: int,
    fab_n_updates: int,
    aft_mala_steps: int,
    alps_discovery: OracleCost = OracleCost(),
    alps_n_chains: int | None = None,
    aft_n: int | None = None,
) -> Dict[str, OracleCost]:
    """Recompute the nominal ledger with adaptive methods' realized stages.

    `alps_n_chains`/`aft_n` mirror the same-named knobs in
    `matched_baseline_config` -- pass the actual chain/population size used
    for the run so the realized ledger matches what was really computed,
    when it differs from the shared `n`. Both default to `n`.
    """
    _aft_n = int(n if aft_n is None else aft_n)
    out: Dict[str, OracleCost] = {}
    if results.get("APT") is not None:
        out["APT"] = apt_cost(
            n_temperatures=n_temperatures,
            n_iterations=config["apt_iterations"],
        )
    if results.get("APT-DEO") is not None:
        out["APT-DEO"] = apt_cost(
            n_temperatures=n_temperatures,
            n_iterations=config["apt_iterations"],
        )
    if results.get("ALPS") is not None:
        out["ALPS"] = alps_cost(n_target_evals=results["ALPS"].n_target_evals,
                                n_target_grad_evals=results["ALPS"].n_target_grad_evals,
                                discovery=alps_discovery)
    if results.get("FAB") is not None:
        out["FAB"] = fab_cost(
            batch_size=fab_batch_size,
            n_rounds=config["fab_rounds"],
            n_intermediate_distributions=fab_n_intermediate_distributions,
            hmc_n_outer_steps=fab_hmc_n_outer_steps,
            hmc_n_inner_steps=fab_hmc_n_inner_steps,
            n_updates_per_smc_forward_pass=fab_n_updates,
        )
    tf = results.get("TemperFlow")
    if tf is not None:
        out["TemperFlow"] = temperflow_cost(
            n=n,
            n_train=n_train,
            n_probe=n_probe,
            n_stages=len(tf.betas_used) - 1,
            train_steps=config["temperflow_train_steps"],
            topup_steps=getattr(tf, "topup_steps", 0),
        )
    aft = results.get("AFT")
    if aft is not None:
        out["AFT"] = aft_cost(
            n=_aft_n,
            n_stages=len(aft.betas_used) - 1,
            train_steps=config["aft_train_steps"],
            mala_steps=aft_mala_steps,
            mala_steps_resample=aft.info.get("mala_steps_resample"),
            n_resampled=aft.info.get("n_resampled", 0),
        )
    return out


def format_cost(label: str, cost: OracleCost, dim: int) -> str:
    v, g, h = cost.triplet()
    return (
        f"{label:<18} | value={v:>12,d} | value+score={g:>12,d} | "
        f"Hessian={h:>7,d} | weighted={cost.weighted(dim):>14,d}"
    )
