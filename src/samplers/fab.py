# fab.py
"""FAB: Flow Annealed importance sampling Bootstrap (Midgley, Stimper, Simm, Scholkopf &
Hernandez-Lobato, ICLR 2023; arXiv:2208.01893).

Runs the official implementation, `github.com/lollcat/fab-jax`, vendored (and patched for
current JAX/blackjax) into `_fabjax/`; see `_fabjax/__init__.py` for what is vendored and
the patch list. The flow (`_fabjax.flow.build_flow`), the SMC/AIS ladder with an HMC
transition operator and dual-averaging step-size adaptation (`_fabjax.sampling`), the
prioritised replay buffer (`_fabjax.buffer`) and the buffer-based training loop with the
alpha = 2 divergence loss (`_fabjax.train.fab_with_buffer`) are fab-jax's own code; this
module wires them to `logpi` and runs the outer round loop.

Per training round:
  1. sample a minibatch from the prioritised buffer, take `n_updates_per_smc_forward_pass`
     Adam steps on the alpha-divergence buffer loss, and write the recomputed log
     densities back into the buffer;
  2. run one SMC/AIS forward pass from the updated flow toward pi through
     `n_intermediate_distributions` HMC-transitioned rungs and push the result into the
     buffer.

`transform_type='real_nvp'` with a fixed number of intermediate distributions and HMC
steps, sized for the problems here.

**Mode-informed variant** (`modes=...`): the flow's N(0, I) base is replaced by the Laplace
mixture b of the discovered modes (`_mode_base.LaplaceMixtureBase`, wrapped as a distrax
base through the flow recipe's `make_base`). fab-jax's flow is identity-initialized, so
training starts from q = b with every discovered mode covered. `modes=None` is the N(0, I)
base.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jr
import optax

from ._fabjax.flow.build_flow import FlowDistConfig, create_flow_recipe
from ._fabjax.flow.flow import create_flow
from ._fabjax.sampling.smc import build_smc
from ._fabjax.sampling.mcmc.hmc import build_blackjax_hmc
from ._fabjax.buffer.prioritised_buffer import build_prioritised_buffer
from ._fabjax.train.fab_with_buffer import build_fab_with_buffer_init_step_fns
from ._mode_base import DistraxLikeMixture, build_mode_base

Array = jax.Array
LogPiBatch = Callable[[Array], Array]   # (N,d) -> (N,)
LogPiSingle = Callable[[Array], Array]  # (d,) -> scalar


@dataclass
class FABResult:
    params: Any
    flow_model: Any                    # fab-jax `Flow` namedtuple (sample_apply/log_prob_apply/...)
    samples: Array
    q_log_prob: Callable[[Array], Array]
    loss_history: List[float]
    ess_history: List[float]
    info: Dict[str, Any]


def _build_round_runner(step_fn, chunk_size: int):
    """Run sequential FAB rounds without transferring diagnostics each round.

    The final partial chunk skips inactive rounds, including all RNG, HMC,
    optimizer and replay-buffer updates. Only scalar histories are stacked.
    """
    @jax.jit
    def run(state, n_active):
        def advance(state):
            state, info = step_fn(state)
            return state, (info["fab_loss"], info["ess_smc_final"])

        diagnostic_shapes = jax.eval_shape(advance, state)[1]

        def body(state, i):
            # Match the diagnostic dtypes without executing an extra round.
            def skip(state):
                return state, jax.tree.map(
                    lambda a: jnp.zeros(a.shape, a.dtype), diagnostic_shapes
                )

            return jax.lax.cond(i < n_active, advance, skip, state)

        return jax.lax.scan(body, state, jnp.arange(chunk_size))

    return run


def fab_train(
    key: Array,
    logpi_batch: LogPiBatch,  # unused (fab-jax vmaps `logpi_single` internally); kept
                              # for calling-convention parity with `temperflow_fit`/
                              # `annealing_flow_fit`'s `(key, logpi, logpi_single, d)`.
    logpi_single: LogPiSingle,
    d: int,
    *,
    n_rounds: int = 300,
    n_ais: int = 256,
    n_intermediate_distributions: int = 4,
    spacing_type: str = "linear",
    hmc_n_outer_steps: int = 3,
    hmc_n_inner_steps: int = 5,
    hmc_init_step_size: float = 0.3,
    hmc_adapt_step_size: bool = True,
    hmc_target_p_accept: float = 0.65,
    buffer_size: int = 8192,
    batch_size: int = 256,
    n_updates_per_smc_forward_pass: int = 4,
    alpha: float = 2.0,
    w_adjust_clip: float = 10.0,
    n_coupling: int = 6,
    hidden: int = 128,
    depth: int = 3,
    transform_type: str = "real_nvp",
    lr: float = 1e-3,
    grad_clip: float = 5.0,
    n_snapshot: Optional[int] = None,
    modes: Optional[Sequence[Any]] = None,
    mode_base_inflate: float = 1.0,
    dtype=jnp.float64,
    verbose: bool = False,
    round_chunk_size: int = 32,
) -> FABResult:
    """Train FAB with sequential rounds grouped into compiled chunks.

    ``round_chunk_size`` only controls dispatch and diagnostic transfers;
    it does not change the updates, RNG sequence or number of rounds.
    Verbose progress is flushed at chunk boundaries.
    ``modes`` (a ModeExplore inventory) replaces the flow's N(0, I) base
    with the Laplace mixture `b` -- see module docstring.
    """
    if d < 2:
        raise ValueError("fab-jax's flow requires dim >= 2.")
    if round_chunk_size < 1:
        raise ValueError("round_chunk_size must be positive.")
    min_length_to_sample = int(batch_size) * int(n_updates_per_smc_forward_pass)
    if n_ais < min_length_to_sample:
        raise ValueError(
            "n_ais must be >= batch_size * n_updates_per_smc_forward_pass "
            "(the buffer's min_length_to_sample, filled by AIS passes of "
            "n_ais samples each)."
        )

    flow_config = FlowDistConfig(
        dim=d, n_layers=n_coupling, conditioner_mlp_units=(hidden,) * depth,
        transform_type=transform_type,
    )
    recipe = create_flow_recipe(flow_config)
    mode_base = build_mode_base(modes, mode_base_inflate, dtype)
    if mode_base is not None:
        if mode_base.dim != d:
            raise ValueError(f"modes have dimension {mode_base.dim}, expected d={d}.")
        recipe = recipe._replace(make_base=lambda: DistraxLikeMixture(mode_base))
    flow = create_flow(recipe)

    hmc = build_blackjax_hmc(
        dim=d, n_outer_steps=hmc_n_outer_steps, n_inner_steps=hmc_n_inner_steps,
        init_step_size=hmc_init_step_size, adapt_step_size=hmc_adapt_step_size,
        target_p_accept=hmc_target_p_accept,
    )
    smc = build_smc(
        transition_operator=hmc, n_intermediate_distributions=n_intermediate_distributions,
        spacing_type=spacing_type, alpha=alpha, use_resampling=False,
    )

    buffer = build_prioritised_buffer(
        dim=d, max_length=int(buffer_size), min_length_to_sample=min_length_to_sample,
    )

    optimizer = optax.chain(optax.clip_by_global_norm(grad_clip), optax.adam(lr))

    init_fn, step_fn = build_fab_with_buffer_init_step_fns(
        flow=flow, log_p_fn=logpi_single, smc=smc, buffer=buffer, optimizer=optimizer,
        batch_size=int(batch_size), n_updates_per_smc_forward_pass=int(n_updates_per_smc_forward_pass),
        alpha=alpha, w_adjust_clip=w_adjust_clip,
    )
    chunk_size = min(int(round_chunk_size), max(1, int(n_rounds)))
    run_rounds = _build_round_runner(step_fn, chunk_size)

    key, k_init = jr.split(key)
    state = init_fn(k_init)

    loss_hist: List[float] = []
    ess_hist: List[float] = []

    for start in range(0, int(n_rounds), chunk_size):
        n_active = min(chunk_size, int(n_rounds) - start)
        state, history = run_rounds(state, jnp.asarray(n_active, dtype=jnp.int32))
        losses, esses = jax.device_get(history)
        for offset in range(n_active):
            r = start + offset
            loss = float(losses[offset])
            ess = float(esses[offset]) / float(n_ais)
            loss_hist.append(loss)
            ess_hist.append(ess)
            if verbose and (r % max(1, int(n_rounds) // 10) == 0 or r == int(n_rounds) - 1):
                print(f"[fab] round={r:04d} | loss={loss:.4e} | AIS_ESS_ratio={ess:.3f}")

    if n_snapshot is None:
        n_snapshot = int(n_ais)
    key, k_snap = jr.split(key)
    samples = flow.sample_apply(state.flow_params, k_snap, (int(n_snapshot),))

    final_params = state.flow_params

    def q_log_prob(x):
        return flow.log_prob_apply(final_params, jnp.asarray(x, dtype=dtype))

    info = {
        "n_rounds": int(n_rounds),
        "n_ais": int(n_ais),
        "n_intermediate_distributions": int(n_intermediate_distributions),
        "alpha": float(alpha),
        "w_adjust_clip": float(w_adjust_clip),
        "buffer_is_full": bool(state.buffer_state.is_full),
        "final_loss": loss_hist[-1] if loss_hist else float("nan"),
        "final_ais_ess_ratio": ess_hist[-1] if ess_hist else float("nan"),
        "n_modes": 0 if mode_base is None else mode_base.n_modes,
    }

    return FABResult(
        params=final_params,
        flow_model=flow,
        samples=samples,
        q_log_prob=q_log_prob,
        loss_history=loss_hist,
        ess_history=ess_hist,
        info=info,
    )
