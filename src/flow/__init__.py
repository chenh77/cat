# flow/__init__.py
"""Flow components.

  - `vector_field.py` -- the time-conditioned MLP vector field v(beta, x) of CAT, GAT and
                         the CNF (whitened inputs, time features, per-coordinate spline term).
  - `nets.py`         -- bounded RealNVP (`RealNVPBounded`) and the diagonal affine flow
                         (`DiagonalAffineBounded`) of the discrete-flow baselines.
  - `chain.py`        -- composition of RealNVP blocks into flow chains and mixtures of
                         transported base distributions.
"""
from __future__ import annotations

from .nets import (
    smoothstep5,
    bump_clip,
    MLP,
    MarginalRationalQuadraticSpline,
    AffineCouplingBounded,
    RealNVPBounded,
    odd_even_split_scheme,
    DiagonalAffineBounded,
)
from .chain import (
    stack_chain,
    chain_forward,
    chain_inverse,
    component_log_q,
    mixture_log_q,
    sample_mixture,
)

__all__ = [
    "smoothstep5",
    "bump_clip",
    "MLP",
    "MarginalRationalQuadraticSpline",
    "AffineCouplingBounded",
    "RealNVPBounded",
    "odd_even_split_scheme",
    "DiagonalAffineBounded",
    "stack_chain",
    "chain_forward",
    "chain_inverse",
    "component_log_q",
    "mixture_log_q",
    "sample_mixture",
]
