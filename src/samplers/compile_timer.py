"""src/samplers/compile_timer.py

Wall time spent in JAX compilation -- tracing, lowering to MLIR and the XLA backend
compile -- from JAX's own monitoring time spans, recorded from import on. Nested jits
emit nested trace spans, so overlapping spans are merged before summing.
"""
from __future__ import annotations

from typing import Iterable, List, Tuple

from jax._src import monitoring

_EVENTS = frozenset({
    "/jax/core/compile/jaxpr_trace_duration",
    "/jax/core/compile/jaxpr_to_mlir_module_duration",
    "/jax/core/compile/backend_compile_duration",
})
_SPANS: List[Tuple[float, float]] = []


def _listener(event, start, end, **kwargs):
    if event in _EVENTS:
        _SPANS.append((start, end))


monitoring.register_event_time_span_listener(_listener)


def compile_seconds(t0: float, t1: float, exclude: Iterable[Tuple[float, float]] = ()) -> float:
    """Compile wall time within [t0, t1] (time.time() stamps), minus the windows in
    `exclude` (e.g. evaluation callbacks, whose time is not charged to the method)."""
    cuts = sorted((max(a, t0), min(b, t1)) for a, b in exclude if b > t0 and a < t1)
    total, cur_a, cur_b = 0.0, None, None
    for a, b in sorted((max(a, t0), min(b, t1)) for a, b in _SPANS if b > t0 and a < t1):
        if cur_b is None or a > cur_b:
            if cur_b is not None:
                total += cur_b - cur_a - _overlap(cur_a, cur_b, cuts)
            cur_a, cur_b = a, b
        else:
            cur_b = max(cur_b, b)
    if cur_b is not None:
        total += cur_b - cur_a - _overlap(cur_a, cur_b, cuts)
    return total


def _overlap(a: float, b: float, cuts: List[Tuple[float, float]]) -> float:
    return sum(max(0.0, min(b, y) - max(a, x)) for x, y in cuts)
