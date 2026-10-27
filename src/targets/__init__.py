# targets/__init__.py
"""Target distributions of the experiments.

Each target implements `log_prob(x)`: (N, d) or (d,) -> (N,) unnormalized log density, and
`dim`.

  - `SkewNormalMixture` -- mixture of skew-normals; `build_two_mode` gives the two-mode
                           target of the experiments (`skew_normal.py`).
  - `SkewTMixture`      -- two-mode skew-t mixture with matched Laplace approximations
                           (`skew_t_mixture.py`), the target of the path figures.
  - `SURModel`          -- seemingly unrelated regression profile posterior (`sur.py`).
  - `IohmmReg`          -- posteriordb `iohmm_reg_simulated-iohmm_reg`: input-output HMM
                           regression, K = 3 states (`iohmm.py`).
  - `LdaK2`             -- posteriordb `three_men1-ldaK2`: latent Dirichlet allocation with
                           K = 2 topics (`lda.py`).

The posteriordb targets are native JAX ports of the Stan models, in Stan's unconstrained
coordinates (`constraints.py` holds the constraining transforms).
"""
from __future__ import annotations

from .skew_normal import SkewNormalMixture, build_two_mode
from .skew_t_mixture import SkewTMixture
from .sur import SURModel
from .iohmm import IohmmReg
from .lda import LdaK2

__all__ = ["SkewNormalMixture", "build_two_mode", "SkewTMixture", "SURModel", "IohmmReg", "LdaK2"]
