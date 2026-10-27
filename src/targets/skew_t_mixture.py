# targets/skew_t_mixture.py
"""Two-mode skew-t mixture in d dimensions with matched Laplace approximations.

In rotated coordinates z = R^T x (R a fixed random rotation),

    pi(z) = (1 - w_R) ST(z_1; -s/2, w1, -a1, nu) prod_{j>=2} N(z_j; 0, 1)                  (L)
          + w_R       ST(z_1; +s/2, w1, +a1, nu) prod_{j>=2} e^{-z_j^2/2 - c z_j^4} / Z_c   (R)

with ST(u; xi, w, a, nu) = (2/w) t_nu(y) T_{nu+1}(a y sqrt((nu + 1) / (nu + y^2))),
y = (u - xi) / w, the Azzalini skew-t. The z_1 factors are mirror images skewed outward
(their Laplace errors cancel), and c = `quartic_c(d)` makes the quartic transverse factor of R
lose exactly the mass that sets the true weights to (1 - w_R, w_R) while both unnormalized
components keep equal peaks and Hessians: the Laplace mixture b is 50/50. The separation s
(`gamma` in the figures) sets the barrier along z_1; the polynomial outward tails make the
geometric path b^{1-beta} pi^beta move the tail mass late.

The density is normalized, and exact iid draws are available (`reference`,
`component_draws`), as is the exact z_1 marginal (`z1_density`).
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from ..mode_explore.explore import Mode

W_R = 0.2


def quartic_c(d, w_R=W_R):
    """c with (int N(u) e^{-c u^4} du)^{d-1} = w_R / (1 - w_R)."""
    r_target = (w_R / (1 - w_R)) ** (1.0 / (d - 1))
    r = lambda c: quad(lambda u: np.exp(-0.5 * u * u - c * u ** 4), -np.inf, np.inf)[0] / np.sqrt(2 * np.pi)
    return brentq(lambda c: r(c) - r_target, 1e-12, 1e3)


def sn_mode(a):
    """Mode and curvature -d^2/du^2 log at the mode of the unit skew-normal 2 phi(u) Phi(a u)."""
    from scipy.stats import norm
    g = lambda u: -u + a * np.exp(norm.logpdf(a * u) - norm.logcdf(a * u))
    m = brentq(g, -5, 5) if a else 0.0
    r = np.exp(norm.logpdf(a * m) - norm.logcdf(a * m))     # phi/Phi at a m
    return m, 1 + a * a * r * (a * m + r)


def t_logcdf(x, df):
    """log of the Student-t CDF, differentiable (betainc; first-order Taylor where |x| < 1e-6,
    since the betainc gradient is NaN at x = 0)."""
    from jax.scipy.special import betainc, gammaln
    small = jnp.abs(x) < 1e-6
    xs = jnp.where(small, 1.0, x)
    tail = jnp.log(0.5) + jnp.log(betainc(df / 2, 0.5, df / (df + xs * xs)))   # log P(T < -|x|)
    full = jnp.where(xs < 0, tail, jnp.log1p(-jnp.exp(tail)))
    log_pdf0 = gammaln((df + 1) / 2) - gammaln(df / 2) - 0.5 * jnp.log(df * jnp.pi)
    return jnp.where(small, jnp.log(0.5) + 2 * jnp.exp(log_pdf0) * x, full)


class SkewTMixture:
    """The target above in dimension `d` at separation `s`; `alpha` is a1, `nu` the degrees of
    freedom, `w1` the z_1 scale. `comps(x)` returns the two normalized, weighted component
    log densities (log pi = logaddexp of them); `side(x)` is 1 where R's exceeds L's."""

    def __init__(self, d, s, alpha=4.0, nu=3.0, w1=2.0):
        from jax.scipy.stats import norm as jnorm, t as jt
        self.d, self.s, self.c = d, s, quartic_c(d)
        self.A1, self.NU, self.W1 = alpha, nu, w1
        q, _ = np.linalg.qr(np.random.default_rng(12345 + d).standard_normal((d, d)))
        self.R = q                                            # x = R z, z = x @ R
        Rj = jnp.asarray(q)
        a1, c = self.A1, self.c
        log_zc = np.log(quad(lambda u: np.exp(-0.5 * u * u - c * u ** 4), -np.inf, np.inf)[0])
        lst = lambda y, a: (jnp.log(2.0) + jt.logpdf(y, nu)
                            + t_logcdf(a * y * jnp.sqrt((nu + 1) / (nu + y * y)), nu + 1))   # unit ST

        def comps(x):
            z = x @ Rj
            lL = (np.log(1 - W_R) + lst((z[..., 0] + s / 2) / w1, -a1) - np.log(w1)
                  + jnp.sum(jnorm.logpdf(z[..., 1:]), -1))
            lR = (np.log(W_R) + lst((z[..., 0] - s / 2) / w1, a1) - np.log(w1)
                  + jnp.sum(-0.5 * z[..., 1:] ** 2 - c * z[..., 1:] ** 4 - log_zc, -1))
            return lL, lR

        self.comps = jax.jit(comps)
        self.logpi_single = lambda x: jnp.logaddexp(*comps(x))
        self.logpi = jax.jit(lambda x: jnp.logaddexp(*comps(x)))

    @property
    def dim(self) -> int:
        return self.d

    def log_prob(self, x):
        return self.logpi(jnp.atleast_2d(x))

    def side(self, x):
        lL, lR = self.comps(jnp.asarray(x))
        return np.asarray(lR > lL, dtype=float)

    def modes(self):
        """Laplace approximations at the two local maxima of log pi: Newton in x from each
        component's mode."""
        m1, _ = sn_mode(self.A1)
        f = lambda x: jnp.logaddexp(*self.comps(x))
        g, H = jax.jit(jax.grad(f)), jax.jit(jax.hessian(f))
        out = []
        for zc in (np.r_[-self.s / 2 - self.W1 * m1, np.zeros(self.d - 1)],
                   np.r_[self.s / 2 + self.W1 * m1, np.zeros(self.d - 1)]):
            x = jnp.asarray(self.R @ zc)
            for _ in range(50):
                x = x - jnp.linalg.solve(H(x), g(x))
            Hx = np.asarray(H(x))
            assert np.all(np.linalg.eigvalsh(Hx) < 0), f"s={self.s}: no separate modes"
            Sigma = np.linalg.inv(-Hx)
            out.append(Mode(mu=x, Sigma=jnp.asarray(0.5 * (Sigma + Sigma.T)),
                            w=float(np.exp(f(x)) * np.sqrt(np.linalg.det(Sigma)))))
        return out

    def component_draws(self, j, n, seed=0):
        """n exact draws from component j (0 = L, 1 = R): a skew-normal z_1 offset divided by
        sqrt(chi^2_nu / nu); transverse factors by inverse CDF (R) or standard normal (L)."""
        rng = np.random.default_rng(seed)
        dl = self.A1 / np.sqrt(1 + self.A1 ** 2)
        u1 = self.W1 * (dl * np.abs(rng.standard_normal(n)) + np.sqrt(1 - dl * dl) * rng.standard_normal(n))
        z = np.empty((n, self.d))
        z[:, 0] = -self.s / 2 - u1 if j == 0 else self.s / 2 + u1
        if j == 0:
            z[:, 1:] = rng.standard_normal((n, self.d - 1))
        else:
            u = np.linspace(-8, 8, 200001)
            p = np.exp(-0.5 * u ** 2 - self.c * u ** 4); cdf = np.cumsum(p); cdf /= cdf[-1]
            z[:, 1:] = np.interp(rng.random((n, self.d - 1)), cdf, u)
        z = (z @ self.R.T) @ self.R
        xi = -self.s / 2 if j == 0 else self.s / 2
        v = np.random.default_rng(seed + 1000).chisquare(self.NU, n) / self.NU
        z[:, 0] = xi + (z[:, 0] - xi) / np.sqrt(v)
        return z @ self.R.T

    def log_component(self, j, x):
        """Normalized log density of component j."""
        return np.asarray(self.comps(jnp.asarray(x))[j]) - np.log(W_R if j else 1 - W_R)

    def reference(self, n, seed=0):
        """n exact iid draws of pi."""
        nR = int(np.random.default_rng(seed).binomial(n, W_R))
        x = np.concatenate([self.component_draws(0, n - nR, seed + 1), self.component_draws(1, nR, seed + 2)])
        return x[np.random.default_rng(seed + 3).permutation(n)]

    def z1_density(self, u):
        """Exact z_1 marginal of pi on the grid u."""
        from scipy.stats import t as T
        nu, a, w = self.NU, self.A1, self.W1
        st = lambda y, a: 2 / w * T.pdf(y, nu) * T.cdf(a * y * np.sqrt((nu + 1) / (nu + y * y)), nu + 1)
        return (1 - W_R) * st((u + self.s / 2) / w, -a) + W_R * st((u - self.s / 2) / w, a)
