# utils.py
from __future__ import annotations

from typing import Callable
import math

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.scipy as jsp

Array = np.ndarray
JaxArray = jnp.ndarray


# ============================================================
# Type checks
# ============================================================

def is_numpy(x) -> bool:
    return isinstance(x, np.ndarray)

def is_jax(x) -> bool:
    # Works for DeviceArray and tracers
    return isinstance(x, (jnp.ndarray, jax.Array))


# ============================================================
# Shape helpers (NumPy)  (unchanged)
# ============================================================

def as1d_np(x: Array) -> Array:
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError(f"Expected 1D array, got shape {x.shape}")
    return x

def as2d_np(X: Array) -> Array:
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {X.shape}")
    return X


# ============================================================
# Linear algebra helpers (NumPy)  (unchanged)
# ============================================================

def sym_np(A: Array) -> Array:
    A = np.asarray(A, dtype=float)
    if A.ndim != 2:
        raise ValueError(f"Expected matrix, got shape {A.shape}")
    return 0.5 * (A + A.T)

def chol_jitter_np(S: Array, jitter0: float = 1e-10, max_tries: int = 12) -> Array:
    S = sym_np(S)
    if S.ndim != 2 or S.shape[0] != S.shape[1]:
        raise ValueError(f"Expected square matrix, got shape {S.shape}")
    d = S.shape[0]
    I = np.eye(d)

    jitter = 0.0
    for _ in range(max_tries):
        try:
            return np.linalg.cholesky(S + jitter * I)
        except np.linalg.LinAlgError:
            jitter = jitter0 if jitter == 0.0 else jitter * 10.0

    raise np.linalg.LinAlgError("Cholesky failed even after adding jitter.")

def logdet_spd_np(S: Array) -> float:
    L = chol_jitter_np(S)
    return 2.0 * float(np.sum(np.log(np.diag(L))))

def quad_inv_np(S: Array, v: Array) -> float:
    v = as1d_np(v)
    S = np.asarray(S, dtype=float)
    d = v.size
    if S.shape != (d, d):
        raise ValueError(f"Shape mismatch: S is {S.shape}, v is {v.shape}")
    L = chol_jitter_np(S)
    y = np.linalg.solve(L, v)
    return float(np.dot(y, y))

def log_mvnorm_np(x: Array, mu: Array, Sigma: Array) -> float:
    x = as1d_np(x)
    mu = as1d_np(mu)
    d = x.size
    if mu.size != d:
        raise ValueError(f"mu has shape {mu.shape}, expected ({d},)")
    Sigma = np.asarray(Sigma, dtype=float)
    if Sigma.shape != (d, d):
        raise ValueError(f"Sigma has shape {Sigma.shape}, expected ({d},{d})")

    L = chol_jitter_np(Sigma)
    diff = x - mu
    y = np.linalg.solve(L, diff)
    quad = float(np.dot(y, y))
    ld = 2.0 * float(np.sum(np.log(np.diag(L))))
    return -0.5 * (d * np.log(2.0 * np.pi) + ld + quad)

def fd_hess_np(f: Callable[[Array], float], x: Array, eps: float = 1e-4) -> Array:
    x = as1d_np(x)
    d = x.size
    H = np.zeros((d, d))
    fx = float(f(x))
    for i in range(d):
        xi = x.copy()
        xi[i] += eps
        fxi = float(f(xi))
        for j in range(i, d):
            xij = xi.copy()
            xij[j] += eps
            fij = float(f(xij))
            xj = x.copy()
            xj[j] += eps
            fxj = float(f(xj))
            hij = (fij - fxi - fxj + fx) / (eps * eps)
            H[i, j] = hij
            H[j, i] = hij
    return sym_np(H)


# ============================================================
# Shape helpers (JAX)
# ============================================================

def as1d_jax(x) -> JaxArray:
    x = jnp.asarray(x)
    if x.ndim != 1:
        raise ValueError(f"Expected 1D array, got shape {tuple(x.shape)}")
    return x

def as2d_jax(X) -> JaxArray:
    X = jnp.asarray(X)
    if X.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {tuple(X.shape)}")
    return X


# ============================================================
# Linear algebra helpers (JAX)
# ============================================================

def sym_jax(A: JaxArray) -> JaxArray:
    """
    Symmetrize a matrix or a batch of matrices.

    Accepts:
      - (d, d)
      - (n, d, d)

    Returns same shape.
    """
    A = jnp.asarray(A)
    if A.ndim not in (2, 3):
        raise ValueError(f"Expected matrix or batched matrices, got shape {tuple(A.shape)}")
    return 0.5 * (A + jnp.swapaxes(A, -1, -2))




def chol_jitter_jax(S: Array, jitter0: float = 1e-10, max_tries: int = 12) -> Array:
    """
    Cholesky with adaptive diagonal jitter.

    Supports:
      - S shape (d, d)    -> returns (d, d)
      - S shape (n, d, d) -> returns (n, d, d)
    """
    S = sym_jax(S)

    if S.ndim == 2:
        d = S.shape[0]
        if S.shape[1] != d:
            raise ValueError(f"Expected square matrix, got shape {S.shape}")
        I = jnp.eye(d, dtype=S.dtype)

        jitter = 0.0
        last_err = None
        for _ in range(max_tries):
            try:
                return jnp.linalg.cholesky(S + jitter * I)
            except Exception as e:
                last_err = e
                jitter = jitter0 if jitter == 0.0 else 10.0 * jitter

        raise RuntimeError("Cholesky failed even after adding jitter.") from last_err

    elif S.ndim == 3:
        n, d, d2 = S.shape
        if d != d2:
            raise ValueError(f"Expected square batched matrices, got shape {S.shape}")
        I = jnp.eye(d, dtype=S.dtype)[None, :, :]

        jitter = 0.0
        last_err = None
        for _ in range(max_tries):
            try:
                return jnp.linalg.cholesky(S + jitter * I)
            except Exception as e:
                last_err = e
                jitter = jitter0 if jitter == 0.0 else 10.0 * jitter

        raise RuntimeError("Batched Cholesky failed even after adding jitter.") from last_err

    else:
        raise ValueError(f"Expected ndim 2 or 3, got shape {S.shape}")


def logdet_spd_jax(S: Array, jitter0: float = 1e-10, max_tries: int = 12) -> Array:
    L = chol_jitter_jax(S, jitter0=jitter0, max_tries=max_tries)
    if L.ndim == 2:
        return 2.0 * jnp.sum(jnp.log(jnp.diag(L)))
    elif L.ndim == 3:
        return 2.0 * jnp.sum(jnp.log(jnp.diagonal(L, axis1=-2, axis2=-1)), axis=-1)
    else:
        raise ValueError(f"Expected ndim 2 or 3, got shape {L.shape}")


def quad_inv_jax(S: Array, v: Array, jitter0: float = 1e-10, max_tries: int = 12) -> Array:
    v = as1d_jax(v)
    S = jnp.asarray(S)
    d = v.shape[0]
    if S.shape != (d, d):
        raise ValueError(f"Shape mismatch: S is {S.shape}, v is {v.shape}")
    L = chol_jitter_jax(S, jitter0=jitter0, max_tries=max_tries)
    y = jax.scipy.linalg.solve_triangular(L, v[:, None], lower=True).squeeze(-1)
    return jnp.dot(y, y)


def log_mvnorm_jax(x: JaxArray, mu: JaxArray, Sigma: JaxArray,
                   jitter0: float = 1e-10, max_tries: int = 12) -> JaxArray:
    """
    log N(x | mu, Sigma) for single x,mu,Sigma.
    x, mu: (d,)
    Sigma: (d,d)
    """
    x = as1d_jax(x)
    mu = as1d_jax(mu)
    d = x.shape[0]
    if mu.shape[0] != d:
        raise ValueError("mu dimension mismatch")
    if Sigma.shape != (d, d):
        raise ValueError("Sigma shape mismatch")

    L = chol_jitter_jax(Sigma, jitter0=jitter0, max_tries=max_tries)
    diff = x - mu
    y = jsp.linalg.solve_triangular(L, diff, lower=True)
    quad = jnp.dot(y, y)
    ld = 2.0 * jnp.sum(jnp.log(jnp.diag(L)))
    return -0.5 * (d * math.log(2.0 * math.pi) + ld + quad)


def fd_hess_jax(f: Callable[[JaxArray], JaxArray], x: JaxArray, eps: float = 1e-4) -> JaxArray:
    """
    Finite-difference Hessian (JAX, NOT jitted by default).
    Mirrors your NumPy fd_hess_np structure.
    """
    x = as1d_jax(x)
    d = x.shape[0]
    H = jnp.zeros((d, d), dtype=x.dtype)
    fx = f(x)

    # Python loops are fine for small d; if you need big d, we can vectorize.
    for i in range(d):
        xi = x.at[i].add(eps)
        fxi = f(xi)
        for j in range(i, d):
            xij = xi.at[j].add(eps)
            fij = f(xij)
            xj = x.at[j].add(eps)
            fxj = f(xj)
            hij = (fij - fxi - fxj + fx) / (eps * eps)
            H = H.at[i, j].set(hij)
            H = H.at[j, i].set(hij)

    return sym_jax(H)