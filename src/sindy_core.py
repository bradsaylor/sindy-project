# SINDy core: JAX STLSQ + helpers + model class

from dataclasses import dataclass
from typing import Sequence, Optional

import jax
import jax.numpy as jnp
import numpy as np

# Function libraries

import itertools
import jax.numpy as jnp


def generate_powers(n_features, degree, include_bias=True):
    """
    Generate exponent vectors alpha in N^d with total degree <= degree,
    ordered in a human-friendly way, similar to sklearn's PolynomialFeatures.

    for d = 2, degree = 2, include_bias = True, this yields:
        (0,0),
        (1,0), (0,1),
        (2,0), (1,1), (0,2)
    """

    powers = []

    min_deg = 0 if include_bias else 1
    for total_deg in range(min_deg, degree + 1):
        # collect all exps for this total degree
        exps_for_deg = []
        for exps in itertools.product(range(total_deg + 1), repeat=n_features):
            if sum(exps) == total_deg:
                exps_for_deg.append(exps)
        # sort in descending lexographic order so x comes before y, etc.
        exps_for_deg.sort(reverse=True)
        powers.extend(exps_for_deg)

    return powers


def make_feature_names(powers, var_names=None):
    """
    Create human-readable feature names from exponent vectors.
    """
    if not powers:
        return []

    d = len(powers[0])
    if var_names is None:
        var_names = [f"x{j+1}" for j in range(d)]

    names = []
    for alpha in powers:
        # all-zero -> constant term
        if all(a == 0 for a in alpha):
            names.append("1")
            continue

        pieces = []
        for vname, a in zip(var_names, alpha):
            if a == 0:
                continue
            elif a == 1:
                pieces.append(vname)
            else:
                pieces.append(f"{vname}^{a}")
        names.append(" ".join(pieces))

    return names


def build_polynomial_block(X, degree, include_bias=True, var_names=None):
    """
    Build a polynomial lfeature library up to a given total degree.

    Parameters
    ----------
    X: array-like, shape (N,d)
        Input data.
    degree: int
        Maximum total degree of the polynomial terms.
    include_bias: bool
        Whether to include the constant term.
    var_names: list of str or None
        Names of the variables for feature naming. If None, uses x1, x2, ...


    Returns
    -------
    Theta_poly: jnp.ndarray, shape (N,p)
        Polynomial feature matrix.
    feature_names: list of str, length p
        Human-readable names for the columns.
    powers: np.ndarray, shape (p,d)
        Exponent vectors for each feature.
    """

    X = jnp.asarray(X)
    N, d = X.shape

    powers_list = generate_powers(d, degree, include_bias=include_bias)
    p = len(powers_list)

    # Convert to np array
    powers = np.array(powers_list, dtype=int)

    # Allocate theta
    Theta = jnp.ones((N, p))

    # Fill columns
    for j in range(p):
        alpha = powers[j]
        col = jnp.ones(N)
        for k in range(d):
            a = int(alpha[k])
            if a != 0:
                col = col * (X[:, k] ** a)
        Theta = Theta.at[:, j].set(col)

    feature_names = make_feature_names(powers_list, var_names=var_names)
    return Theta, feature_names, powers


def build_fourier_block(
    z, k_max, include_sin=True, include_cos=True, prefix="t"
):
    """
    Build a Fourier feature library sin(k z), cos(k z) for k = 1, ..., k_max

    Parameters
    -----------
    z: array-like, shape (N,)
        Input variable (e.g. time).
    k_max: int
        Maximum harmonic
    include_sin, include_cos: bool
        Whether to include sin and/or cos terms.
    prefix: str
        Name of the variable for feature naming.


    Returns
    --------
    Theta_four: jnp.ndarray, shape (N,p_four)
        Fourier feature matrix (no bias)
    feature_names: list of str
        Names of columns.
    ks: list of int
        Harmonics used (repeated if both sin and cos).
    """

    z = jnp.asarray(z).ravel()
    N = z.shape[0]

    cols = []
    names = []
    ks = []

    for k in range(1, k_max + 1):
        if include_sin:
            cols.append(jnp.sin(k * z))
            names.append(f"sin({k}{prefix})")
            ks.append(k)
        if include_cos:
            cols.append(jnp.cos(k * z))
            names.append(f"cos({k}{prefix})")
            ks.append(k)

    if not cols:
        # Edge case: nothing requested
        Theta_four = jnp.empty((N, 0))
    else:
        Theta_four = jnp.column_stack(cols)

    return Theta_four, names, ks


def combine_libraries(blocks):
    """
    blocks: list of (Theta_block, names_block)

    Returns:
        Theta_full, names_full
    """

    Thetas, names_list = zip(*blocks)
    Theta_full = jnp.column_stack(Thetas)
    names_full = [name for names in names_list for name in names]
    return Theta_full, names_full


# STLSQ (normalized and JIT-safe)


def _stlsq_core(Theta, Xdot, threshold, n_iter):
    """
    STLSQ on a fixed design matrix Theta (no normalization inside).
    Uses masking instead of dynamic boolean indexing so it's JIT-safe
    """
    Theta = jnp.asarray(Theta)
    Xdot = jnp.asarray(Xdot)

    N, p = Theta.shape
    _, d = Xdot.shape

    # Initial least squares
    Xi, _, _, _ = jnp.linalg.lstsq(Theta, Xdot, rcond=None)  # (p,d)

    for _ in range(n_iter):
        # threshold small coefficients
        small = jnp.abs(Xi) < threshold  # (p,d)
        keep = (~small).astype(Xi.dtype)  # 1 where keey, 0 where drop
        Xi = Xi * keep  # zero-out small entries

        # refit each column using only active terms (via masked Theta)
        for j in range(d):
            mask = keep[:, j]  # (p,)
            Theta_masked = Theta * mask[None, :]  # (N,p)
            y = Xdot[:, j]  # (N,)
            xi_col, _, _, _ = jnp.linalg.lstsq(Theta_masked, y, rcond=None)
            Xi = Xi.at[:, j].set(xi_col)

    return Xi


def stlsq_normalized(Theta, Xdot, threshold, n_iter=10):
    """
    Column-normalized STLSQ (what is actually used)
    """
    Theta = jnp.asarray(Theta)
    Xdot = jnp.asarray(Xdot)

    # column L2 norms
    col_norms = jnp.linalg.norm(Theta, axis=0)
    col_norms = jnp.where(col_norms == 0, 1.0, col_norms)

    Theta_scaled = Theta / col_norms

    Xi_scaled = _stlsq_core(Theta_scaled, Xdot, threshold, n_iter)
    Xi = Xi_scaled / col_norms[:, None]  # undo scaling
    return Xi


stlsq_jit = jax.jit(stlsq_normalized, static_argnames=("n_iter",))


# Postprocessing + pretty print


def postprocess_Xi(Xi, tol=1e-2):
    """Zero out tiny coefficients for disply / interpretability"""
    Xi = jnp.asarray(Xi)
    mask = (jnp.abs(Xi) >= tol).astype(Xi.dtype)
    return Xi * mask


def print_sindy_equations(feature_names, Xi, state_names=None, coeff_tol=1e-6):
    """
    Pretty-print SINDy equations:
        dx/dt = ...
        dy/dt = ...
    """
    Xi = np.array(Xi)
    p, d = Xi.shape

    if state_names is None:
        state_names = [f"x{i+1}" for i in range(d)]

    for j in range(d):
        terms = []
        for i in range(p):
            c = Xi[i, j]
            if abs(c) < coeff_tol:
                continue

            name = feature_names[i]

            if not terms:
                # first term: include sign in the coefficient itself
                if name == "1":
                    term_str = f"{c:.6g}"
                else:
                    term_str = f"{c:.6g}*{name}"
            else:
                # subsequent terms: explicit " + " or " - "
                sign = " + " if c >= 0 else " - "
                c_abs = abs(c)
                if name == "1":
                    term_str = f"{sign}{c_abs:.6g}"
                else:
                    term_str = f"{sign}{c_abs:.6g}*{name}"

            terms.append(term_str)

        rhs = "".join(terms) if terms else "0"
        print(f"d{state_names[j]}/dt = {rhs}")


# Config  + SindyModel class


@dataclass
class SINDyConfig:
    poly_degree: int = 3
    include_bias: bool = True
    threshold: float = 0.1  # STLSQ threshold (normalized space)
    n_iter: int = 10
    post_tol: float = 1e-2  # zero coefficients smaller than this for display
    var_names: Sequence[str] = ("x", "y", "z")


class SINDyModel:
    """
    Minimal SINDy model wrapping polynomial library + STLSQ.
    Assumes a 'build_polynomial_block(X, degree, include_bias, var_names)'
    function is available in the same namespace.
    """

    def __init__(self, config: Optional[SINDyConfig] = None):
        self.config = config or SINDyConfig()
        self.Xi = None
        self.feature_names = None
        self.powers = None
        self.Theta_shape = None

    def fit(self, X, Xdot):
        """
        X: (N,d) state snapshots
        Xdot (N,d) time derivatives
        """
        cfg = self.config

        Theta, names, powers = build_polynomial_block(
            X,
            degree=cfg.poly_degree,
            include_bias=cfg.include_bias,
            var_names=list(cfg.var_names),
        )

        Xi = stlsq_jit(Theta, Xdot, threshold=cfg.threshold, n_iter=cfg.n_iter)
        Xi = postprocess_Xi(Xi, tol=cfg.post_tol)

        self.Xi = Xi
        self.feature_names = names
        self.powers = powers
        self.Theta_shape = Theta.shape
        return self

    def rhs(self, X):
        """
        Evaluate learned RHS at one or more points

        X: (d,) or (N,d)
        returns: (d,) or (N,d)
        """
        if self.Xi is None:
            raise RuntimeError("Call .fit() before .rhs()")

        cfg = self.config
        X = jnp.asarray(X)
        if X.ndim == 1:
            X = X[None, :]

        Theta, _, _ = build_polynomial_block(
            X,
            degree=cfg.poly_degree,
            include_bias=cfg.include_bias,
            var_names=list(cfg.var_names),
        )
        return Theta @ self.Xi

    def simulate(self, x0, ts, method="rk4"):
        """
        Simulate the learned SINDy model forward in time.

        Parameters
        ----------
        x0 : array_like, shape (d,)
            Initial condition.
        ts : array_like, shape (N,)
            Time grid (not required to be exactly uniform).
        method : {"euler", "rk4"}
            Time-stepping scheme.

        Returns
        -------
        xs : np.ndarray, shape (N, d)
            Simulated trajectory from the SINDy model.
        """
        if self.Xi is None:
            raise RuntimeError("Call .fit() before .simulate().")

        t = np.asarray(ts, dtype=float)
        if t.ndim != 1 or t.size < 2:
            raise ValueError("ts must be a 1D array with at least two points.")

        # Allow slightly non-uniform spacing: use per-step dt
        dts = np.diff(t)
        if np.any(dts <= 0):
            raise ValueError("ts must be strictly increasing.")

        x0 = np.asarray(x0, dtype=float)
        if x0.ndim != 1:
            raise ValueError("x0 must be a 1D state vector.")

        N = t.size
        d = x0.size
        xs = np.zeros((N, d), dtype=float)
        xs[0] = x0

        def rhs_numpy(x):
            # Reuse existing JAX-based rhs, but convert in/out via NumPy
            x_jax = jnp.asarray(x)
            xdot = self.rhs(x_jax)[0]  # rhs returns shape (1, d) for 1D input
            return np.asarray(xdot, dtype=float)

        if method == "euler":
            for k in range(N - 1):
                dt = dts[k]
                xk = xs[k]
                k1 = rhs_numpy(xk)
                xs[k + 1] = xk + dt * k1

        elif method == "rk4":
            for k in range(N - 1):
                dt = dts[k]
                xk = xs[k]
                k1 = rhs_numpy(xk)
                k2 = rhs_numpy(xk + 0.5 * dt * k1)
                k3 = rhs_numpy(xk + 0.5 * dt * k2)
                k4 = rhs_numpy(xk + dt * k3)
                xs[k + 1] = xk + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        else:
            raise ValueError("method must be 'euler' or 'rk4'.")

        return xs

    def print(self):
        if self.Xi is None:
            raise RuntimeError("Call .fit() before .print()")
        print_sindy_equations(
            self.feature_names,
            np.array(self.Xi),
            state_names=list(self.config.var_names),
        )
