# src/derivatives.py

import numpy as np
import jax.numpy as jnp
import jax

# ---------- Finite-difference derivative ----------


@jax.jit
def finite_difference_derivative(ts, xs):
    """
    2nd-order finite difference derivative (JAX/JIT friendly).

    Parameters
    ----------
    ts : jnp.ndarray, shape (n,)
    xs : jnp.ndarray, shape (n, d)

    Returns
    -------
    Xdot : jnp.ndarray, shape (n, d)
    """
    t = jnp.ravel(ts)
    X = xs
    if X.ndim == 1:
        X = X[:, None]

    dt = jnp.diff(t)
    dt_mean = jnp.mean(dt)

    def body(X):
        n, d = X.shape
        Xdot = jnp.empty_like(X)

        # central difference for interior
        Xdot = Xdot.at[1:-1, :].set((X[2:, :] - X[:-2, :]) / (2.0 * dt_mean))

        # endpoints
        Xdot = Xdot.at[0, :].set((X[1, :] - X[0, :]) / dt_mean)
        Xdot = Xdot.at[-1, :].set((X[-1, :] - X[-2, :]) / dt_mean)
        return Xdot

    Xdot = body(X)
    return Xdot.squeeze()  # (n,) if 1D input, else (n,d)


# ---------- Savitzky–Golay derivative ----------

try:
    from scipy.signal import savgol_filter
except ImportError:  # pragma: no cover
    savgol_filter = None


import numpy as np
import jax.numpy as jnp
from scipy.signal import savgol_filter


import numpy as np
import jax.numpy as jnp
from scipy.signal import savgol_filter


def sgolay_derivative(ts, X, window_length=21, polyorder=3):
    """
    Savitzky–Golay derivative with relaxed dt handling.

    Assumes *approximately* uniform sampling; uses the mean dt.
    """
    ts_np = np.asarray(ts, dtype=float).ravel()
    X_np = np.asarray(X, dtype=float)

    if X_np.ndim == 1:
        X_np = X_np[:, None]

    # Use mean dt and only warn if strongly non-uniform
    dt_all = np.diff(ts_np)
    dt_mean = float(dt_all.mean())

    if np.max(np.abs(dt_all - dt_mean)) > 1e-3 * max(1.0, abs(dt_mean)):
        print(
            "Warning: sgolay_derivative detected noticeably non-uniform dt; "
            "using mean dt for delta."
        )

    if window_length % 2 == 0:
        raise ValueError("window_length must be odd.")

    Xdot = np.zeros_like(X_np)
    for j in range(X_np.shape[1]):
        Xdot[:, j] = savgol_filter(
            X_np[:, j],
            window_length=window_length,
            polyorder=polyorder,
            deriv=1,
            delta=dt_mean,
            mode="interp",
        )

    return jnp.asarray(Xdot if Xdot.shape[1] > 1 else Xdot[:, 0])


# ---------- TV-regularized derivative ----------

try:
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
except ImportError:  # pragma: no cover
    sp = None
    spla = None


def _tvregdiff_1d(
    y,
    t,
    alpha=1e-4,
    n_iters=15,
    eps=1e-6,
    diffkernel="abs",  # "abs" for TV, "sq" for Tikhonov-like
    cgtol=1e-4,
    cgmaxit=200,
):
    """
    Total-variation-regularized numerical differentiation (TVRegDiff)
    for a single 1D signal y(t).
    """
    if sp is None or spla is None:
        raise ImportError(
            "tv_derivative requires SciPy (scipy.sparse, scipy.sparse.linalg)."
        )

    y = np.asarray(y, dtype=float).ravel()
    t = np.asarray(t, dtype=float).ravel()
    n = y.size
    if t.size != n:
        raise ValueError("t and y must have the same length")

    dt = float(np.mean(np.diff(t)))
    data = y - y[0]

    # Integration operator A(u)_k = dt * sum_{j<=k} u_j
    def A(u):
        return dt * np.cumsum(u)

    # Adjoint A^T(w)_j = dt * sum_{k>=j} w_k
    def AT(w):
        w = np.asarray(w)
        prefix = np.concatenate(([0.0], np.cumsum(w[:-1])))
        return dt * (np.sum(w) - prefix)

    # First-order forward difference D
    diag_main = -np.ones(n)
    diag_upper = np.ones(n)
    D = sp.diags([diag_main, diag_upper], [0, 1], shape=(n, n)) / dt
    D = D.tolil()
    D[-1, :] = 0.0
    D = D.tocsc()
    DT = D.transpose()

    # 2nd-order finite-difference init
    u = np.empty_like(data)
    u[0] = (data[1] - data[0]) / dt
    u[1:-1] = (data[2:] - data[:-2]) / (2.0 * dt)
    u[-1] = (data[-1] - data[-2]) / dt

    ATd = AT(data)

    for _ in range(n_iters):
        if diffkernel == "abs":
            Du = D @ u
            weights = 1.0 / np.sqrt(Du**2 + eps)
            Q = sp.diags(weights, 0, shape=(n, n))
            L = DT @ Q @ D
        elif diffkernel == "sq":
            L = DT @ D
        else:
            raise ValueError("diffkernel must be 'abs' or 'sq'")

        # gradient of objective
        g = AT(A(u) - data) + alpha * (L @ u)

        # Hessian-like 
        def linop(v):
            return alpha * (L @ v) + AT(A(v))

        H = spla.LinearOperator((n, n), matvec=linop)

        s, info = spla.cg(H, -g, rtol=cgtol, maxiter=cgmaxit)
        if info < 0:
            raise RuntimeError(f"CG failed with error code {info}")
        u = u + s

    return u  # already du/dt


def tvreg_diff(
    t,
    X,
    alpha=1e-4,
    n_iters=15,
    order=1,
    eps=1e-6,
    diffkernel="abs",
    cgtol=1e-4,
    cgmaxit=200,
):
    """
    TV-regularized differentiation for 1D or multi-dimensional data (NumPy).
    """
    if order != 1:
        raise NotImplementedError("Only first-order TVRegDiff is implemented.")

    X = np.asarray(X, dtype=float)
    t = np.asarray(t, dtype=float).ravel()

    if X.ndim == 1:
        return _tvregdiff_1d(
            X,
            t,
            alpha=alpha,
            n_iters=n_iters,
            eps=eps,
            diffkernel=diffkernel,
            cgtol=cgtol,
            cgmaxit=cgmaxit,
        )

    if X.ndim == 2:
        n, d = X.shape
        if t.size != n:
            raise ValueError("Length of t must match number of rows in X")
        Xdot = np.empty_like(X)
        for j in range(d):
            Xdot[:, j] = _tvregdiff_1d(
                X[:, j],
                t,
                alpha=alpha,
                n_iters=n_iters,
                eps=eps,
                diffkernel=diffkernel,
                cgtol=cgtol,
                cgmaxit=cgmaxit,
            )
        return Xdot

    raise ValueError("X must be 1D or 2D (n,) or (n, d)")


def tv_derivative(
    ts,
    xs,
    alpha=1e-4,
    n_iters=15,
    **kwargs,
):
    """
    JAX-friendly wrapper around tvreg_diff.
    """
    t_np = np.asarray(ts, dtype=float)
    X_np = np.asarray(xs, dtype=float)

    Xdot_np = tvreg_diff(
        t_np,
        X_np,
        alpha=alpha,
        n_iters=n_iters,
        **kwargs,
    )

    return jnp.asarray(Xdot_np)
