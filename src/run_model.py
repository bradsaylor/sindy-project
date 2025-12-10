# run_model.py

import os
import numpy as np

from derivatives import (
    finite_difference_derivative,
    sgolay_derivative,
    tv_derivative,
)
from sindy_core import SINDyConfig, SINDyModel

# Map keys to derivative functions
DERIV_FUNS = {
    "fd": finite_difference_derivative,
    "sg": sgolay_derivative,
    "tv": tv_derivative,
}


def sindy_equations_to_string(
    feature_names, Xi, state_names=None, coeff_tol=1e-6
):
    """
    Build a pretty-printed SINDy model string like:
        dx/dt = ...
        dv/dt = ...

    Parameters
    ----------
    feature_names : list[str]
        Names of library columns.
    Xi : array, shape (n_features, d)
        Coefficient matrix.
    state_names : list[str] or None
        Names of state variables; if None, use x1, x2, ...
    coeff_tol : float
        Coefficients with |c| < coeff_tol are treated as zero.

    Returns
    -------
    s : str
        Multi-line string with one equation per line.
    """
    Xi = np.asarray(Xi)
    p, d = Xi.shape

    if state_names is None:
        state_names = [f"x{i+1}" for i in range(d)]

    lines = []

    for j in range(d):
        terms = []
        for i in range(p):
            c = Xi[i, j]
            if abs(c) < coeff_tol:
                continue

            name = feature_names[i]

            if not terms:
                # first term: sign included in coefficient
                if name == "1":
                    term_str = f"{c:.6g}"
                else:
                    term_str = f"{c:.6g}*{name}"
            else:
                sign = " + " if c >= 0 else " - "
                c_abs = abs(c)
                if name == "1":
                    term_str = f"{sign}{c_abs:.6g}"
                else:
                    term_str = f"{sign}{c_abs:.6g}*{name}"

            terms.append(term_str)

        rhs = "".join(terms) if terms else "0"
        lines.append(f"d{state_names[j]}/dt = {rhs}")

    return "\n".join(lines)


def true_equations_to_string(problem):
    """
    Return a string with the analytic governing equations for a given
    ProblemDefinition instance, with parameters substituted.
    """
    name = getattr(problem, "name", "")

    # All parameter values are plain floats inside problem.parameters
    p = {k: float(v) for k, v in problem.parameters.items()}

    if name == "Duffing":
        alpha = p["alpha"]
        beta = p["beta"]
        gamma = p["gamma"]
        delta = p["delta"]
        omega = p["omega"]
        lines = [
            "dx/dt = v",
            (
                "dv/dt = "
                f"{gamma:.6g}*cos({omega:.6g}*t) "
                f"- {delta:.6g}*v "
                f"- {alpha:.6g}*x "
                f"- {beta:.6g}*x^3"
            ),
        ]

    elif name == "DampedOscillator":
        delta = p["delta"]
        omega = p["omega"]
        lines = [
            "dx/dt = v",
            ("dv/dt = " f"- {delta:.6g}*v " f"- {omega**2:.6g}*x"),
        ]

    elif name == "Lorenz":
        sigma = p["sigma"]
        rho = p["rho"]
        beta = p["beta"]
        lines = [
            f"dx/dt = {sigma:.6g}*(y - x)",
            f"dy/dt = x*({rho:.6g} - z) - y",
            f"dz/dt = x*y - {beta:.6g}*z",
        ]

    elif name == "Hopf":
        mu = p["mu"]
        omega = p["omega"]
        lines = [
            "dx/dt = " f"{mu:.6g}*x - {omega:.6g}*y - (x^2 + y^2)*x",
            "dy/dt = " f"{omega:.6g}*x + {mu:.6g}*y - (x^2 + y^2)*y",
        ]

    else:
        # fallback: we don't know, but at least say so
        lines = [f"[true equations not implemented for system '{name}']"]

    return "\n".join(lines)


def print_equations_from_result(result):
    """
    Pretty-print the true and SINDy equations for a loaded experiment.
    """
    system_name = result.get("system_name", "System")
    deriv_key = result.get("deriv_key", "")
    noise_level = float(result.get("noise_level", 0.0))

    header = f"{system_name} ({deriv_key}, noise={noise_level:.3f})"
    print(header)
    print("=" * len(header))

    true_eq = result.get("true_eq_str", "[true equations not available]")
    sindy_eq = result.get("sindy_eq_str", "[SINDy equations not available]")

    print("\nTrue system:")
    print(true_eq)
    print("\nSINDy model:")
    print(sindy_eq)


def run_sindy_experiment(
    problem,
    deriv_key: str,
    noise_level: float,
    sindy_config: SINDyConfig,
    outdir: str,
    drop_transient: float = 0.0,
):
    """
    Run a single SINDy experiment and save results to .npz.

    Parameters
    ----------
    problem : ProblemDefinition
        Any of LorenzDefinition, DuffingDefinition, HopfDefinition,
        DampedOscillatorDefinition, etc. (already constructed).
    deriv_key : {"fd", "sg", "tv"}
        Which derivative method to use.
    noise_level : float
        Relative noise level (e.g. 0.0, 0.01).
    sindy_config : SINDyConfig
        Configuration for the feature library and STLSQ.
    outdir : str
        Directory in which to save results.
    drop_transient : float
        Time (in same units as problem.t0, problem.tf) to discard
        from the beginning of the trajectory before fitting.
    """
    os.makedirs(outdir, exist_ok=True)

    # 1) Simulate (with or without noise)
    if noise_level > 0.0:
        # uses ProblemDefinition.simulate_with_noise from systems.py
        ts, xs_data = problem.simulate_with_noise(noise_pct=noise_level)
    else:
        ts, xs_data = problem.simulate()

    ts = np.asarray(ts)
    xs_data = np.asarray(xs_data)

    # 2) Optional transient removal (after noise)
    if drop_transient > 0.0:
        dt = float(ts[1] - ts[0])
        N_transient = int(drop_transient / dt)
        ts_used = ts[N_transient:]
        xs_used = xs_data[N_transient:, :]
    else:
        ts_used = ts
        xs_used = xs_data

    # 3) Compute derivatives
    deriv_fun = DERIV_FUNS[deriv_key]
    xdot_used = deriv_fun(ts_used, xs_used)

    # 4) Trim endpoints to avoid FD boundary artifacts
    ts_fit = ts_used[1:-1]
    X = xs_used[1:-1, :]
    Xdot = xdot_used[1:-1, :]

    # 5) Fit SINDy
    model = SINDyModel(sindy_config)

    if sindy_config.mode in ("fourier", "polynomial_and_fourier"):
        model.fit(X, Xdot, ts=ts_fit)
    else:
        model.fit(X, Xdot)

    # 6) Simulate learned model on the same interval (use standard interface)
    x0_sindy = np.array(X[0])
    ts_sindy = np.array(ts_fit)
    xs_sindy = model.simulate(x0_sindy, ts_sindy)

    # 6b) Build equation strings (true + SINDy)
    state_names = None
    cfg = sindy_config
    if getattr(cfg, "var_names", None) is not None:
        # var_names may be a tuple; cast to list of strings
        state_names = [str(v) for v in cfg.var_names]

    sindy_eq_str = sindy_equations_to_string(
        model.feature_names,
        model.Xi,
        state_names=state_names,
    )

    true_eq_str = true_equations_to_string(problem)

    # 7) Save results
    fname = f"{problem.name}_{deriv_key}_noise{noise_level:.3f}.npz"
    path = os.path.join(outdir, fname)

    np.savez(
        path,
        ts_fit=ts_fit,
        X=X,
        Xdot=Xdot,
        xs_true_seg=X,  # true data segment used for fitting
        xs_sindy=xs_sindy,  # SINDy reconstruction on same grid
        Xi=model.Xi,
        feature_names=np.array(model.feature_names),
        deriv_key=deriv_key,
        noise_level=noise_level,
        system_name=problem.name,
        config=dict(sindy_config.__dict__),
        sindy_eq_str=sindy_eq_str,
        true_eq_str=true_eq_str,
    )

    print(f"Saved: {path}")
    return path


def run_all_for_problem(
    problem,
    sindy_config: SINDyConfig,
    out_root: str,
    drop_transient: float = 0.0,
    deriv_keys=("fd", "sg", "tv"),
    noise_levels=(0.0, 0.01),
):
    """
    Convenience wrapper: run all derivative/noise combos for a single problem.

    Parameters
    ----------
    problem : ProblemDefinition
        Instantiated system (with chosen parameters, t0, tf, dt, x0).
    sindy_config : SINDyConfig
        Configuration for that problem.
    out_root : str
        Root directory; results go into out_root / problem.name.
    drop_transient : float
        Time to discard from start before fitting.
    deriv_keys : iterable of str
        Derivative method keys, defaults to ("fd", "sg", "tv").
    noise_levels : iterable of float
        Noise levels to run, defaults to (0.0, 0.01).
    """
    outdir = os.path.join(out_root, problem.name)

    for deriv in deriv_keys:
        for noise in noise_levels:
            run_sindy_experiment(
                problem=problem,
                deriv_key=deriv,
                noise_level=noise,
                sindy_config=sindy_config,
                outdir=outdir,
                drop_transient=drop_transient,
            )


# ---------------------------------------------------------------------
# Helpers for loading results and making the "usual" plots
# ---------------------------------------------------------------------

import numpy as np
from plotting import (
    plot_time_series_comparison,
    plot_phase,
    plot_error_time_series,
)


def load_experiment(path):
    """
    Load a saved SINDy experiment .npz file and return a dict-like object.

    Parameters
    ----------
    path : str
        Path to the .npz file saved by run_sindy_experiment().

    Returns
    -------
    result : dict
        Keys (as saved):
            - "ts_fit"        : (N,) time vector
            - "X"             : (N, d) true states used for fit
            - "Xdot"          : (N, d) estimated derivatives
            - "xs_true_seg"   : (N, d) same as X (for convenience)
            - "xs_sindy"      : (N, d) SINDy-simulated trajectory
            - "Xi"            : (n_features, d) coefficient matrix
            - "feature_names" : (n_features,) array of strings
            - "deriv_key"     : scalar string ("fd", "sg", "tv")
            - "noise_level"   : scalar float
            - "system_name"   : scalar string
            - "config"        : dict of SINDyConfig attributes
    """
    data = np.load(path, allow_pickle=True)
    result = {k: data[k] for k in data.files}

    # config was saved as a dict inside a 0-d array; unwrap it:
    if "config" in result:
        result["config"] = result["config"].item()

    return result


def make_standard_plots(
    result,
    phase_i=0,
    phase_j=1,
    labels=None,
    title_prefix=None,
):
    """
    Given a loaded experiment result, make the 'usual' set of plots:
    - time-series comparison (true vs SINDy)
    - phase portrait (true)
    - phase portrait (SINDy)
    - error vs time

    Parameters
    ----------
    result : dict
        As returned by load_experiment().
    phase_i, phase_j : int
        State indices for the phase portrait (e.g. 0,1 for x-v).
    labels : list of str or None
        Optional state labels for the time-series comparison.
        If None, use ["x0", "x1", ...].
    title_prefix : str or None
        Optional text to prepend in plot titles. If None, a default
        based on system name / derivative / noise is used.

    Returns
    -------
    figs : dict
        Dictionary of Matplotlib figure handles with keys:
            "time_series", "phase_true", "phase_sindy", "error"
    """
    ts = np.asarray(result["ts_fit"])
    xs_true = np.asarray(result["xs_true_seg"])
    xs_sindy = np.asarray(result["xs_sindy"])

    d = xs_true.shape[1]

    if labels is None:
        labels = [f"x{i}" for i in range(d)]

    system_name = str(result.get("system_name", "System"))
    deriv_key = str(result.get("deriv_key", ""))
    noise_level = float(result.get("noise_level", 0.0))

    if title_prefix is None:
        title_prefix = (
            f"{system_name} ({deriv_key}, noise={noise_level:.3f})"
            if deriv_key
            else system_name
        )

    figs = {}

    # 1) Time-series comparison
    figs["time_series"], _ = plot_time_series_comparison(
        ts,
        xs_true,
        xs_sindy,
        labels=labels,
        title=f"{title_prefix} – time series (true vs SINDy)",
    )

    # 2) Phase portrait (true)
    figs["phase_true"], _ = plot_phase(
        xs_true,
        i=phase_i,
        j=phase_j,
        title=f"{title_prefix} – true phase portrait",
    )

    # 3) Phase portrait (SINDy)
    figs["phase_sindy"], _ = plot_phase(
        xs_sindy,
        i=phase_i,
        j=phase_j,
        title=f"{title_prefix} – SINDy phase portrait",
    )

    # 4) Error vs time
    error_traj = np.linalg.norm(xs_true - xs_sindy, axis=1)
    figs["error"], _ = plot_error_time_series(
        ts,
        error_traj,
        title=f"{title_prefix} – trajectory error",
    )

    return figs


import numpy as np
import matplotlib.pyplot as plt


import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d  # optional smoothing


import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d  # optional smoothing


def make_composite_figure(
    result,
    phase_i=0,
    phase_j=1,
    labels=None,
    title_prefix=None,
    smooth_error=True,
    error_sigma=5.0,
):
    """
    Make a single figure with:
      (a) Time series (true vs SINDy, all states)
      (b) True phase portrait
      (c) SINDy phase portrait
      (d) Trajectory error vs time

    Parameters
    ----------
    result : dict
        As returned by load_experiment().
    phase_i, phase_j : int
        State indices for the phase portrait (e.g. 0,1 for x–v).
    labels : list of str or None
        Optional state labels. If None, defaults to ["x0", "x1", ...].
    title_prefix : str or None
        Optional text to use in the figure suptitle. If None, a default
        based on system name / derivative / noise is constructed.
    smooth_error : bool
        If True, overlay a smoothed version of the error curve.
    error_sigma : float
        Standard deviation used in gaussian_filter1d for smoothing.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : dict
        {
          "time_series": ax_ts,
          "phase_true": ax_phase_true,
          "phase_sindy": ax_phase_sindy,
          "error": ax_err,
        }
    """
    # --- unpack result ---
    ts = np.asarray(result["ts_fit"])
    xs_true = np.asarray(result["xs_true_seg"])
    xs_sindy = np.asarray(result["xs_sindy"])

    d = xs_true.shape[1]

    if labels is None:
        labels = [f"x{i}" for i in range(d)]

    system_name = str(result.get("system_name", "System"))
    deriv_key = str(result.get("deriv_key", ""))
    noise_level = float(result.get("noise_level", 0.0))

    if title_prefix is None:
        if deriv_key:
            title_prefix = (
                f"{system_name} – {deriv_key}, noise={noise_level:.3f}"
            )
        else:
            title_prefix = system_name

    # --- compute error ---
    error_traj = np.linalg.norm(xs_true - xs_sindy, axis=1)
    if smooth_error:
        error_smooth = gaussian_filter1d(error_traj, sigma=error_sigma)
    else:
        error_smooth = None

    # --- common phase limits (for fair comparison) ---
    x_true = xs_true[:, phase_i]
    y_true = xs_true[:, phase_j]
    x_sindy = xs_sindy[:, phase_i]
    y_sindy = xs_sindy[:, phase_j]

    xmin = min(x_true.min(), x_sindy.min())
    xmax = max(x_true.max(), x_sindy.max())
    ymin = min(y_true.min(), y_sindy.min())
    ymax = max(y_true.max(), y_sindy.max())

    # add a small margin
    dx = 0.05 * (xmax - xmin + 1e-9)
    dy = 0.05 * (ymax - ymin + 1e-9)
    xmin -= dx
    xmax += dx
    ymin -= dy
    ymax += dy

    # --- figure layout: 3 rows, middle row has 2 cols ---
    fig = plt.figure(figsize=(10, 8))
    gs = fig.add_gridspec(
        nrows=3,
        ncols=2,
        height_ratios=[2.0, 2.0, 1.2],
        hspace=0.5,
        wspace=0.3,
    )

    # (a) Top row: time series (full width)
    ax_ts = fig.add_subplot(gs[0, :])
    # plot all states; use solid for true, dashed for SINDy
    last_true_line = None
    last_sindy_line = None
    for i in range(d):
        (line_true,) = ax_ts.plot(
            ts,
            xs_true[:, i],
            label=f"true {labels[i]}",
        )
        (line_sindy,) = ax_ts.plot(
            ts,
            xs_sindy[:, i],
            "--",
            label=f"SINDy {labels[i]}",
        )
        last_true_line = line_true
        last_sindy_line = line_sindy

    ax_ts.set_ylabel("state")
    ax_ts.set_xlabel("t")
    ax_ts.set_title("Time series (true vs SINDy)")

    # clean legend: just "true" vs "SINDy"
    if last_true_line is not None and last_sindy_line is not None:
        ax_ts.legend(
            [last_true_line, last_sindy_line],
            ["true", "SINDy"],
            loc="upper right",
        )

    # panel label
    ax_ts.text(
        -0.08,
        1.05,
        "(a)",
        transform=ax_ts.transAxes,
        fontsize=12,
        fontweight="bold",
    )

    # (b) Middle left: true phase portrait
    ax_phase_true = fig.add_subplot(gs[1, 0])
    ax_phase_true.plot(x_true, y_true, linewidth=1.0)
    ax_phase_true.set_xlabel(labels[phase_i])
    ax_phase_true.set_ylabel(labels[phase_j])
    ax_phase_true.set_title("True phase portrait")
    ax_phase_true.set_xlim(xmin, xmax)
    ax_phase_true.set_ylim(ymin, ymax)
    ax_phase_true.text(
        -0.18,
        1.05,
        "(b)",
        transform=ax_phase_true.transAxes,
        fontsize=12,
        fontweight="bold",
    )

    # (c) Middle right: SINDy phase portrait
    ax_phase_sindy = fig.add_subplot(gs[1, 1])
    ax_phase_sindy.plot(x_sindy, y_sindy, linewidth=1.0)
    ax_phase_sindy.set_xlabel(labels[phase_i])
    ax_phase_sindy.set_ylabel(labels[phase_j])
    ax_phase_sindy.set_title("SINDy phase portrait")
    ax_phase_sindy.set_xlim(xmin, xmax)
    ax_phase_sindy.set_ylim(ymin, ymax)
    ax_phase_sindy.text(
        -0.18,
        1.05,
        "(c)",
        transform=ax_phase_sindy.transAxes,
        fontsize=12,
        fontweight="bold",
    )

    # (d) Bottom row: error vs time (full width)
    ax_err = fig.add_subplot(gs[2, :])
    ax_err.plot(ts, error_traj, linewidth=0.8, alpha=0.7)
    if error_smooth is not None:
        ax_err.plot(ts, error_smooth, linewidth=1.4, alpha=0.9)
    ax_err.set_xlabel("t")
    ax_err.set_ylabel(r"$\|x_{\mathrm{true}} - x_{\mathrm{SINDY}}\|$")
    ax_err.set_title("Trajectory error")
    ax_err.text(
        -0.08,
        1.05,
        "(d)",
        transform=ax_err.transAxes,
        fontsize=12,
        fontweight="bold",
    )

    # Global title
    fig.suptitle(title_prefix, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    axes = {
        "time_series": ax_ts,
        "phase_true": ax_phase_true,
        "phase_sindy": ax_phase_sindy,
        "error": ax_err,
    }

    return fig, axes
