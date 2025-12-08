# plotting.py

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa


def plot_time_series(ts, xs, labels=None, title=None):
    """
    Plot time evolution of all state variables.

    Parameters
    ----------
    ts : array, shape (N,)
        Time vector
    xs : array, shape (N, d)
        State trajectory
    labels : list of str (optional)
    title : str (optional)
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    d = xs.shape[1]

    for i in range(d):
        label = labels[i] if labels else f"x{i}"
        ax.plot(ts, xs[:, i], label=label)

    ax.set_xlabel("t")
    ax.set_ylabel("state")
    ax.set_title(title or "Time series")
    ax.legend()
    plt.tight_layout()
    return fig, ax


def plot_phase(xs, i=0, j=1, title=None):
    """
    2D phase portrait x_i vs x_j.
    """
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(xs[:, i], xs[:, j], linewidth=1.0)

    ax.set_xlabel(f"x{i}")
    ax.set_ylabel(f"x{j}")
    ax.set_title(title or f"Phase plot: x{i} vs x{j}")
    plt.tight_layout()
    return fig, ax


def plot_3d(xs, indices=(0, 1, 2), title=None):
    """
    3D trajectory plot for three selected coordinates.
    """
    i, j, k = indices
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(xs[:, i], xs[:, j], xs[:, k], linewidth=0.8)

    ax.set_xlabel(f"x{i}")
    ax.set_ylabel(f"x{j}")
    ax.set_zlabel(f"x{k}")
    ax.set_title(title or "3D trajectory")
    plt.tight_layout()
    return fig, ax


def plot_time_series_comparison(ts, xs_true, xs_pred, labels=None, title=None):
    """
    Overlay true vs predicted time series for each state.

    Parameters
    ----------
    ts : array, shape (N,)
        Time vector.
    xs_true : array, shape (N, d)
        True trajectory.
    xs_pred : array, shape (N, d)
        Predicted trajectory (e.g., from SINDy).
    labels : list of str (optional)
        Labels for each state; default x0, x1, ...
    title : str (optional)
    """
    ts = np.asarray(ts)
    xs_true = np.asarray(xs_true)
    xs_pred = np.asarray(xs_pred)

    N, d = xs_true.shape
    if labels is None:
        labels = [f"x{i}" for i in range(d)]

    fig, axes = plt.subplots(d, 1, figsize=(8, 2.5 * d), sharex=True)

    if d == 1:
        axes = [axes]

    for i in range(d):
        axes[i].plot(ts, xs_true[:, i], label=f"true {labels[i]}")
        axes[i].plot(ts, xs_pred[:, i], "--", label=f"SINDy {labels[i]}")
        axes[i].set_ylabel(labels[i])
        axes[i].legend(loc="best")

    axes[-1].set_xlabel("t")
    fig.suptitle(title or "True vs Predicted Time Series")
    fig.tight_layout()
    return fig, axes


def plot_3d_comparison(
    xs_true, xs_pred, indices=(0, 1, 2), title_true=None, title_pred=None
):
    """
    Side-by-side 3D trajectories: true vs predicted.

    Parameters
    ----------
    xs_true : array, shape (N, d)
    xs_pred : array, shape (N, d)
    indices : tuple of int
        Which coordinates to use (default (0,1,2)).
    """
    xs_true = np.asarray(xs_true)
    xs_pred = np.asarray(xs_pred)
    i, j, k = indices

    fig = plt.figure(figsize=(10, 4))

    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1.plot(xs_true[:, i], xs_true[:, j], xs_true[:, k], linewidth=0.8)
    ax1.set_xlabel(f"x{i}")
    ax1.set_ylabel(f"x{j}")
    ax1.set_zlabel(f"x{k}")
    ax1.set_title(title_true or "True 3D trajectory")

    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    ax2.plot(xs_pred[:, i], xs_pred[:, j], xs_pred[:, k], linewidth=0.8)
    ax2.set_xlabel(f"x{i}")
    ax2.set_ylabel(f"x{j}")
    ax2.set_zlabel(f"x{k}")
    ax2.set_title(title_pred or "Predicted 3D trajectory")

    fig.tight_layout()
    return fig, (ax1, ax2)


def plot_error_time_series(ts, error, title=None):
    """
    Plot a scalar error measure vs time, e.g. ||x_true - x_pred||.

    Parameters
    ----------
    ts : array, shape (N,)
    error : array, shape (N,)
    """
    ts = np.asarray(ts)
    error = np.asarray(error)

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(ts, error)
    ax.set_xlabel("t")
    ax.set_ylabel("error")
    ax.set_title(title or "Trajectory error")
    plt.tight_layout()
    return fig, ax
