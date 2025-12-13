
# **SINDy Project — Sparse Identification of Nonlinear Dynamics**

### *ME 69700 — Advanced Scientific Machine Learning (Purdue University)*

This repository contains a full research-grade implementation of the **Sparse Identification of Nonlinear Dynamics (SINDy)** framework, including:

* General ODE system definitions
* Automated dataset generation via JAX/diffrax
* Multiple derivative estimation methods (FD, Savitzky-Golay, TV regularization)
* Polynomial + Fourier feature libraries
* Two independent STLSQ solvers (JAX-compatible and Brunton-style pruned)
* Batch experiment pipelines
* Automated visualization and reporting tools

This code supports all four systems used in the ME697 final project:

* Lorenz System
* Hopf oscillator
* Duffing oscillator
* Damped harmonic oscillator (toy example)

---

## **Project Structure**

```
sindy-project/
│
├── src/
│   ├── systems.py           # Definitions of all dynamical systems
│   ├── derivatives.py       # FD, SG, TV derivative estimators
│   ├── sindy_core.py        # Feature library + STLSQ implementations
│   ├── plotting.py          # Composite figure + equation print utilities
│   └── run_model.py         # Batch runner and single-run pipeline
│
├── notebooks/
│   ├── 01_lorenz.ipynb
│   ├── 02_hopf.ipynb
│   ├── 03_duffing.ipynb
│   └── 04_damped_SHO.ipynb
│
├── outputs/                 # Generated .npz results (ignored by git)
├── figures/                 # Optional saved plots (ignored by git)
│
├── README.md
└── .gitignore
```

---

## **How to Run a Single SINDy Experiment**

Example: Duffing oscillator.

```python
from systems import DuffingDefinition
from sindy_core import SINDyConfig
from run_model import run_single

import jax.numpy as jnp

theta = jnp.array([1.0, 5.0, 0.37, 0.1, 1.0])
params = dict(alpha=1, beta=5, gamma=0.37, delta=0.1, omega=1.0)

problem = DuffingDefinition(
    parameters=params,
    x0_vector=jnp.array([1.0, 0.0]),
    t0=0.0,
    tf=50.0,
    dt=0.01,
)

cfg = SINDyConfig(
    poly_degree=3,
    include_bias=True,
    threshold=0.1,
    n_iter=10,
    mode="polynomial_and_fourier",
    var_names=("x", "v"),
    include_sin=True,
    include_cos=True,
)

results = run_single(problem, cfg, drop_transient=10.0)
```

This produces:

* true trajectory
* derivative estimates
* feature matrix
* sparse coefficients
* learned model
* reconstructed simulation
* error metrics
* report-ready figures and printed equations

---

## **Batch Experiments**

To automatically run **all derivative methods** × **all noise levels**:

```python
from run_model import run_all_for_problem

run_all_for_problem(
    problem=problem,
    sindy_config=cfg,
    out_root="../outputs",
    drop_transient=10.0,
)
```

Each run generates:

```
outputs/SystemName/SystemName_fd_noise0.000.npz
outputs/SystemName/SystemName_sg_noise0.010.npz
...
```

and corresponding figures.

---

## **STLSQ Options (Sparse Regression)**

This project includes **two** implementations:

### 1. JAX-Friendly STLSQ

* Uses masking (no dynamic shapes)
* Fully JIT-compatible
* Slower for rich libraries

### 2. Brunton-Style Pruned STLSQ

* Removes inactive columns each iteration
* Matches 2016 SINDy algorithm
* ~30% faster for large libraries
* Not JIT-safe
* Used in batch runs for efficiency

Both implementations produce **identical models** for Lorenz and Duffing when thresholding is well-chosen.

---

## **Normalization vs. PySINDy Differences**

A key insight from this project:

* Our implementation **normalizes feature columns**, thresholds in normalized space, and optionally applies final pruning (`post_tol`).
* PySINDy **does not normalize by default**, causing different sparsity behavior—especially with richer libraries (degree ≥ 3).
* When PySINDy is run with `normalize_columns=True`, it matches our implementation extremely closely.

This is discussed extensively in the project report.

---

## **Dependencies**

Recommended environment:

```
python >= 3.10
jax
diffrax
numpy
scipy
matplotlib
pysindy (optional for comparison)
```

Create a conda env:

```bash
conda create -n sindy python=3.10
pip install jax diffrax numpy scipy matplotlib pysindy
```

---

## **Course Context**

This repository contains the complete implementation for the final project in:

**ME 69700 — Advanced Scientific Machine Learning**
Purdue University, Fall 2025

It demonstrates:

* SINDy algorithm internals
* Derivative estimation effects
* Library design
* Sparsity/threshold tuning
* Model recovery quality
* Effect of normalization
* Comparison to PySINDy

---

## **Planned Features**

* Config switch: `normalize=True/False`
* λ-sweeping automation (threshold tuning)
* Library sensitivity experiments
* Additional reporting tools

