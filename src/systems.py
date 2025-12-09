# systems.py
import jax.numpy as jnp
import jax.random as jr
from diffrax import diffeqsolve, ODETerm, SaveAt, Tsit5
import equinox as eqx

# Helper functions for synthetic data + noise


def add_noise(X, sigma, rng):
    return X + sigma * jr.normal(rng, X.shape)


def add_relative_noise(X, pct, rng, eps=1e-12):
    std = jnp.maximum(jnp.std(X, axis=0, ddof=1), eps)
    sigma = pct * std
    return X + sigma * jr.normal(rng, X.shape)


# General problem class plus specific sub-class types


class ProblemDefinition(eqx.Module):
    name: str = eqx.field(static=True)
    state_dim: int = eqx.field(static=True)
    parameters: dict = eqx.field(static=True)
    x0_vector: jnp.ndarray
    t0: float
    tf: float
    dt: float

    def __init__(
        self,
        name: str,
        state_dim: int,
        parameters: dict,
        x0_vector: jnp.ndarray,
        t0: float,
        tf: float,
        dt: float,
    ):
        self.name = name
        self.state_dim = state_dim
        self.parameters = parameters
        self.x0_vector = x0_vector
        self.t0 = t0
        self.tf = tf
        self.dt = dt

    def rhs(self, t, x):
        raise NotImplementedError

    def simulate(self, ts=None, max_steps=200_000):
        if ts is None:
            ts = jnp.arange(self.t0, self.tf, self.dt)

        def rhs_wrapper(t, x, args):
            return args.rhs(t, x)

        term = ODETerm(rhs_wrapper)

        sol = diffeqsolve(
            term,
            Tsit5(),
            self.t0,
            self.tf,
            self.dt,  # dt0
            self.x0_vector,
            saveat=SaveAt(ts=ts),
            args=self,
            max_steps=max_steps,
        )

        return sol.ts, sol.ys

    import jax.random as jr

    def simulate_with_noise(
        self, noise_pct=None, noise_sigma=None, key=None, ts=None
    ):
        """
        Simulate the system and optionally add noise.

        Parameters
        ----------
        noise_pct : float or None
            Relative noise level (e.g. 0.01 = 1%).
        noise_sigma : float or None
            Absolute noise std dev.
        key : jax.random.PRNGKey or None
        ts : array or None

        Returns
        -------
        ts, ys_noisy
        """
        ts, ys = self.simulate(ts)

        if key is None:
            key = jr.PRNGKey(0)

        if noise_pct is not None:
            ys_noisy = add_relative_noise(ys, noise_pct, key)
        elif noise_sigma is not None:
            ys_noisy = add_noise(ys, noise_sigma, key)
        else:
            ys_noisy = ys

        return ts, ys_noisy


class LorenzDefinition(ProblemDefinition):
    def __init__(
        self,
        parameters: dict,
        x0_vector: jnp.ndarray,
        t0: float,
        tf: float,
        dt: float,
    ):
        super().__init__(
            name="Lorenz",
            state_dim=3,
            parameters=parameters,
            x0_vector=x0_vector,
            t0=t0,
            tf=tf,
            dt=dt,
        )

    def rhs(self, t, x):
        sigma = self.parameters["sigma"]
        rho = self.parameters["rho"]
        beta = self.parameters["beta"]

        dx = sigma * (x[1] - x[0])
        dy = x[0] * (rho - x[2]) - x[1]
        dz = x[0] * x[1] - beta * x[2]

        return jnp.array([dx, dy, dz])


class HopfDefinition(ProblemDefinition):
    """
    2D Hopf normal form in real coordinates:

        dx/dt = mu * x - omega * y - (x^2 + y^2) * x
        dy/dt = omega * x + mu * y - (x^2 + y^2) * y

    Parameters dict should contain:
        "mu" : float
        "omega" : float
    """

    def __init__(
        self,
        parameters: dict,
        x0_vector: jnp.ndarray,
        t0: float,
        tf: float,
        dt: float,
    ):
        super().__init__(
            name="Hopf",
            state_dim=2,
            parameters=parameters,
            x0_vector=x0_vector,
            t0=t0,
            tf=tf,
            dt=dt,
        )

    def rhs(self, t, x):
        mu = self.parameters["mu"]
        omega = self.parameters["omega"]

        r2 = x[0] ** 2 + x[1] ** 2

        dx = mu * x[0] - omega * x[1] - r2 * x[0]
        dy = omega * x[0] + mu * x[1] - r2 * x[1]

        return jnp.array([dx, dy])


class DuffingDefinition(ProblemDefinition):
    r"""
    Forced Duffing oscillator in first-order form:

        dx/dt = v
        dv/dt = γ cos(ω t) - δ v - α x - β x^3

    Parameters dict should contain:
        "alpha" : float   # linear stiffness
        "beta"  : float   # cubic stiffness
        "gamma" : float   # forcing amplitude
        "delta" : float   # damping
        "omega" : float   # forcing frequency
    """

    def __init__(
        self,
        parameters: dict,
        x0_vector: jnp.ndarray,
        t0: float,
        tf: float,
        dt: float,
    ):
        super().__init__(
            name="Duffing",
            state_dim=2,
            parameters=parameters,
            x0_vector=x0_vector,
            t0=t0,
            tf=tf,
            dt=dt,
        )

    def rhs(self, t, x):
        alpha = self.parameters["alpha"]
        beta = self.parameters["beta"]
        gamma = self.parameters["gamma"]
        delta = self.parameters["delta"]
        omega = self.parameters["omega"]

        x_pos = x[0]
        v = x[1]

        dx = v
        dv = (
            gamma * jnp.cos(omega * t)
            - delta * v
            - alpha * x_pos
            - beta * x_pos**3
        )

        return jnp.array([dx, dv])
