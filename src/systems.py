# systems.py
import jax.numpy as jnp
from diffrax import diffeqsolve, ODETerm, SaveAt, Tsit5
import equinox as eqx


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

    def simulate(self, ts=None):
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
            self.dt,
            self.x0_vector,
            saveat=SaveAt(ts=ts),
            args=self,
        )

        return sol.ts, sol.ys


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
