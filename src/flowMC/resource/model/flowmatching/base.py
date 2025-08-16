import equinox as eqx
from jaxtyping import PRNGKeyArray, Float, Array, PyTree
from numpy import diff
import optax
from flowMC.resource.base import Resource
from flowMC.resource.model.common import MLP
from typing_extensions import Self
import jax.numpy as jnp
import jax
from diffrax import diffeqsolve, ODETerm, Dopri5, AbstractSolver

class Solver:

    model: MLP # Shape should be [input_dim + t_dim, hiddens, output_dim]
    method: AbstractSolver

    def __init__(self, model: MLP, method: str = "dopri5"):
        self.model = model
        self.method = Dopri5()
        

    def sample(self, rng_key: PRNGKeyArray, n_samples: int, dt: Float = 1e-1) -> Float[Array, "n_samples n_dim"]:
        """Sample points from the solver."""
        term = ODETerm(self.model_wrapper)
        x0 = jax.random.normal(rng_key, (n_samples, self.model.n_input-1)) 
        sols = eqx.filter_vmap(self.solve_ode, in_axes=(0, None))(x0, dt)
        return sols
        

    def log_prob():
        pass
    
    def solve_ode(self, y0: Float[Array, " n_dims"], dt: Float = 1e-1) -> Float[Array, " n_dims"]:
        """Solve the ODE with initial condition y0."""
        term = ODETerm(self.model_wrapper)
        sol = diffeqsolve(
            term,
            self.method,
            t0=0.0,
            t1=1.0,
            dt0=dt,
            y0=y0,
        )
        return sol.ys[-1] # type: ignore
    
    def model_wrapper(self, t: Float, x: Float[Array, "n_dims"], args: PyTree) -> Float[Array, "n_dim"]:
        """Wrapper for the model to be used in the ODE solver."""
        t = jnp.expand_dims(t, axis=-1)
        x = jnp.concatenate([x, t], axis=-1)
        return self.model(x)


class Scheduler:

    def __call__(self, t: Float) -> tuple[Float, Float, Float, Float]:
        """Return the parameters of the scheduler at time t."""
        raise NotImplementedError

class CondOTScheduler(Scheduler):
    """Conditional Optimal Transport Scheduler."""

    def __call__(self, t: Float) -> tuple[Float, Float, Float, Float]:
        """Return the parameters of the scheduler at time t."""
        # Implement the logic to compute alpha_t, d_alpha_t, sigma_t, d_sigma_t
        return t, 1., 1. - t, -1.

class Path:

    scheduler: Scheduler

    def sample(self, x0: Float, x1: Float, t:Float) -> Float:
        """Sample a point along the path between x0 and x1 at time t."""
        alpha_t, d_alpha_t, sigma_t, d_sigma_t = self.scheduler(t)
        x_t = sigma_t * x0 + alpha_t * x1
        dx_t = d_sigma_t * x0 + d_alpha_t * x1
        return x_t, dx_t

class FlowMatchingModel(eqx.Module, Resource):

    solver: Solver
    path: Path

    def __init__(self, solver: Solver, path: Path):
        self.solver = solver
        self.path = path

    def sample(self, rng_key: PRNGKeyArray, num_samples: int) -> Float[Array, " n_dim"]:
        raise NotImplementedError

    def log_prob(self, x: Float[Array, " n_dim"]) -> Float:
        raise NotImplementedError

    def save_model(self, path: str):
        eqx.tree_serialise_leaves(path + ".eqx", self)

    def load_model(self, path: str) -> Self:
        return eqx.tree_deserialise_leaves(path + ".eqx", self)

    @eqx.filter_value_and_grad
    def loss_fn(self, x: Float[Array, "n_batch n_dim"]) -> Float:
        return -jnp.mean(jax.vmap(self.log_prob)(x))

    @eqx.filter_jit
    def train_step(
        model: Self,
        x: Float[Array, "n_batch n_dim"],
        optim: optax.GradientTransformation,
        state: optax.OptState,
    ) -> tuple[Float[Array, " 1"], Self, optax.OptState]:
        raise NotImplementedError

    def train(
        self: Self,
        rng: PRNGKeyArray,
        data: Array,
        optim: optax.GradientTransformation,
        state: optax.OptState,
        num_epochs: int,
        batch_size: int,
        verbose: bool = True,
    ) -> tuple[PRNGKeyArray, Self, optax.OptState, Array]:
        raise NotImplementedError

    save_resource = save_model
    load_resource = load_model
