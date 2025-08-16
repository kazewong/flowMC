import equinox as eqx
from jaxtyping import PRNGKeyArray, Float, Array
import optax
from flowMC.resource.base import Resource
from typing_extensions import Self
import jax.numpy as jnp
import jax
from diffrax import diffeqsolve, ODETerm, Dopri5

class Solver:
    pass
    
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
    