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
from tqdm import trange, tqdm

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
    
    def __init__(self, scheduler: Scheduler):
        self.scheduler = scheduler

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
        rng_key, subkey = jax.random.split(rng_key)
        samples = self.solver.sample(subkey, num_samples)
        return samples

    def log_prob(self, x: Float[Array, " n_dim"]) -> Float:
        raise NotImplementedError

    def save_model(self, path: str):
        eqx.tree_serialise_leaves(path + ".eqx", self)

    def load_model(self, path: str) -> Self:
        return eqx.tree_deserialise_leaves(path + ".eqx", self)

    @eqx.filter_value_and_grad
    def loss_fn(self, x: Float[Array, "n_batch n_dim"], t:Float[Array, "n_batch 1"], dx_t: Float[Array, "n_batch n_dim"],) -> Float:
        x = jnp.concatenate([x, t], axis=-1)
        return jnp.mean((eqx.filter_vmap(self.solver.model, in_axes=(0))(x) - dx_t) ** 2)

    @eqx.filter_jit
    def train_step(
        model: Self,
        x_t: Float[Array, "n_batch n_dim"],
        t: Float[Array, "n_batch 1"],
        dx_t: Float[Array, "n_batch n_dim"],
        optim: optax.GradientTransformation,
        state: optax.OptState,
    ) -> tuple[Float[Array, " 1"], Self, optax.OptState]:
        print("Compiling training step")
        loss, grads = model.loss_fn(x_t, t, dx_t)
        updates, state = optim.update(grads, state, model)  # type: ignore
        model = eqx.apply_updates(model, updates)
        return loss, model, state

    def train_epoch(
        self: Self,
        rng: PRNGKeyArray,
        optim: optax.GradientTransformation,
        state: optax.OptState,
        data: tuple[Float[Array, "n_example n_dim"], Float[Array, "n_example n_dim"], Float[Array, "n_example 1"]],
        batch_size: Float,
    ) -> tuple[Float, Self, optax.OptState]:
        """Train for a single epoch."""
        value = 1e9
        model = self
        train_ds_size = len(data)
        steps_per_epoch = train_ds_size // batch_size
        if steps_per_epoch > 0:
            perms = jax.random.permutation(rng, train_ds_size)

            perms = perms[: steps_per_epoch * batch_size]  # skip incomplete batch
            perms = perms.reshape((steps_per_epoch, batch_size))
            for perm in perms:
                batch_x0, batch_x1, batch_t = data[0][perm, ...], data[1][perm, ...], data[2][perm, ...]
                batch_x_t, batch_dx_t = self.path.sample(batch_x0, batch_x1, batch_t)
                value, model, state = model.train_step(
                    batch_x_t, batch_t, batch_dx_t, optim, state
                )
        else:
            x_t, dx_t = self.path.sample(data[0], data[1], data[2])
            value, model, state = model.train_step(
                x_t, data[2], dx_t, optim, state
            )
        return value, model, state

    def train(
        self: Self,
        rng: PRNGKeyArray,
        data: tuple[Float[Array, "n_example n_dim"], Float[Array, "n_example n_dim"], Float[Array, "n_example 1"]],
        optim: optax.GradientTransformation,
        state: optax.OptState,
        num_epochs: int,
        batch_size: int,
        verbose: bool = True,
    ) -> tuple[PRNGKeyArray, Self, optax.OptState, Array]:
        """Train a normalizing flow model.

        Args:
            rng (PRNGKeyArray): JAX PRNGKey.
            model (eqx.Module): NF model to train.
            data (Array): Training data.
            num_epochs (int): Number of epochs to train for.
            batch_size (int): Batch size.
            verbose (bool): Whether to print progress.

        Returns:
            rng (PRNGKeyArray): Updated JAX PRNGKey.
            model (eqx.Model): Updated NF model.
            loss_values (Array): Loss values.
        """
        loss_values = jnp.zeros(num_epochs)
        if verbose:
            pbar = trange(num_epochs, desc="Training NF", miniters=int(num_epochs / 10))
        else:
            pbar = range(num_epochs)

        best_model = model = self
        best_state = state
        best_loss = 1e9
        # model = eqx.tree_at(lambda m: m._data_mean, model, jnp.mean(data, axis=0))
        # model = eqx.tree_at(lambda m: m._data_cov, model, jnp.cov(data.T))
        for epoch in pbar:
            # Use a separate PRNG key to permute image data during shuffling
            rng, input_rng = jax.random.split(rng)
            # Run an optimization step over a training batch
            value, model, state = model.train_epoch(
                input_rng, optim, state, data, batch_size
            )
            loss_values = loss_values.at[epoch].set(value)
            if loss_values[epoch] < best_loss:
                best_model = model
                best_state = state
                best_loss = loss_values[epoch]
            if verbose:
                assert isinstance(pbar, tqdm)
                if num_epochs > 10:
                    if epoch % int(num_epochs / 10) == 0:
                        pbar.set_description(f"Training NF, current loss: {value:.3f}")
                else:
                    if epoch == num_epochs:
                        pbar.set_description(f"Training NF, current loss: {value:.3f}")

        return rng, best_model, best_state, loss_values
        
    save_resource = save_model
    load_resource = load_model

    def print_parameters(self):
        raise NotImplementedError("print_parameters is not implemented for FlowMatchingModel")
        
    