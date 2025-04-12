from flowMC.strategy.base import Strategy
from flowMC.resource.base import Resource
from flowMC.resource.buffers import Buffer
from flowMC.resource.nf_model.base import NFModel
from flowMC.resource.optimizer import Optimizer
from jaxtyping import Array, Float, PRNGKeyArray
import jax
import jax.numpy as jnp


class TrainModel(Strategy):
    model_resource: str
    data_resource: str
    optimizer_resource: str
    n_epochs: int
    batch_size: int
    n_max_examples: int
    verbose: bool
    thinning: int

    def __repr__(self):
        return "Train " + self.model_resource

    def __init__(
        self,
        model_resource: str,
        data_resource: str,
        optimizer_resource: str,
        loss_buffer_name: str = "",
        n_epochs: int = 100,
        batch_size: int = 64,
        n_max_examples: int = 10000,
        history_window: int = 100,
        verbose: bool = False,
    ):
        self.model_resource = model_resource
        self.data_resource = data_resource
        self.optimizer_resource = optimizer_resource
        self.loss_buffer_name = loss_buffer_name

        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.n_max_examples = n_max_examples
        self.verbose = verbose
        self.history_window = history_window

    def __call__(
        self,
        rng_key: PRNGKeyArray,
        resources: dict[str, Resource],
        initial_position: Float[Array, "n_chains n_dim"],
        data: dict,
    ) -> tuple[
        PRNGKeyArray,
        dict[str, Resource],
        Float[Array, "n_chains n_dim"],
    ]:
        model = resources[self.model_resource]
        assert isinstance(model, NFModel), "Target resource must be a NFModel"
        data_resource = resources[self.data_resource]
        assert isinstance(data_resource, Buffer), "Data resource must be a buffer"
        optimizer = resources[self.optimizer_resource]
        assert isinstance(
            optimizer, Optimizer
        ), "Optimizer resource must be an optimizer"
        n_chains = data_resource.data.shape[0]
        n_dims = data_resource.data.shape[-1]
        training_data = data_resource.data[
            jnp.isfinite(data_resource.data).all(axis=-1)
        ].reshape(n_chains, -1, n_dims)
        training_data = training_data[:, -self.history_window :].reshape(-1, n_dims)
        subkey, rng_key = jax.random.split(rng_key)
        training_data = training_data[
            jax.random.choice(
                subkey,
                jnp.arange(training_data.shape[0]),
                shape=(self.n_max_examples,),
                replace=True,
            )
        ]
        rng_key, subkey = jax.random.split(rng_key)

        if self.verbose:
            print("Training model")
            print(f"Training data shape: {training_data.shape}")
            print(f"n_epochs: {self.n_epochs}")
            print(f"batch_size: {self.batch_size}")

        (rng_key, model, optim_state, loss_values) = model.train(
            rng=subkey,
            data=training_data,
            optim=optimizer.optim,
            state=optimizer.optim_state,
            num_epochs=self.n_epochs,
            batch_size=self.batch_size,
            verbose=self.verbose,
        )

        if self.loss_buffer_name != "":
            loss_buffer = resources[self.loss_buffer_name]
            assert isinstance(
                loss_buffer, Buffer
            ), "Loss buffer resource must be a buffer"
            loss_buffer.update_buffer(loss_values, start=loss_buffer.cursor)
            loss_buffer.cursor += len(loss_values)
            resources[self.loss_buffer_name] = loss_buffer

        optimizer.optim_state = optim_state
        resources[self.model_resource] = model
        resources[self.optimizer_resource] = optimizer
        # print(f"Training loss: {loss_values}")
        return rng_key, resources, initial_position
