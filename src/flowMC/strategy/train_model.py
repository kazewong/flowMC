from flowMC.strategy.base import Strategy
from flowMC.resource.base import Resource
from flowMC.resource.buffers import Buffer
from flowMC.resource.nf_model.base import NFModel
from flowMC.resource.optimizer import Optimizer
from jaxtyping import Array, Float, PRNGKeyArray
import jax


class TrainModel(Strategy):

    model_resource: str
    data_resource: str
    optimizer_resource: str
    n_epochs: int
    batch_size: int
    n_max_examples: int
    verbose: bool
    thinning: int

    def __str__(self):
        return "Train " + self.model_resource

    def __init__(
        self,
        model_resource: str,
        data_resource: str,
        optimizer_resource: str,
        n_epochs: int = 100,
        batch_size: int = 64,
        n_max_examples: int = 10000,
        thinning: int = 1,
        verbose: bool = False,
    ):
        self.model_resource = model_resource
        self.data_resource = data_resource
        self.optimizer_resource = optimizer_resource

        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.n_max_examples = n_max_examples
        self.verbose = verbose
        self.thinning = thinning

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
        training_data = data_resource.buffer.reshape(-1, data_resource.n_dims)[:: self.thinning]
        if training_data.shape[0] > self.n_max_examples:
            training_data = training_data[: self.n_max_examples]
        rng_key, subkey = jax.random.split(rng_key)
        (rng_key, model, optim_state, loss_values) = model.train(
            rng=subkey,
            data=training_data,
            optim=optimizer.optim,
            state=optimizer.optim_state,
            num_epochs=self.n_epochs,
            batch_size=self.batch_size,
            verbose=self.verbose,
        )

        optimizer.optim_state = optim_state
        resources[self.model_resource] = model
        resources[self.optimizer_resource] = optimizer
        return rng_key, resources, initial_position
