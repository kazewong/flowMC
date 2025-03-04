import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from flowMC.strategy.base import Strategy
from flowMC.resource.base import Resource
from flowMC.resource_strategy_bundles import ResourceStrategyBundle


class Sampler:
    """Top level API that the users primarily interact with.

    Args:
        n_dim (int): Dimension of the parameter space.
        n_chains (int): Number of chains to sample.
        rng_key (PRNGKeyArray): Jax PRNGKey.
        logpdf (Callable[[Float[Array, "n_dim"], dict], Float): Log probability function.
        resources (dict[str, Resource]): Resources to be used by the sampler.
        strategies (list[Strategy]): List of strategies to be used by the sampler.
        verbose (bool): Whether to print out progress. Defaults to False.
        logging (bool): Whether to log the progress. Defaults to True.
        outdir (str): Directory to save the logs. Defaults to "./outdir/".
    """

    # Essential parameters
    n_dim: int
    n_chains: int
    rng_key: PRNGKeyArray
    resources: dict[str, Resource]
    strategies: list[Strategy]

    # Logging hyperparameters
    verbose: bool = False
    logging: bool = True
    outdir: str = "./outdir/"

    def __init__(
        self,
        n_dim: int,
        n_chains: int,
        rng_key: PRNGKeyArray,
        resources: None | dict[str, Resource] = None,
        strategies: None | list[Strategy] = None,
        resource_strategy_bundles: None | ResourceStrategyBundle = None,
        **kwargs,
    ):
        # Copying input into the model

        self.n_dim = n_dim
        self.n_chains = n_chains
        self.rng_key = rng_key

        if resources is not None and strategies is not None:
            print(
                "Resources and strategies provided. Ignoring resource strategy bundles."
            )
            self.resources = resources
            self.strategies = strategies
        else:
            print(
                "Resources or strategies not provided. Using resource strategy bundles."
            )
            assert (
                resource_strategy_bundles is not None
            ), "Resource strategy bundles must be provided if resources and strategies are not."
            self.resources = resource_strategy_bundles.resources
            self.strategies = resource_strategy_bundles.strategies

        # Set and override any given hyperparameters
        class_keys = list(self.__class__.__dict__.keys())
        for key, value in kwargs.items():
            if key in class_keys:
                if not key.startswith("__"):
                    setattr(self, key, value)

    def sample(self, initial_position: Float[Array, "n_chains n_dim"], data: dict):
        """Sample from the posterior using the local sampler.

        Args:
            initial_position (Device Array): Initial position.
            data (dict): Data to be used by the likelihood functions
        """

        initial_position = jnp.atleast_2d(initial_position)  # type: ignore
        rng_key = self.rng_key
        last_step = initial_position
        for strategy in self.strategies:
            (
                rng_key,
                self.resources,
                last_step,
            ) = strategy(rng_key, self.resources, last_step, data)

    # TODO: Implement quick access and summary functions that operates on buffer

    # def get_sampler_state(self, training: bool = False) -> dict:
    #     """
    #     Get the sampler state. There are two sets of sampler outputs one can get,
    #     the training set and the production set.
    #     The training set is produced during the global tuning step, and the production set
    #     is produced during the production run.
    #     Only the training set contains information about the loss function.
    #     Only the production set should be used to represent the final set of samples.

    #     Args:
    #         training (bool): Whether to get the training set sampler state. Defaults to False.

    #     """
    #     if training is True:
    #         return self.summary["GlobalTuning"]
    #     else:
    #         return self.summary["GlobalSampling"]

    # def get_global_acceptance_distribution(
    #     self, n_bins: int = 10, training: bool = False
    # ) -> tuple[Int[Array, "n_bin n_loop"], Float[Array, "n_bin n_loop"]]:
    #     """
    #     Get the global acceptance distribution as a histogram per epoch.

    #     Returns:
    #         axis (Device Array): Axis of the histogram.
    #         hist (Device Array): Histogram of the global acceptance distribution.
    #     """
    #     if training is True:
    #         n_loop = self.n_loop_training
    #         global_accs = self.summary["training"]["global_accs"]
    #     else:
    #         n_loop = self.n_loop_production
    #         global_accs = self.summary["production"]["global_accs"]

    #     hist = [
    #         jnp.histogram(
    #             global_accs[
    #                 :,
    #                 i
    #                 * (self.n_global_steps // self.output_thinning - 1) : (i + 1)
    #                 * (self.n_global_steps // self.output_thinning - 1),
    #             ].mean(axis=1),
    #             bins=n_bins,
    #         )
    #         for i in range(n_loop)
    #     ]
    #     axis = jnp.array([hist[i][1][:-1] for i in range(n_loop)]).T
    #     hist = jnp.array([hist[i][0] for i in range(n_loop)]).T
    #     return axis, hist

    # def get_local_acceptance_distribution(
    #     self, n_bins: int = 10, training: bool = False
    # ) -> tuple[Int[Array, "n_bin n_loop"], Float[Array, "n_bin n_loop"]]:
    #     """
    #     Get the local acceptance distribution as a histogram per epoch.

    #     Returns:
    #         axis (Device Array): Axis of the histogram.
    #         hist (Device Array): Histogram of the local acceptance distribution.
    #     """
    #     if training is True:
    #         n_loop = self.n_loop_training
    #         local_accs = self.summary["training"]["local_accs"]
    #     else:
    #         n_loop = self.n_loop_production
    #         local_accs = self.summary["production"]["local_accs"]

    #     hist = [
    #         jnp.histogram(
    #             local_accs[
    #                 :,
    #                 i
    #                 * (self.n_local_steps // self.output_thinning - 1) : (i + 1)
    #                 * (self.n_local_steps // self.output_thinning - 1),
    #             ].mean(axis=1),
    #             bins=n_bins,
    #         )
    #         for i in range(n_loop)
    #     ]
    #     axis = jnp.array([hist[i][1][:-1] for i in range(n_loop)]).T
    #     hist = jnp.array([hist[i][0] for i in range(n_loop)]).T
    #     return axis, hist

    # def get_log_prob_distribution(
    #     self, n_bins: int = 10, training: bool = False
    # ) -> tuple[Int[Array, "n_bin n_loop"], Float[Array, "n_bin n_loop"]]:
    #     """
    #     Get the log probability distribution as a histogram per epoch.

    #     Returns:
    #         axis (Device Array): Axis of the histogram.
    #         hist (Device Array): Histogram of the log probability distribution.
    #     """
    #     if training is True:
    #         n_loop = self.n_loop_training
    #         log_prob = self.summary["training"]["log_prob"]
    #     else:
    #         n_loop = self.n_loop_production
    #         log_prob = self.summary["production"]["log_prob"]

    #     hist = [
    #         jnp.histogram(
    #             log_prob[
    #                 :,
    #                 i
    #                 * (self.n_local_steps // self.output_thinning - 1) : (i + 1)
    #                 * (self.n_local_steps // self.output_thinning - 1),
    #             ].mean(axis=1),
    #             bins=n_bins,
    #         )
    #         for i in range(n_loop)
    #     ]
    #     axis = jnp.array([hist[i][1][:-1] for i in range(n_loop)]).T
    #     hist = jnp.array([hist[i][0] for i in range(n_loop)]).T
    #     return axis, hist

    # def save_summary(self, path: str):
    #     """
    #     Save the summary to a file.

    #     Args:
    #         path (str): Path to save the summary.
    #     """
    #     with open(path, "wb") as f:
    #         pickle.dump(self.summary, f)

    # def print_summary(self) -> None:
    #     """
    #     Print summary to the screen about log probabilities and local/global acceptance rates.
    #     """
    #     train_summary = self.get_sampler_state(training=True)
    #     production_summary = self.get_sampler_state(training=False)

    #     training_log_prob = train_summary["log_prob"]
    #     training_local_acceptance = train_summary["local_accs"]
    #     training_global_acceptance = train_summary["global_accs"]
    #     training_loss = train_summary["loss_vals"]

    #     production_log_prob = production_summary["log_prob"]
    #     production_local_acceptance = production_summary["local_accs"]
    #     production_global_acceptance = production_summary["global_accs"]

    #     print("Training summary")
    #     print("=" * 10)
    #     print(
    #         f"Log probability: {training_log_prob.mean():.3f} +/- {training_log_prob.std():.3f}"
    #     )
    #     print(
    #         f"Local acceptance: {training_local_acceptance.mean():.3f} +/- {training_local_acceptance.std():.3f}"
    #     )
    #     print(
    #         f"Global acceptance: {training_global_acceptance.mean():.3f} +/- {training_global_acceptance.std():.3f}"
    #     )
    #     print(
    #         f"Max loss: {training_loss.max():.3f}, Min loss: {training_loss.min():.3f}"
    #     )

    #     print("Production summary")
    #     print("=" * 10)
    #     print(
    #         f"Log probability: {production_log_prob.mean():.3f} +/- {production_log_prob.std():.3f}"
    #     )
    #     print(
    #         f"Local acceptance: {production_local_acceptance.mean():.3f} +/- {production_local_acceptance.std():.3f}"
    #     )
    #     print(
    #         f"Global acceptance: {production_global_acceptance.mean():.3f} +/- {production_global_acceptance.std():.3f}"
    #     )
