import pickle
import jax
import jax.numpy as jnp
from jaxtyping import Array, Int, Float, PRNGKeyArray
from tqdm import tqdm
import equinox as eqx
import optax
from flowMC.proposal.NF_proposal import NFProposal
from flowMC.proposal.base import ProposalBase
from flowMC.nfmodel.base import NFModel
from flowMC.strategy.base import Strategy
from flowMC.strategy.global_tuning import GlobalTuning, GlobalSampling


class Sampler:
    """
    Sampler class that host configuration parameters, NF model, and local sampler

    Args:
        n_dim (int): Dimension of the problem.
        rng_key (PRNGKeyArray): Jax PRNGKey.
        data (dict): Data to be passed to the logpdf function.
        local_sampler (ProposalBase): Local sampler.
        nf_model (NFModel): Normalizing flow model.

    Keyword Args:
        n_chains (int): Number of chains.
        n_local_steps (int): Number of local steps.
        n_global_steps (int): Number of global steps.
        n_loop_training (int): Number of training loops.
        n_loop_production (int): Number of production loops.
        train_thinning (int): Thinning parameter for training.
        output_thinning (int): Thinning parameter for sampling.

        use_global (bool): Whether to use the global sampler.
        batch_size (int): Batch size for training.
        n_epochs (int): Number of epochs per training loop
        learning_rate (float): Learning rate of the optimizer.
        momentum (float): Momentum of the optimizer.
        n_max_examples (int): Maximum number of examples per training step.
        n_flow_sample (int): Number of samples to generate from the normalizing flow.

        precompile (bool): Whether to precompile the local sampler.
        verbose (bool): Whether to print verbose output.
        logging (bool): Whether to log the output.
        outdir (str): Output directory.
    """

    # Essential parameters
    n_dim: int
    rng_key: PRNGKeyArray
    data: dict
    local_sampler: ProposalBase
    strategies: list[Strategy]

    # Sampling hyperparameters
    n_chains: int = 20
    n_local_steps: int = 50
    n_global_steps: int = 50
    n_loop_training: int = 3
    n_loop_production: int = 3
    train_thinning: int = 1
    output_thinning: int = 1
    local_autotune: bool = False

    # Normalizing flow hyperparameters
    global_sampler: NFProposal
    use_global: bool = True
    batch_size: int = 10000
    n_epochs: int = 30
    learning_rate: float = 0.001
    momentum: float = 0.9
    n_max_examples: int = 10000
    n_flow_sample: int = 10000

    # Logging hyperparameters
    precompile: bool = False
    verbose: bool = False
    logging: bool = True
    outdir: str = "./outdir/"

    @property
    def nf_model(self):
        return self.global_sampler.model

    def __init__(
        self,
        n_dim: int,
        rng_key: PRNGKeyArray,
        data: dict,
        local_sampler: ProposalBase,
        nf_model: NFModel,
        **kwargs,
    ):
        # Copying input into the model

        self.n_dim = n_dim
        self.rng_key = rng_key
        self.data = data
        self.local_sampler = local_sampler

        # Set and override any given hyperparameters
        class_keys = list(self.__class__.__dict__.keys())
        for key, value in kwargs.items():
            if key in class_keys:
                if not key.startswith("__"):
                    setattr(self, key, value)

        # Initialized local and global samplers

        self.local_sampler = local_sampler
        if self.precompile:
            self.local_sampler.precompilation(
                n_chains=self.n_chains,
                n_dims=n_dim,
                n_step=self.n_local_steps,
                data=data,
            )

        self.global_sampler = NFProposal(
            self.local_sampler.logpdf,
            jit=self.local_sampler.jit,
            model=nf_model,
            n_flow_sample=self.n_flow_sample,
        )

        self.likelihood_vec = self.local_sampler.logpdf_vmap

        self.optim = optax.chain(
            optax.clip(1.0), optax.adam(self.learning_rate, self.momentum)
        )
        self.optim_state = self.optim.init(eqx.filter(self.nf_model, eqx.is_array))

        self.strategies = [
            GlobalTuning(
                n_dim=self.n_dim,
                n_chains=self.n_chains,
                n_local_steps=self.n_local_steps,
                n_global_steps=self.n_global_steps,
                n_loop=self.n_loop_training,
                output_thinning=self.output_thinning,
                train_thinning=self.train_thinning,
                optim=self.optim,
                optim_state=self.optim_state,
                n_epochs=self.n_epochs,
                batch_size=self.batch_size,
                n_max_examples=self.n_max_examples,
                verbose=self.verbose,
            ),
            GlobalSampling(
                n_dim=self.n_dim,
                n_chains=self.n_chains,
                n_local_steps=self.n_local_steps,
                n_global_steps=self.n_global_steps,
                n_loop=self.n_loop_production,
                output_thinning=self.output_thinning,
                verbose=self.verbose,
            ),
        ]

        if kwargs.get("strategies") is not None:
            kwargs_strategies = kwargs.get("strategies")
            assert isinstance(kwargs_strategies, list)
            self.strategies = kwargs_strategies

        self.summary = {}

    def sample(self, initial_position: Float[Array, "n_chains n_dim"], data: dict):
        """
        Sample from the posterior using the local sampler.

        Args:
            initial_position (Device Array): Initial position.

        Returns:
            chains (Device Array): Samples from the posterior.
            nf_samples (Device Array): (n_nf_samples, n_dim)
            local_accs (Device Array): (n_chains, n_local_steps * n_loop)
            global_accs (Device Array): (n_chains, n_global_steps * n_loop)
            loss_vals (Device Array): (n_epoch * n_loop,)
        """

        # Note that auto-tune function needs to have the same number of steps
        # as the actual sampling loop to avoid recompilation.

        initial_position = jnp.atleast_2d(initial_position) # type: ignore
        rng_key = self.rng_key
        last_step = initial_position
        for strategy in self.strategies:
            (
                rng_key,
                last_step,
                self.local_sampler,
                self.global_sampler,
                summary,
            ) = strategy(
                rng_key, self.local_sampler, self.global_sampler, last_step, data
            )
            self.summary[strategy.__name__] = summary

    def get_sampler_state(self, training: bool = False) -> dict:
        """
        Get the sampler state. There are two sets of sampler outputs one can get,
        the training set and the production set.
        The training set is produced during the global tuning step, and the production set
        is produced during the production run.
        Only the training set contains information about the loss function.
        Only the production set should be used to represent the final set of samples.

        Args:
            training (bool): Whether to get the training set sampler state. Defaults to False.

        """
        if training is True:
            return self.summary["GlobalTuning"]
        else:
            return self.summary["GlobalSampling"]

    def sample_flow(
        self, rng_key: PRNGKeyArray, n_samples: int
    ) -> Float[Array, "n_samples n_dim"]:
        """
        Sample from the normalizing flow.

        Args:
            n_samples (int): Number of samples to generate.

        Returns:
            Device Array: Samples generated using the normalizing flow.
        """

        samples = self.nf_model.sample(rng_key, n_samples)
        return samples

    def evalulate_flow(
        self, samples: Float[Array, "n_samples n_dim"]
    ) -> Float[Array, "n_samples"]:
        """
        Evaluate the log probability of the normalizing flow.

        Args:
            samples (Device Array): Samples to evaluate.

        Returns:
            Device Array: Log probability of the samples.
        """
        log_prob = self.nf_model.log_prob(samples)
        return log_prob

    def save_flow(self, path: str):
        """
        Save the normalizing flow to a file.

        Args:
            path (str): Path to save the normalizing flow.
        """
        self.nf_model.save_model(path)

    def load_flow(self, path: str):
        """
        Save the normalizing flow to a file.

        Args:
            path (str): Path to save the normalizing flow.
        """
        self.nf_model.load_model(path)

    def reset(self):
        """
        Reset the sampler state.

        """
        self.summary = {}

    def get_global_acceptance_distribution(
        self, n_bins: int = 10, training: bool = False
    ) -> tuple[Int[Array, "n_bin n_loop"], Float[Array, "n_bin n_loop"]]:
        """
        Get the global acceptance distribution as a histogram per epoch.

        Returns:
            axis (Device Array): Axis of the histogram.
            hist (Device Array): Histogram of the global acceptance distribution.
        """
        if training is True:
            n_loop = self.n_loop_training
            global_accs = self.summary["training"]["global_accs"]
        else:
            n_loop = self.n_loop_production
            global_accs = self.summary["production"]["global_accs"]

        hist = [
            jnp.histogram(
                global_accs[
                    :,
                    i
                    * (self.n_global_steps // self.output_thinning - 1) : (i + 1)
                    * (self.n_global_steps // self.output_thinning - 1),
                ].mean(axis=1),
                bins=n_bins,
            )
            for i in range(n_loop)
        ]
        axis = jnp.array([hist[i][1][:-1] for i in range(n_loop)]).T
        hist = jnp.array([hist[i][0] for i in range(n_loop)]).T
        return axis, hist

    def get_local_acceptance_distribution(
        self, n_bins: int = 10, training: bool = False
    ) -> tuple[Int[Array, "n_bin n_loop"], Float[Array, "n_bin n_loop"]]:
        """
        Get the local acceptance distribution as a histogram per epoch.

        Returns:
            axis (Device Array): Axis of the histogram.
            hist (Device Array): Histogram of the local acceptance distribution.
        """
        if training is True:
            n_loop = self.n_loop_training
            local_accs = self.summary["training"]["local_accs"]
        else:
            n_loop = self.n_loop_production
            local_accs = self.summary["production"]["local_accs"]

        hist = [
            jnp.histogram(
                local_accs[
                    :,
                    i
                    * (self.n_local_steps // self.output_thinning - 1) : (i + 1)
                    * (self.n_local_steps // self.output_thinning - 1),
                ].mean(axis=1),
                bins=n_bins,
            )
            for i in range(n_loop)
        ]
        axis = jnp.array([hist[i][1][:-1] for i in range(n_loop)]).T
        hist = jnp.array([hist[i][0] for i in range(n_loop)]).T
        return axis, hist

    def get_log_prob_distribution(
        self, n_bins: int = 10, training: bool = False
    ) -> tuple[Int[Array, "n_bin n_loop"], Float[Array, "n_bin n_loop"]]:
        """
        Get the log probability distribution as a histogram per epoch.

        Returns:
            axis (Device Array): Axis of the histogram.
            hist (Device Array): Histogram of the log probability distribution.
        """
        if training is True:
            n_loop = self.n_loop_training
            log_prob = self.summary["training"]["log_prob"]
        else:
            n_loop = self.n_loop_production
            log_prob = self.summary["production"]["log_prob"]

        hist = [
            jnp.histogram(
                log_prob[
                    :,
                    i
                    * (self.n_local_steps // self.output_thinning - 1) : (i + 1)
                    * (self.n_local_steps // self.output_thinning - 1),
                ].mean(axis=1),
                bins=n_bins,
            )
            for i in range(n_loop)
        ]
        axis = jnp.array([hist[i][1][:-1] for i in range(n_loop)]).T
        hist = jnp.array([hist[i][0] for i in range(n_loop)]).T
        return axis, hist

    def save_summary(self, path: str):
        """
        Save the summary to a file.

        Args:
            path (str): Path to save the summary.
        """
        with open(path, "wb") as f:
            pickle.dump(self.summary, f)
