import pickle
from typing import Tuple
import jax.numpy as jnp
from jaxtyping import Array, Int, Float
from flowMC.nfmodel.utils import make_training_loop
from flowMC.sampler.NF_proposal import NFProposal
import optax
from flowMC.sampler.Proposal_Base import ProposalBase
from flowMC.nfmodel.base import NFModel
from tqdm import tqdm
import equinox as eqx
import numpy as np

default_hyperparameters = {
    "n_loop_training": 3,
    "n_loop_production": 3,
    "n_local_steps": 50,
    "n_global_steps": 50,
    "n_chains": 20,
    "n_epochs": 30,
    "learning_rate": 0.001,
    "max_samples": 100000,
    "momentum": 0.9,
    "batch_size": 10000,
    "use_global": True,
    "global_sampler": None,
    "logging": True,
    "keep_quantile": 0,
    "local_autotune": None,
    "train_thinning": 1,
    "output_thinning": 1,
    "n_sample_max": 10000,
    "precompile": False,
    "verbose": False,
    "outdir": "./outdir/",
}

class Sampler:
    """
    Sampler class that host configuration parameters, NF model, and local sampler

    Args:
        "n_loop_training": "(int): Number of training loops.",
        "n_loop_production": "(int): Number of production loops.",
        "n_local_steps": "(int) Number of local steps per loop.",
        "n_global_steps": "(int) Number of local steps per loop.",
        "n_chains": "(int) Number of chains",
        "n_epochs": "(int) Number of epochs to train the NF per training loop",
        "learning_rate": "(float) Learning rate used in the training of the NF",
        "max_samples": "(int) Maximum number of samples fed to training the NF model",
        "momentum": "(float) Momentum used in the training of the NF model with the Adam optimizer",
        "batch_size": "(int) Size of batches used to train the NF",
        "use_global": "(bool) Whether to use an NF proposal as global sampler",
        "global_sampler": "(NFProposal) Global sampler",
        "logging": "(bool) Whether to log the training process",
        "keep_quantile": "Quantile of chains to keep when training the normalizing flow model",
        "local_autotune": "(Callable) Auto-tune function for the local sampler",
        "train_thinning": "(int) Thinning parameter for the data used to train the normalizing flow",
        "output_thinning": "(int) Thinning parameter with which to save the data ",
        "n_sample_max": "(int) Maximum number of samples fed to training the NF model",
        "precompile": "(bool) Whether to precompile",
        "verbose": "(bool) Show steps of algorithm in detail",
        "outdir": "(str) Location to which to save plots, samples and hyperparameter settings. Note: should ideally start with `./` and also end with `/`"
    """

    @property
    def nf_model(self):
        return self.global_sampler.model

    def __init__(
        self,
        n_dim: int,
        rng_key_set: Tuple,
        data: jnp.ndarray,
        local_sampler: ProposalBase,
        nf_model: NFModel,
        **kwargs,
    ):
        rng_key_init, rng_keys_mcmc, rng_keys_nf, init_rng_keys_nf = rng_key_set

        # Copying input into the model

        self.rng_keys_nf = rng_keys_nf
        self.rng_keys_mcmc = rng_keys_mcmc
        self.n_dim = n_dim


        # Set and override any given hyperparameters
        self.hyperparameters = default_hyperparameters
        hyperparameter_names = list(default_hyperparameters.keys())
        for key, value in kwargs.items():
            if key in hyperparameter_names:
                self.hyperparameters[key] = value
        for key, value in self.hyperparameters.items():
            setattr(self, key, value)

        self.variables = {"mean": None, "var": None}

        # Initialized local and global samplers

        self.local_sampler = local_sampler
        if self.precompile:
            self.local_sampler.precompilation(
                n_chains=self.n_chains, n_dims=n_dim, n_step=self.n_local_steps, data=data
            )

        if self.global_sampler is None:
            self.global_sampler = NFProposal(self.local_sampler.logpdf, jit=self.local_sampler.jit, model=nf_model, n_sample_max=self.n_sample_max)

        self.likelihood_vec = self.local_sampler.logpdf_vmap

        tx = optax.chain(optax.clip(1.0), optax.adam(self.learning_rate, self.momentum))
        self.optim_state = tx.init(eqx.filter(self.nf_model, eqx.is_array))
        self.nf_training_loop, train_epoch, train_step = make_training_loop(tx)

        # Initialized result dictionary
        training = {}
        training["chains"] = jnp.empty((self.n_chains, 0, self.n_dim))
        training["log_prob"] = jnp.empty((self.n_chains, 0))
        training["local_accs"] = jnp.empty((self.n_chains, 0))
        training["global_accs"] = jnp.empty((self.n_chains, 0))
        training["loss_vals"] = jnp.empty((0, self.n_epochs))

        production = {}
        production["chains"] = jnp.empty((self.n_chains, 0, self.n_dim))
        production["log_prob"] = jnp.empty((self.n_chains, 0))
        production["local_accs"] = jnp.empty((self.n_chains, 0))
        production["global_accs"] = jnp.empty((self.n_chains, 0))

        self.summary = {}
        self.summary["training"] = training
        self.summary["production"] = production


    def sample(self, initial_position: Array, data: dict):
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

        self.local_sampler_tuning(initial_position, data)
        last_step = initial_position
        if self.use_global is True:
            last_step = self.global_sampler_tuning(last_step, data)

        last_step = self.production_run(last_step, data)

    def sampling_loop(
        self, initial_position: jnp.array, data: jnp.array, training=False
    ) -> jnp.array:
        """
        One sampling loop that iterate through the local sampler
        and potentially the global sampler.
        If training is set to True, the loop will also train the normalizing flow model.

        Args:
            initial_position (jnp.array): Initial position. Shape (n_chains, n_dim)
            training (bool, optional):
            Whether to train the normalizing flow model. Defaults to False.

        Returns:
            chains (jnp.array):
            Samples from the posterior. Shape (n_chains, n_local_steps + n_global_steps, n_dim)
        """

        if training is True:
            summary_mode = "training"
        else:
            summary_mode = "production"

        (
            self.rng_keys_mcmc,
            positions,
            log_prob,
            local_acceptance,
        ) = self.local_sampler.sample(
            self.rng_keys_mcmc,
            self.n_local_steps,
            initial_position,
            data,
            verbose=self.verbose,
        )

        self.summary[summary_mode]["chains"] = jnp.append(
            self.summary[summary_mode]["chains"],
            positions[:, :: self.output_thinning],
            axis=1,
        )
        self.summary[summary_mode]["log_prob"] = jnp.append(
            self.summary[summary_mode]["log_prob"],
            log_prob[:, :: self.output_thinning],
            axis=1,
        )

        self.summary[summary_mode]["local_accs"] = jnp.append(
            self.summary[summary_mode]["local_accs"],
            local_acceptance[:, 1 :: self.output_thinning],
            axis=1,
        )

        if self.use_global is True:
            if training is True:
                positions = self.summary["training"]["chains"][
                    :, :: self.train_thinning
                ]
                log_prob_output = self.summary["training"]["log_prob"][
                    :, :: self.train_thinning
                ]

                if self.keep_quantile > 0:
                    max_log_prob = jnp.max(log_prob_output, axis=1)
                    cut = jnp.quantile(max_log_prob, self.keep_quantile)
                    cut_chains = positions[max_log_prob > cut]
                else:
                    cut_chains = positions
                chain_size = cut_chains.shape[0] * cut_chains.shape[1]
                if chain_size > self.max_samples:
                    flat_chain = cut_chains[
                        :, -int(self.max_samples / self.n_chains) :
                    ].reshape(-1, self.n_dim)
                else:
                    flat_chain = cut_chains.reshape(-1, self.n_dim)

                if flat_chain.shape[0] < self.max_samples:
                    # This is to pad the training data to avoid recompilation.
                    flat_chain = jnp.repeat(
                        flat_chain,
                        (self.max_samples // flat_chain.shape[0]) + 1,
                        axis=0,
                    )
                    flat_chain = flat_chain[: self.max_samples]

                self.variables["mean"] = jnp.mean(flat_chain, axis=0)
                self.variables["cov"] = jnp.cov(flat_chain.T)
                self.global_sampler.model = eqx.tree_at(
                    lambda m: m._data_mean, self.nf_model, self.variables["mean"]
                )
                self.global_sampler.model = eqx.tree_at(
                    lambda m: m._data_cov, self.nf_model, self.variables["cov"]
                )

                (
                    self.rng_keys_nf,
                    self.global_sampler.model,
                    self.optim_state,
                    loss_values,
                ) = self.nf_training_loop(
                    self.rng_keys_nf,
                    self.nf_model,
                    flat_chain,
                    self.optim_state,
                    self.n_epochs,
                    self.batch_size,
                    self.verbose,
                )
                self.summary["training"]["loss_vals"] = jnp.append(
                    self.summary["training"]["loss_vals"],
                    loss_values.reshape(1, -1),
                    axis=0,
                )

            (
                self.rng_keys_nf,
                nf_chain,
                log_prob,
                global_acceptance,
            ) = self.global_sampler.sample(
                self.rng_keys_nf,
                self.n_global_steps,
                positions[:, -1],
                data,
                verbose = self.verbose,
                mode = summary_mode
            )

            self.summary[summary_mode]["chains"] = jnp.append(
                self.summary[summary_mode]["chains"],
                nf_chain[:, :: self.output_thinning],
                axis=1,
            )
            self.summary[summary_mode]["log_prob"] = jnp.append(
                self.summary[summary_mode]["log_prob"],
                log_prob[:, :: self.output_thinning],
                axis=1,
            )

            self.summary[summary_mode]["global_accs"] = jnp.append(
                self.summary[summary_mode]["global_accs"],
                global_acceptance[:, 1 :: self.output_thinning],
                axis=1,
            )

        last_step = self.summary[summary_mode]["chains"][:, -1]

        return last_step

    def local_sampler_tuning(
        self, initial_position: jnp.array, data: jnp.array, max_iter: int = 100
    ):
        """
        Tuning the local sampler. This runs a number of iterations of the local sampler,
        and then uses the acceptance rate to adjust the local sampler parameters.
        Since this is mostly for a fast adaptation, we do not carry the sample state forward.
        Instead, we only adapt the sampler parameters using the initial position.

        Args:
            n_steps (int): Number of steps to run the local sampler.
            initial_position (Device Array): Initial position for the local sampler.
            max_iter (int): Number of iterations to run the local sampler.
        """
        if self.local_autotune is not None:
            print("Autotune found, start tuning sampler_params")
            kernel_vmap = self.local_sampler.kernel_vmap
            self.local_sampler.params = self.local_autotune(
                kernel_vmap,
                self.rng_keys_mcmc,
                initial_position,
                self.likelihood_vec(initial_position),
                data,
                self.local_sampler.params,
                max_iter,
            )
        else:
            print("No autotune found, use input sampler_params")

    def global_sampler_tuning(
        self, initial_position: jnp.ndarray, data: jnp.array
    ) -> jnp.array:
        """
        Tuning the global sampler. This runs both the local sampler and the global sampler,
        and train the normalizing flow on the run.
        To adapt the normalizing flow, we need to keep certain amount of the data generated during the sampling.
        The data is stored in the summary dictionary and can be accessed through the `get_sampler_state` method.
        This tuning run is meant to be followed by a production run as defined below.

        Args:
            initial_position (Device Array): Initial position for the sampler, shape (n_chains, n_dim)

        """
        print("Training normalizing flow")
        last_step = initial_position
        for _ in tqdm(
            range(self.n_loop_training),
            desc="Tuning global sampler",
        ):
            last_step = self.sampling_loop(last_step, data, training=True)
        return last_step

    def production_run(
        self, initial_position: jnp.ndarray, data: jnp.array
    ) -> jnp.array:
        """
        Sampling procedure that produce the final set of samples.
        The main difference between this and the global tuning step is
        we do not train the normalizing flow, omitting training allows to maintain detail balance.
        The data is stored in the summary dictionary and can be accessed through the `get_sampler_state` method.

        Args:
            initial_position (Device Array): Initial position for the sampler, shape (n_chains, n_dim)

        """
        print("Starting Production run")
        last_step = initial_position
        for _ in tqdm(
            range(self.n_loop_production),
            desc="Production run",
        ):
            last_step = self.sampling_loop(last_step, data)
        return last_step

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
            return self.summary["training"]
        else:
            return self.summary["production"]

    def sample_flow(self, n_samples: int) -> jnp.ndarray:
        """
        Sample from the normalizing flow.

        Args:
            n_samples (int): Number of samples to generate.

        Returns:
            Device Array: Samples generated using the normalizing flow.
        """

        samples = self.nf_model.sample(self.rng_keys_nf, n_samples)
        return samples

    def evalulate_flow(self, samples: jnp.ndarray) -> jnp.ndarray:
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
        self.nf_model = self.nf_model.load_model(path)

    def reset(self):
        """
        Reset the sampler state.

        """
        training = {}
        training["chains"] = jnp.empty((self.n_chains, 0, self.n_dim))
        training["log_prob"] = jnp.empty((self.n_chains, 0))
        training["local_accs"] = jnp.empty((self.n_chains, 0))
        training["global_accs"] = jnp.empty((self.n_chains, 0))
        training["loss_vals"] = jnp.empty((0, self.n_epochs))

        production = {}
        production["chains"] = jnp.empty((self.n_chains, 0, self.n_dim))
        production["log_prob"] = jnp.empty((self.n_chains, 0))
        production["local_accs"] = jnp.empty((self.n_chains, 0))
        production["global_accs"] = jnp.empty((self.n_chains, 0))

        self.summary = {}
        self.summary["training"] = training
        self.summary["production"] = production

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
            np.histogram(
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
        axis = np.array([hist[i][1][:-1] for i in range(n_loop)]).T
        hist = np.array([hist[i][0] for i in range(n_loop)]).T
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
            np.histogram(
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
        axis = np.array([hist[i][1][:-1] for i in range(n_loop)]).T
        hist = np.array([hist[i][0] for i in range(n_loop)]).T
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
            np.histogram(
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
        axis = np.array([hist[i][1][:-1] for i in range(n_loop)]).T
        hist = np.array([hist[i][0] for i in range(n_loop)]).T
        return axis, hist

    def save_summary(self, path: str):
        """
        Save the summary to a file.

        Args:
            path (str): Path to save the summary.
        """
        with open(path, "wb") as f:
            pickle.dump(self.summary, f)
