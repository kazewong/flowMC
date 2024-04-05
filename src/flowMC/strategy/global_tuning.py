from flowMC.proposal.base import ProposalBase
from flowMC.proposal.NF_proposal import NFProposal
from flowMC.strategy.base import Strategy
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray, PyTree
import optax
import equinox as eqx
from tqdm import tqdm


class GlobalTuning(Strategy):

    optim: optax.GradientTransformation
    optim_state: optax.OptState

    n_dim: int
    n_chains: int
    n_local_steps: int
    n_global_steps: int
    n_loop: int
    output_thinning: int
    train_thinning: int

    n_epochs: int
    batch_size: int
    n_max_examples: int
    verbose: bool

    @property
    def __name__(self):
        return "GlobalTuning"

    def __init__(
        self,
        **kwargs,
    ):
        class_keys = list(self.__class__.__annotations__.keys())
        for key, value in kwargs.items():
            if key in class_keys:
                if not key.startswith("__"):
                    setattr(self, key, value)

    def __call__(
        self,
        rng_key: PRNGKeyArray,
        local_sampler: ProposalBase,
        global_sampler: NFProposal,
        initial_position: Float[Array, "n_chains n_dim"],
        data: dict,
    ) -> tuple[
        PRNGKeyArray,
        Float[Array, "n_chains n_dim"],
        ProposalBase,
        NFProposal,
        PyTree,
    ]:
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

        summary = {}
        summary["chains"] = jnp.empty((self.n_chains, 0, self.n_dim))
        summary["log_prob"] = jnp.empty((self.n_chains, 0))
        summary["local_accs"] = jnp.empty((self.n_chains, 0))
        summary["global_accs"] = jnp.empty((self.n_chains, 0))
        summary["loss_vals"] = jnp.empty((0, self.n_epochs))
        for _ in tqdm(
            range(self.n_loop),
            desc="Global Tuning",
        ):
            rng_key, rng_keys_mcmc = jax.random.split(rng_key)
            rng_keys_mcmc = jax.random.split(rng_keys_mcmc, self.n_chains)
            (
                rng_keys_mcmc,
                positions,
                log_prob,
                local_acceptance,
            ) = local_sampler.sample(
                rng_keys_mcmc,
                self.n_local_steps,
                initial_position,
                data,
                verbose=self.verbose,
            )

            summary["chains"] = jnp.append(
                summary["chains"],
                positions[:, :: self.output_thinning],
                axis=1,
            )
            summary["log_prob"] = jnp.append(
                summary["log_prob"],
                log_prob[:, :: self.output_thinning],
                axis=1,
            )

            summary["local_accs"] = jnp.append(
                summary["local_accs"],
                local_acceptance[:, 1 :: self.output_thinning],
                axis=1,
            )

            rng_key, rng_keys_nf = jax.random.split(rng_key)
            positions = summary["chains"][:, :: self.train_thinning]
            chain_size = positions.shape[0] * positions.shape[1]
            if chain_size > self.n_max_examples:
                flat_chain = positions[
                    :, -int(self.n_max_examples / self.n_chains) :
                ].reshape(-1, self.n_dim)
            else:
                flat_chain = positions.reshape(-1, self.n_dim)

            if flat_chain.shape[0] < self.n_max_examples:
                # This is to pad the training data to avoid recompilation.
                flat_chain = jnp.repeat(
                    flat_chain,
                    (self.n_max_examples // flat_chain.shape[0]) + 1,
                    axis=0,
                )
                flat_chain = flat_chain[: self.n_max_examples]

            data_mean = jnp.mean(flat_chain, axis=0)
            data_cov = jnp.cov(flat_chain.T)
            global_sampler.model = eqx.tree_at(
                lambda m: m._data_mean, global_sampler.model, data_mean
            )
            global_sampler.model = eqx.tree_at(
                lambda m: m._data_cov, global_sampler.model, data_cov
            )

            (
                rng_keys_nf,
                global_sampler.model,
                self.optim_state,
                loss_values,
            ) = global_sampler.model.train(
                rng_keys_nf,
                flat_chain,
                self.optim,
                self.optim_state,
                self.n_epochs,
                self.batch_size,
                self.verbose,
            )
            summary["loss_vals"] = jnp.append(
                summary["loss_vals"],
                loss_values.reshape(1, -1),
                axis=0,
            )

            (
                rng_keys_nf,
                nf_chain,
                log_prob,
                global_acceptance,
            ) = global_sampler.sample(
                rng_keys_nf,
                self.n_global_steps,
                positions[:, -1],
                data,
                verbose=self.verbose,
            )

            summary["chains"] = jnp.append(
                summary["chains"],
                nf_chain[:, :: self.output_thinning],
                axis=1,
            )
            summary["log_prob"] = jnp.append(
                summary["log_prob"],
                log_prob[:, :: self.output_thinning],
                axis=1,
            )

            summary["global_accs"] = jnp.append(
                summary["global_accs"],
                global_acceptance[:, 1 :: self.output_thinning],
                axis=1,
            )

        return rng_key, summary['chains'][:, -1], local_sampler, global_sampler, summary


class GlobalSampling(Strategy):

    n_dim: int
    n_chains: int
    n_local_steps: int
    n_global_steps: int
    n_loop: int
    output_thinning: int
    verbose: bool

    @property
    def __name__(self):
        return "GlobalSampling"

    def __init__(
        self,
        **kwargs,
    ):
        class_keys = list(self.__class__.__annotations__.keys())
        print(class_keys)
        for key, value in kwargs.items():
            if key in class_keys:
                if not key.startswith("__"):
                    setattr(self, key, value)

    def __call__(
        self,
        rng_key: PRNGKeyArray,
        local_sampler: ProposalBase,
        global_sampler: NFProposal,
        initial_position: Float[Array, "n_chains n_dim"],
        data: dict,
    ) -> tuple[
        PRNGKeyArray,
        Float[Array, "n_chains n_dim"],
        ProposalBase,
        NFProposal,
        PyTree,
    ]:

        summary = {}
        summary["chains"] = jnp.empty((self.n_chains, 0, self.n_dim))
        summary["log_prob"] = jnp.empty((self.n_chains, 0))
        summary["local_accs"] = jnp.empty((self.n_chains, 0))
        summary["global_accs"] = jnp.empty((self.n_chains, 0))

        for _ in tqdm(
            range(self.n_loop),
            desc="Global Sampling",
        ):
            rng_key, rng_keys_mcmc = jax.random.split(rng_key)
            rng_keys_mcmc = jax.random.split(rng_keys_mcmc, self.n_chains)
            (
                rng_keys_mcmc,
                positions,
                log_prob,
                local_acceptance,
            ) = local_sampler.sample(
                rng_keys_mcmc,
                self.n_local_steps,
                initial_position,
                data,
                verbose=self.verbose,
            )

            summary["chains"] = jnp.append(
                summary["chains"],
                positions[:, :: self.output_thinning],
                axis=1,
            )
            summary["log_prob"] = jnp.append(
                summary["log_prob"],
                log_prob[:, :: self.output_thinning],
                axis=1,
            )

            summary["local_accs"] = jnp.append(
                summary["local_accs"],
                local_acceptance[:, 1 :: self.output_thinning],
                axis=1,
            )

            rng_key, rng_keys_nf = jax.random.split(rng_key)
            (
                rng_keys_nf,
                nf_chain,
                log_prob,
                global_acceptance,
            ) = global_sampler.sample(
                rng_keys_nf,
                self.n_global_steps,
                positions[:, -1],
                data,
                verbose=self.verbose,
            )

            summary["chains"] = jnp.append(
                summary["chains"],
                nf_chain[:, :: self.output_thinning],
                axis=1,
            )
            summary["log_prob"] = jnp.append(
                summary["log_prob"],
                log_prob[:, :: self.output_thinning],
                axis=1,
            )

            summary["global_accs"] = jnp.append(
                summary["global_accs"],
                global_acceptance[:, 1 :: self.output_thinning],
                axis=1,
            )

        return rng_key, summary['chains'][:, -1], local_sampler, global_sampler, summary
