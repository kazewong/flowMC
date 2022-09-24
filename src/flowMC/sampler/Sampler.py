from logging import lastResort
import jax
import jax.numpy as jnp
import numpy as np
from flowMC.nfmodel.utils import sample_nf, make_training_loop
from flowMC.sampler.NF_proposal import make_nf_metropolis_sampler
from flax.training import train_state  # Useful dataclass to keep train state
import flax
import optax

# Optimizers
class Sampler(object):
    """
    Sampler class that host configuration parameters, NF model, and local sampler

    Args:
        rng_key_set (Device Array): RNG keys set generated using initialize_rng_keys.
        config (dict): Configuration parameters.
        nf_model (flax module): Normalizing flow model.
        local_sampler (function): Local sampler function.
        likelihood (function): Likelihood function.
        d_likelihood (Device Array): Derivative of the likelihood function.
    """

    def __init__(
        self,
        n_dim: int,
        rng_key_set,
        local_sampler,
        sampler_params: dict,
        likelihood,
        nf_model,
        n_loop: int = 2,
        n_local_steps: int = 5,
        n_global_steps: int = 5,
        n_chains: int = 5,
        n_epochs: int = 5,
        n_nf_samples: int = 100,
        learning_rate: float = 0.01,
        max_samples: int = 10000,
        momentum: float = 0.9,
        batch_size: int = 10,
        use_global: bool = True,
        logging: bool = True,
        nf_variable=None,
        keep_quantile=0,
        local_autotune=None,
    ):
        rng_key_init, rng_keys_mcmc, rng_keys_nf, init_rng_keys_nf = rng_key_set

        self.likelihood = likelihood
        self.likelihood_vec = jax.jit(jax.vmap(self.likelihood))
        self.sampler_params = sampler_params
        self.local_sampler = local_sampler(likelihood)
        self.local_autotune= local_autotune

        self.rng_keys_nf = rng_keys_nf
        self.rng_keys_mcmc = rng_keys_mcmc
        self.n_dim = n_dim
        self.n_loop = n_loop
        self.n_local_steps = n_local_steps
        self.n_global_steps = n_global_steps
        self.n_chains = n_chains
        self.n_epochs = n_epochs
        self.n_nf_samples = n_nf_samples
        self.learning_rate = learning_rate
        self.max_samples = max_samples
        self.momentum = momentum
        self.batch_size = batch_size
        self.use_global = use_global
        self.logging = logging


        self.nf_model = nf_model
        model_init = nf_model.init(init_rng_keys_nf, jnp.ones((1, self.n_dim)))
        params = model_init["params"]
        self.variables = model_init["variables"]
        if nf_variable is not None:
            self.variables = self.variables

        self.keep_quantile = keep_quantile
        self.nf_training_loop, train_epoch, train_step = make_training_loop(
            self.nf_model
        )
        self.global_sampler = make_nf_metropolis_sampler(self.nf_model)

        tx = optax.adam(self.learning_rate, self.momentum)
        self.state = train_state.TrainState.create(
            apply_fn=nf_model.apply, params=params, tx=tx
        )

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
        self.summary['training'] = training
        self.summary['production'] = production

    def sample(self, initial_position):
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

        self.local_sampler_tuning(self.n_local_steps, initial_position)
        last_step = initial_position
        if self.use_global == True:
            last_step = self.global_sampler_tuning(last_step)

        last_step = self.production_run(last_step)

    def sampling_loop(self, initial_position, training=False):

        """
        Sampling loop for both the global sampler and the local sampler.

        Args:
            rng_keys_nf (Device Array): RNG keys for the normalizing flow global sampler.
            rng_keys_mcmc (Device Array): RNG keys for the local sampler.
            d_likelihood ?
            TODO: likelihood vs posterior?
            TODO: nf_samples - sometime int, sometimes samples

        """

        self.rng_keys_mcmc, positions, log_prob, local_acceptance, _ = self.local_sampler(
            self.rng_keys_mcmc, self.n_local_steps, initial_position, self.sampler_params
        )

        log_prob_output = np.copy(log_prob)

        if self.use_global == True:
            if training == True:
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

                variables = self.variables.unfreeze()
                variables["base_mean"] = jnp.mean(flat_chain, axis=0)
                variables["base_cov"] = jnp.cov(flat_chain.T)
                self.variables = flax.core.freeze(variables)

                flat_chain = (flat_chain - variables["base_mean"]) / jnp.sqrt(
                    jnp.diag(variables["base_cov"])
                )

                self.rng_keys_nf, self.state, loss_values = self.nf_training_loop(
                    self.rng_keys_nf,
                    self.state,
                    self.variables,
                    flat_chain,
                    self.n_epochs,
                    self.batch_size,
                )
                self.summary['training']['loss_vals'] = jnp.append(
                    self.summary['training']['loss_vals'], loss_values.reshape(1, -1), axis=0
                )

            (
                self.rng_keys_nf,
                nf_chain,
                log_prob,
                log_prob_nf,
                global_acceptance,
            ) = self.global_sampler(
                self.rng_keys_nf,
                self.n_global_steps,
                self.state.params,
                self.variables,
                self.likelihood_vec,
                positions[:, -1],
            )

            positions = jnp.concatenate((positions, nf_chain), axis=1)
            log_prob_output = jnp.concatenate((log_prob_output, log_prob), axis=1)

        if training == True:
            self.summary['training']['chains'] = jnp.append(
                self.summary['training']['chains'], positions, axis=1
            )
            self.summary['training']['log_prob'] = jnp.append(
                self.summary['training']['log_prob'], log_prob_output, axis=1
            )
            self.summary['training']['local_accs'] = jnp.append(
                self.summary['training']['local_accs'], local_acceptance, axis=1
            )
            if self.use_global == True:
                self.summary['training']['global_accs'] = jnp.append(
                    self.summary['training']['global_accs'], global_acceptance, axis=1
                )
        else:
            self.summary['production']['chains'] = jnp.append(
                self.summary['production']['chains'], positions, axis=1
            )
            self.summary['production']['log_prob'] = jnp.append(
                self.summary['production']['log_prob'], log_prob_output, axis=1
            )
            self.summary['production']['local_accs'] = jnp.append(
                self.summary['production']['local_accs'], local_acceptance, axis=1
            )
            if self.use_global == True:
                self.summary['production']['global_accs'] = jnp.append(
                    self.summary['production']['global_accs'], global_acceptance, axis=1
                )
        last_step = positions[:, -1]

        return last_step

    def local_sampler_tuning(self, n_steps, initial_position, max_iter=10):
        if self.local_autotune is not None:
            print("Autotune found, start tuning sampler_params")
            self.sampler_params, self.local_sampler = self.local_autotune(self.local_sampler, self.rng_keys_mcmc, n_steps, initial_position, self.sampler_params, max_iter)
        else:
            print("No autotune found, use input sampler_params")

    def global_sampler_tuning(self,initial_position):
        print("Training normalizing flow")
        last_step = initial_position
        for _ in range(self.n_loop):
            last_step = self.sampling_loop(last_step, training=True)
        return last_step
        
    def production_run(self, initial_position):
        last_step = initial_position
        for _ in range(self.n_loop):
            self.sampling_loop(last_step)

    def get_sampler_state(self, training=False):
        if training == True:
            return self.summary['training']
        else:
            return self.summary['production']

    def sample_flow(self, n_samples=None):
        if n_samples is None:
            n_samples = self.n_nf_samples
        nf_samples = sample_nf(
            self.nf_model,
            self.state.params,
            self.rng_keys_nf,
            n_samples,
            self.variables,
        )
        return nf_samples

    def reset(self):
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
        self.summary['training'] = training
        self.summary['production'] = production