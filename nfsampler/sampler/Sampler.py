import jax
import jax.numpy as jnp
import numpy as np
from nfsampler.nfmodel.utils import sample_nf,train_flow
from nfsampler.sampler.NF_proposal import nf_metropolis_sampler
from flax.training import train_state  # Useful dataclass to keep train state
import optax                           # Optimizers

class Sampler:
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



    def __init__(self, n_dim: int, rng_key_set, nf_model, local_sampler,
                likelihood, 
                d_likelihood = None,
                n_loop: int = 2,
                n_local_steps: int = 5,
                n_global_steps: int = 5,
                n_chains: int = 5,
                n_epochs: int = 5,
                n_nf_samples: int = 100,
                learning_rate: float = 0.01,
                momentum: float = 0.9,
                batch_size: int = 10,
                stepsize: float = 1e-3,
                logging: bool = True):
        rng_key_init ,rng_keys_mcmc, rng_keys_nf, init_rng_keys_nf = rng_key_set

        self.local_sampler = local_sampler
        self.likelihood = likelihood
        self.d_likelihood = d_likelihood
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
        self.momentum = momentum
        self.batch_size = batch_size
        self.stepsize = stepsize
        self.logging = logging

        self.nf_model = nf_model
        params = nf_model.init(init_rng_keys_nf, jnp.ones((self.batch_size,self.n_dim)))['params']

        tx = optax.adam(self.learning_rate, self.momentum)
        self.state = train_state.TrainState.create(apply_fn=nf_model.apply,
                                                   params=params, tx=tx)

    def sample(self,initial_position):
        """
        Sample from the posterior using the local sampler.

        Args:
            initial_position (Device Array): Initial position.

        Returns:
            samples (Device Array): Samples from the posterior.
            log_prob (Device Array): Log probability of the samples.
        """

        last_step = initial_position
        chains = []
        local_accs = []
        global_accs = []
        loss_vals = []

        rng_keys_nf = self.rng_keys_nf
        rng_keys_mcmc = self.rng_keys_mcmc
        state = self.state

        for i in range(self.n_loop):
            rng_keys_nf, rng_keys_mcmc, state, positions, local_acceptance, global_acceptance, loss_values = self.sampling_loop(
                rng_keys_nf, rng_keys_mcmc, state, last_step,)
            last_step = positions[:,-1]
            chains.append(positions)
            local_accs.append(local_acceptance)
            global_accs.append(global_acceptance)
            loss_vals.append(loss_values)


        chains = jnp.concatenate(chains, axis=1)
        local_accs = jnp.stack(local_accs, axis=1).reshape(chains.shape[0], -1)
        global_accs = jnp.stack(global_accs, axis=1).reshape(chains.shape[0], -1)
        loss_vals = jnp.concatenate(jnp.array(loss_vals))

        nf_samples = sample_nf(self.nf_model, state.params, rng_keys_nf, self.n_nf_samples)
        return chains, nf_samples, local_accs, global_accs, loss_vals

    def sampling_loop(self,rng_keys_nf,
                rng_keys_mcmc,
                state,
                initial_position):

        """
        Sampling loop for both the global sampler and the local sampler.

        Args:
            rng_keys_nf (Device Array): RNG keys for the normalizing flow global sampler.
            rng_keys_mcmc (Device Array): RNG keys for the local sampler.
            d_likelihood ?
            TODO: likelihood vs posterior?
            TODO: nf_samples - sometime int, sometimes samples 

        """

        if self.d_likelihood is None:
            rng_keys_mcmc, positions, log_prob, local_acceptance = self.local_sampler(
                rng_keys_mcmc, self.n_local_steps, self.likelihood, initial_position, self.stepsize
                )
        else:
            rng_keys_mcmc, positions, log_prob, local_acceptance = self.local_sampler(
                rng_keys_mcmc, self.n_local_steps, self.likelihood, self.d_likelihood, initial_position, self.stepsize
                )

        flat_chain = positions.reshape(-1, self.n_dim)
        rng_keys_nf, state, loss_values = train_flow(rng_keys_nf, self.nf_model, state, flat_chain,
                                        self.n_epochs, self.batch_size)
        likelihood_vec = jax.vmap(self.likelihood)
        rng_keys_nf, nf_chain, log_prob, log_prob_nf, global_acceptance = nf_metropolis_sampler(
            rng_keys_nf, self.n_global_steps, self.nf_model, state.params , likelihood_vec,
            positions[:,-1]
            )

        positions = jnp.concatenate((positions,nf_chain),axis=1)

        return rng_keys_nf, rng_keys_mcmc, state, positions, local_acceptance, \
            global_acceptance, loss_values



