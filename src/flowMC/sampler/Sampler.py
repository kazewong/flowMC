from logging import lastResort
import jax
import jax.numpy as jnp
import numpy as np
from flowMC.nfmodel.utils import sample_nf,train_flow
from flowMC.sampler.NF_proposal import nf_metropolis_sampler
from flax.training import train_state  # Useful dataclass to keep train state
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
                use_global: bool = True,
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
        self.use_global = use_global
        self.logging = logging

        self.nf_model = nf_model
        params = nf_model.init(init_rng_keys_nf, jnp.ones((self.batch_size,self.n_dim)))['params']

        tx = optax.adam(self.learning_rate, self.momentum)
        self.state = train_state.TrainState.create(apply_fn=nf_model.apply,
                                                   params=params, tx=tx)

        self.chains = []
        self.log_prob = []
        self.local_accs = []
        self.global_accs = []
        self.loss_vals = []

    def sample(self,initial_position):
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

        last_step = initial_position
        rng_keys_nf = self.rng_keys_nf
        rng_keys_mcmc = self.rng_keys_mcmc
        state = self.state

        for i in range(self.n_loop):
            rng_keys_nf, rng_keys_mcmc, state, last_step = self.sampling_loop(
                rng_keys_nf, rng_keys_mcmc, state, last_step,)

  
        # return chains, log_prob, nf_samples, self.local_accs, global_accs, loss_vals

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

        log_prob_output = np.copy(log_prob)
        flat_chain = positions.reshape(-1, self.n_dim)
        if self.use_global == True:
            rng_keys_nf, state, loss_values = train_flow(rng_keys_nf, self.nf_model, state, flat_chain,
                                            self.n_epochs, self.batch_size)
            likelihood_vec = jax.vmap(self.likelihood)
            rng_keys_nf, nf_chain, log_prob, log_prob_nf, global_acceptance = nf_metropolis_sampler(
                rng_keys_nf, self.n_global_steps, self.nf_model, state.params , likelihood_vec,
                positions[:,-1]
                )

            positions = jnp.concatenate((positions,nf_chain),axis=1)
            log_prob_output = jnp.concatenate((log_prob_output,log_prob),axis=1)
        
        self.chains.append(positions)
        self.log_prob.append(log_prob_output)
        self.local_accs.append(local_acceptance)
        if self.use_global == True:
            self.global_accs.append(global_acceptance)
            self.loss_vals.append(loss_values)

        last_step = positions[:, -1]

        return rng_keys_nf, rng_keys_mcmc, state, last_step


    def get_sampler_state(self):
        chains = jnp.concatenate(self.chains, axis=1)
        log_prob = jnp.concatenate(self.log_prob, axis=1)
        local_accs = jnp.stack(self.local_accs, axis=1).reshape(chains.shape[0], -1)
        if self.use_global == True:
            global_accs = jnp.stack(self.global_accs, axis=1).reshape(chains.shape[0], -1)
            loss_vals = jnp.stack(self.loss_vals, axis=1).reshape(self.n_loop, self.n_epochs)
            return chains, log_prob, local_accs, global_accs, loss_vals
        else:
            return chains, log_prob, local_accs, jnp.zeros(0), jnp.zeros(0)

    def sample_flow(self):
        nf_samples = sample_nf(self.nf_model, self.state.params, self.rng_keys_nf, self.n_nf_samples)
        return nf_samples


