from typing import Callable
import jax
import jax.numpy as jnp
from jax.scipy.stats import multivariate_normal
from tqdm import tqdm
from flowMC.sampler.LocalSampler_Base import LocalSamplerBase

class HMC(LocalSamplerBase):
    
    def __init__(self, logpdf: Callable, jit: bool, params: dict) -> Callable:
        super().__init__(logpdf, jit, params)

        self.potential = lambda x: -self.logpdf(x)
        self.grad_potential = jax.grad(self.potential)
        
        self.params = params
        if "condition_matrix" in params:
            self.inverse_metric = params["condition_matrix"]
        else:
            self.inverse_metric = 1

        if "step_size" in params:
            self.step_size = params["step_size"]

        if "n_leapfrog" in params:
            self.n_leapfrog = params["n_leapfrog"]
        else:
            raise NotImplementedError

        self.kinetic = lambda p, params: 0.5*(p**2 * params['inverse_metric']).sum()
        self.grad_kinetic = jax.grad(self.kinetic)

    def get_initial_hamiltonian(self, rng_key: jax.random.PRNGKey, position: jnp.array, params: dict):
        momentum = jax.random.normal(rng_key, shape=position.shape) * params['inverse_metric']
        return self.potential(position) + self.kinetic(momentum, params)

    def make_kernel(self, return_aux = False) -> Callable:
        """

        Making HMC kernel for a single step

        """

        def leapfrog_kernal(carry, data):
            position, momentum, params = carry
            position = position + params['step_size'] * self.grad_kinetic(momentum, params)
            momentum = momentum - params['step_size'] * self.grad_potential(position)
            return (position, momentum, params), data


        def leapfrog_step(position, momentum, params: dict):
            momentum = momentum - 0.5 * params['step_size'] * self.grad_potential(position)
            (position, momentum, params), _ = jax.lax.scan(leapfrog_kernal, (position, momentum, params), jnp.arange(self.n_leapfrog-1))
            position = position + params['step_size'] * self.grad_kinetic(momentum, params)
            momentum = momentum - 0.5*params['step_size'] * self.grad_potential(position)
            return position, momentum

        def hmc_kernel(rng_key, position, H, params: dict):
            """

            Note that since the potential function is the negative log likelihood, hamiltonian is going down, but the likelihood value should go up.

            Args:
            rng_key (n_chains, 2): random key
            position (n_chains, n_dim): current position
            H (n_chains, ): Hamiltonian of the current position
            """
            key1, key2 = jax.random.split(rng_key)

            momentum = jax.random.normal(key1, shape=position.shape) * params['inverse_metric']
            proposed_position, proposed_momentum = leapfrog_step(position, momentum, params)
            proposed_ham = self.potential(proposed_position) + self.kinetic(proposed_momentum, params)
            log_acc = H - proposed_ham
            log_uniform = jnp.log(jax.random.uniform(key2))

            do_accept = log_uniform < log_acc

            position = jnp.where(do_accept, proposed_position, position)
            log_prob = jnp.where(do_accept, proposed_ham, H)

            return position, log_prob, do_accept
        
        if return_aux == False:
            return hmc_kernel
        else:
            return hmc_kernel, leapfrog_kernal, leapfrog_step
            

    def make_update(self) -> Callable:
        """
        
        Making HMC update function for multiple steps

        """

        hmc_kernel = self.make_kernel()

        def hmc_update(i, state):
            key, positions, log_Ham, acceptance, params = state
            _, key = jax.random.split(key)
            new_position, new_log_Ham, do_accept = hmc_kernel(
                key, positions[i - 1], log_Ham[i - 1], params
            )
            positions = positions.at[i].set(new_position)
            log_Ham = log_Ham.at[i].set(new_log_Ham)
            acceptance = acceptance.at[i].set(do_accept)
            return (key, positions, log_Ham, acceptance, params)

        return hmc_update

    def make_sampler(self) -> Callable:
        hmc_update = self.make_update()

        if self.jit:
            hmc_update = jax.jit(hmc_update)

        hmc_update = jax.vmap(hmc_update, in_axes = (None, (0, 0, 0, 0, None)), out_axes=(0, 0, 0, 0, None))
        lh = jax.vmap(self.get_initial_hamiltonian, in_axes=(0, 0, None))

        def hmc_sampler(rng_key, n_steps, initial_position):
            
            keys = jax.vmap(jax.random.split)(rng_key)
            rng_key = keys[:, 0]
            rng_init = keys[:, 1]
            logp = lh(rng_init,initial_position, self.params)
            n_chains = rng_key.shape[0]
            acceptance = jnp.zeros(
                (
                    n_chains,
                    n_steps
                )
            )
            all_positions = (
                jnp.zeros(
                    (
                        n_chains,
                        n_steps,
                    )
                    + initial_position.shape[-1:]
                )
                + initial_position[:, None]
            )
            all_logp = (
                jnp.zeros(
                    (
                        n_chains,
                        n_steps,
                    )
                )
                + logp[:, None]
            )
            state = (rng_key, all_positions, all_logp, acceptance, self.params)
            for i in tqdm(
                range(1, n_steps), desc="Sampling Locally", miniters=int(n_steps / 10)
            ):
                state = hmc_update(i, state)
            return state[:-1]

        return hmc_sampler

    def make_leapforg_kernel(self):
        def leapfrog_kernal(carry, data):
            position, momentum, params = carry
            position = position + self.step_size * self.grad_kinetic(momentum, params)
            momentum = momentum - self.step_size * self.grad_potential(position)
            return position, momentum
        return leapfrog_kernal





from tqdm import tqdm
from functools import partialmethod
