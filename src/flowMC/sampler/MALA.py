from typing import Callable
import jax
import jax.numpy as jnp
from jax.scipy.stats import multivariate_normal
from tqdm import tqdm
from flowMC.sampler.LocalSampler_Base import LocalSamplerBase


class MALA(LocalSamplerBase):
    """
    Metropolis-adjusted Langevin algorithm sampler class builiding the mala_sampler method

    Args:
        logpdf: target logpdf function
        jit: whether to jit the sampler
        params: dictionary of parameters for the sampler
    """

    def __init__(self, logpdf: Callable, jit: bool, params: dict, verbose: bool = False) -> Callable:
        super().__init__(logpdf, jit, params)
        self.params = params
        self.logpdf = logpdf
        self.verbose = verbose

    def make_kernel(self, return_aux = False) -> Callable:
        """
        Make a MALA kernel for a given logpdf.

        Args:
            logpdf : (Callable) The logpdf of the target distribution.

        Returns:
            mala_kernel (Callable) A MALA kernel.
        """
        def body(carry, this_key):
            this_position, dt = carry
            dt2 = dt*dt
            this_log_prob, this_d_log = jax.value_and_grad(self.logpdf)(this_position)
            proposal = this_position + jnp.dot(dt2, this_d_log) / 2
            proposal += jnp.dot(dt, jax.random.normal(this_key, shape=this_position.shape))
            return (proposal,dt), (proposal, this_log_prob, this_d_log)

        def mala_kernel(rng_key, position, log_prob, params = {"step_size": 0.1}):
            """
            Metropolis-adjusted Langevin algorithm kernel.
            This function make a proposal and accept/reject it.

            Args:
                rng_key (n_chains, 2): random key
                position (n_chains, n_dim): current position
                log_prob (n_chains, ): log-probability of the current position

            Returns:
                position (n_chains, n_dim): the new poisiton of the chain
                log_prob (n_chains, ): the log-probability of the new position
                do_accept (n_chains, ): whether to accept the new position

            """
            key1, key2 = jax.random.split(rng_key)

            dt = params['step_size']
            dt2 = dt * dt

            _, (proposal, logprob, d_logprob) = jax.lax.scan(
                body, (position, dt), jnp.array([key1, key1])
            )

            ratio = logprob[1] - logprob[0]
            ratio -= multivariate_normal.logpdf(
                proposal[0], position + jnp.dot(dt2, d_logprob[0]) / 2, dt2
            )
            ratio += multivariate_normal.logpdf(
                position, proposal[0] + jnp.dot(dt2, d_logprob[1]) / 2, dt2
            )

            log_uniform = jnp.log(jax.random.uniform(key2))
            do_accept = log_uniform < ratio

            position = jnp.where(do_accept, proposal[0], position)
            log_prob = jnp.where(do_accept, logprob[1], logprob[0])
            return position, log_prob, do_accept

        return mala_kernel

    def make_update(self) -> Callable:
        """
        Make a MALA update function for multiple steps
        """
        mala_kernel = self.make_kernel()

        def mala_update(i, state):
            key, positions, log_p, acceptance, params = state
            _, key = jax.random.split(key)
            new_position, new_log_p, do_accept = mala_kernel(key, positions[i-1], log_p[i-1], params)
            positions = positions.at[i].set(new_position)
            log_p = log_p.at[i].set(new_log_p)
            acceptance = acceptance.at[i].set(do_accept)
            return (key, positions, log_p, acceptance, params)
        
        return mala_update

    def make_sampler(self) -> Callable:
        """
        Make a MALA sampler for multiple chains given initial positions
        """
        mala_update = self.make_update()
        lp = self.logpdf

        if self.jit:
            mala_update = jax.jit(mala_update)
            lp = jax.jit(self.logpdf)

        mala_update = jax.vmap(mala_update, in_axes = (None, (0, 0, 0, 0, None)), out_axes=(0, 0, 0, 0, None))
        lp = jax.vmap(lp)

        def mala_sampler(rng_key, n_steps, initial_position):
            logp = lp(initial_position)
            n_chains = rng_key.shape[0]
            acceptance = jnp.zeros((n_chains, n_steps))
            all_positions = (jnp.zeros((n_chains, n_steps) + initial_position.shape[-1:])) + initial_position[:, None]
            all_logp = (jnp.zeros((n_chains, n_steps)) + logp[:, None])
            state = (rng_key, all_positions, all_logp, acceptance, self.params)
            if self.verbose:
                iterator_loop = tqdm(range(1, n_steps), desc="Sampling Locally", miniters=int(n_steps / 10))
            else:
                iterator_loop = range(1, n_steps)
            for i in iterator_loop:
                state = mala_update(i, state)
            return state[:-1]

        return mala_sampler 

from tqdm import tqdm
from functools import partialmethod

def mala_sampler_autotune(mala_kernel_vmap, rng_key, initial_position, log_prob, params, max_iter = 30):
    """
    Tune the step size of the MALA kernel using the acceptance rate.

    Args:
        mala_kernel_vmap (Callable): A MALA kernel
        rng_key: Jax PRNGKey
        initial_position (n_chains, n_dim): initial position of the chains
        log_prob (n_chains, ): log-probability of the initial position
        params (dict): parameters of the MALA kernel
        max_iter (int): maximal number of iterations to tune the step size
    """

    tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

    counter = 0
    position, log_prob, do_accept = mala_kernel_vmap(rng_key, initial_position, log_prob, params)
    acceptance_rate = jnp.mean(do_accept)
    while (acceptance_rate <= 0.3) or (acceptance_rate >= 0.5):
        if counter > max_iter:
            print("Maximal number of iterations reached. Existing tuning with current parameters.")
            break
        if acceptance_rate <= 0.3:
            params['step_size'] *= 0.8
        elif acceptance_rate >= 0.5:
            params['step_size'] *= 1.25
        counter += 1
        position, log_prob, do_accept = mala_kernel_vmap(rng_key, initial_position, log_prob, params)
        acceptance_rate = jnp.mean(do_accept)
    tqdm.__init__ = partialmethod(tqdm.__init__, disable=False)
    return params