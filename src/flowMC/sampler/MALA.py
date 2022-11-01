from typing import Callable
import jax
import jax.numpy as jnp
from jax.scipy.stats import multivariate_normal
from tqdm import tqdm


def make_mala_kernel(logpdf: Callable):
    """
    Making MALA kernel for a single step

    Args:
        logpdf (Callable): log-probability density function


    Returns:
        mala_kernel (Callable): MALA kernel for a single step

    """

    def body(carry, this_key):
        this_position, dt = carry
        dt2 = dt*dt
        this_log_prob, this_d_log = jax.value_and_grad(logpdf)(this_position)
        proposal = this_position + jnp.dot(dt2, this_d_log) / 2
        proposal += jnp.dot(dt, jax.random.normal(this_key, shape=this_position.shape))
        return (proposal,dt), (proposal, this_log_prob, this_d_log)

    def mala_kernel(rng_key, position, log_prob, dt = 1e-1):

        """
        Metropolis-adjusted Langevin algorithm kernel.
        This function make a proposal and accept/reject it.

        Args:
            rng_key (n_chains, 2): random key
            logpdf (function) : log-density function
            d_logpdf (function): gradient of log-density function
            position (n_chains, n_dim): current position
            log_prob (n_chains, ): log-probability of the current position
            dt (float): step size of the MALA step

        Returns:
            position (n_chains, n_dim): the new poisiton of the chain
            log_prob (n_chains, ): the log-probability of the new position
            do_accept (n_chains, ): whether to accept the new position

        """
        key1, key2 = jax.random.split(rng_key)

        dt2 = dt*dt

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


def make_mala_update(logpdf):
    """

    Making MALA update function for multiple steps

    Args:
        logpdf (Callable): log-probability density function
        d_logpdf (Callable): gradient of log-probability density function
        dt (float): step size of the MALA step
        M (jnp.array): mass matrix. Currently we only support diagonal mass matrix

    Returns:
        mala_update (Callable): MALA update function for multiple steps

    """
    mala_kernel = make_mala_kernel(logpdf)

    def mala_update(i, state):
        key, positions, log_prob, acceptance, dt = state
        _, key = jax.random.split(key)
        new_position, new_log_prob, do_accept = mala_kernel(
            key, positions[i - 1], log_prob[i - 1], dt
        )
        positions = positions.at[i].set(new_position)
        log_prob = log_prob.at[i].set(new_log_prob)
        acceptance = acceptance.at[i].set(do_accept)
        return (key, positions, log_prob, acceptance, dt)

    # Apperantly jitting after vmap will make compilation much slower.
    # Output the kernel, logpdf, and dlogpdf for warmup jitting.
    # Apperantly passing in a warmed up function will still trigger recompilation.
    # so the warmup need to be done with the output function

    return mala_update, logpdf


def make_mala_sampler(logpdf: Callable, jit: bool=False):
    mala_update, lp = make_mala_update(logpdf)
    # Somehow if I define the function inside the other function,
    # I think it doesn't use the cache and recompile everytime.
    if jit:
        mala_update = jax.jit(mala_update)
        lp = jax.jit(lp)

    mala_update = jax.vmap(mala_update, in_axes=(None, (0, 0, 0, 0, None)), out_axes=(0, 0, 0, 0, None))
    lp = jax.vmap(lp)

    def mala_sampler(rng_key, n_steps, initial_position, sampler_params={'dt':1e-1}):

        """
        Metropolis-adjusted Langevin algorithm sampler.
        This function do n step with the MALA kernel.

        Args:
            rng_key (n_chains, 2): random key for the sampler
            n_steps (int): number of local steps
            logpdf (function): log-density function
            d_logpdf (function): gradient of log-density function
            initial_position (n_chains, n_dim): initial position of the chain
            dt (float): step size of the MALA step

        Returns:
            rng_key (n_chains, 2): random key for the sampler after the sampling
            all_positions (n_chains, n_steps, n_dim): all the positions of the chain
            log_probs (n_chains, ): log probability at the end of the chain
            acceptance: acceptance rate of the chain
        """

        logp = lp(initial_position)
        n_chains = rng_key.shape[0]
        acceptance = jnp.zeros(
            (
                n_chains,
                n_steps,
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
        state = (rng_key, all_positions, all_logp, acceptance, sampler_params['dt'])
        for i in tqdm(
            range(1, n_steps), desc="Sampling Locally", miniters=int(n_steps / 10)
        ):
            state = mala_update(i, state)
        return state

    return mala_sampler
    
from tqdm import tqdm
from functools import partialmethod

def mala_sampler_autotune(mala_sampler, rng_key, n_steps, initial_position, sampler_params, max_iter = 30):
    tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

    counter = 0
    dt = sampler_params['dt']
    rng_keys_mcmc, positions, log_prob, local_acceptance, _ = mala_sampler(rng_key, n_steps, initial_position, {'dt':dt})
    acceptance_rate = jnp.mean(local_acceptance)
    while (acceptance_rate < 0.3) or (acceptance_rate > 0.5):
        if counter > max_iter:
            print("Maximal number of iterations reached. Existing tuning with current parameters.")
            break
        if acceptance_rate < 0.3:
            dt *= 0.8
        elif acceptance_rate > 0.5:
            dt *= 1.25
        counter += 1
        rng_keys_mcmc, positions, log_prob, local_acceptance, _ = mala_sampler(rng_keys_mcmc, n_steps, initial_position, {'dt':dt})
        acceptance_rate = jnp.mean(local_acceptance)
    tqdm.__init__ = partialmethod(tqdm.__init__, disable=False)
    return {"dt": dt}, mala_sampler


################### Scan API ##############################

# def make_mala_kernel(logpdf, d_logpdf, dt):
#     def mala_kernel(carry, data):
#         rng_key, position, log_prob, do_accept = carry
#         rng_key, key1, key2 = jax.random.split(rng_key,3)
#         proposal = position + dt * d_logpdf(position)
#         proposal += dt * jnp.sqrt(2/dt) * jax.random.normal(key1, shape=position.shape)
#         ratio = logpdf(proposal) - logpdf(position)
#         ratio -= ((position - proposal - dt * d_logpdf(proposal)) ** 2 / (4 * dt)).sum()
#         ratio += ((proposal - position - dt * d_logpdf(position)) ** 2 / (4 * dt)).sum()
#         proposal_log_prob = logpdf(proposal)

#         log_uniform = jnp.log(jax.random.uniform(key2))
#         do_accept = log_uniform < ratio

#         position = jax.lax.cond(do_accept, lambda: proposal, lambda: position)
#         log_prob = jax.lax.cond(do_accept, lambda: proposal_log_prob, lambda: log_prob)
#         return (rng_key, position, log_prob, do_accept), (position, log_prob, do_accept)
#     return mala_kernel, logpdf, d_logpdf

# def make_mala_update(logpdf, d_logpdf, dt):
#     mala_kernel, logpdf, d_logpdf = make_mala_kernel(logpdf, d_logpdf, dt)
#     def mala_update(rng_key, position, logp, n_steps=100):
#         carry = (rng_key, position, logp, False)
#         y = jax.lax.scan(mala_kernel, carry, jax.random.split(rng_key,n_steps))
#         return y
#     mala_update = jax.vmap(mala_update, in_axes=(0,0,0,None))
#     mala_kernel_vec = jax.vmap(mala_kernel, in_axes=((0,0,0,0),None))
#     logpdf = jax.vmap(logpdf)
#     d_logpdf = jax.vmap(d_logpdf)
#     return mala_update, mala_kernel_vec, logpdf, d_logpdf
