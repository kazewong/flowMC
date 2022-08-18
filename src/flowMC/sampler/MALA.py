import jax
import jax.numpy as jnp
from jax import grad
from tqdm import tqdm
from flowMC.utils.progressBar import progress_bar_scan

def mala_kernel(rng_key, logpdf, d_logpdf, position, log_prob, dt=0.1):

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
    proposal = position + dt * d_logpdf(position)
    proposal += dt * jnp.sqrt(2/dt) * jax.random.normal(key1, shape=position.shape)
    ratio = logpdf(proposal) - logpdf(position)
    ratio -= ((position - proposal - dt * d_logpdf(proposal)) ** 2 / (4 * dt)).sum()
    ratio += ((proposal - position - dt * d_logpdf(position)) ** 2 / (4 * dt)).sum()
    proposal_log_prob = logpdf(proposal)

    log_uniform = jnp.log(jax.random.uniform(key2))
    do_accept = log_uniform < ratio

    position = jnp.where(do_accept, proposal, position)
    log_prob = jnp.where(do_accept, proposal_log_prob, log_prob)
    return position, log_prob, do_accept

mala_kernel_vec = jax.vmap(mala_kernel, in_axes=(0, None, None, 0, 0, None))


def make_mala_update(logpdf, d_logpdf, dt):
    def mala_update(i,state):
        key, positions, log_prob, acceptance = state
        _, key = jax.random.split(key)
        new_position, new_log_prob, do_accept = mala_kernel(key, logpdf,
                                                                d_logpdf,
                                                                positions[i-1],
                                                                log_prob[i-1],
                                                                dt)
        positions = positions.at[i].set(new_position)
        log_prob = log_prob.at[i].set(new_log_prob)
        acceptance = acceptance.at[i].set(do_accept)
        return (key, positions, log_prob, acceptance)

    mala_update = jax.jit(jax.vmap(mala_update, in_axes=(None,(0,0,0,0))))

    return mala_update

def make_mala_sampler(logpdf, d_logpdf, dt=1e-5):
    mala_update = make_mala_update(logpdf, d_logpdf, dt)
    # Somehow if I define the function inside the other function,
    # I think it doesn't use the cache and recompile everytime.

    def mala_sampler(rng_key, n_steps, logpdf, d_logpdf, initial_position,dt):

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

        logp = jax.vmap(logpdf)(initial_position)
        n_chains = rng_key.shape[0]
        acceptance = jnp.zeros((n_chains,n_steps,))
        all_positions = jnp.zeros((n_chains, n_steps,)+initial_position.shape[-1:]) + initial_position[:,None]
        all_logp = jnp.zeros((n_chains,n_steps,)) + logp[:,None]
        state = (rng_key, all_positions, all_logp, acceptance)
        # Lax for loop takes a long time to compile and end up being slower
        for i in tqdm(range(1, n_steps),desc='Sampling Locally',miniters=int(n_steps/10)):
            state = mala_update(i, state)
        # rng_key, all_positions, all_logp, acceptance = jax.lax.fori_loop(1, n_steps, 
        #                                             mala_update,
        #                                             initial_state)
        
        
        return state

    return mala_sampler