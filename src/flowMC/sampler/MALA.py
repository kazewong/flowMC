import jax
import jax.numpy as jnp
from jax import grad

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

def mala_sampler(rng_key, n_steps, logpdf, d_logpdf, initial_position, dt=1e-5):

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

    def mh_update_sol2(i, state):
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


    logp = logpdf(initial_position)
     # local sampler are vmapped outside, so the chain dimension is 1
    acceptance = jnp.zeros((n_steps,))
    all_positions = jnp.zeros((n_steps,)+initial_position.shape) + initial_position
    all_logp = jnp.zeros((n_steps,)) + logp
    initial_state = (rng_key,all_positions, all_logp, acceptance)
    rng_key, all_positions, all_logp, acceptance = jax.lax.fori_loop(1, n_steps, 
                                                   mh_update_sol2, 
                                                   initial_state)
    
    
    return rng_key, all_positions, all_logp, acceptance