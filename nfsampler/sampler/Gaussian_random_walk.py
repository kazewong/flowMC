import jax
import jax.numpy as jnp
from functools import partial

@partial(jax.jit, static_argnums=(1,))
def rw_metropolis_kernel(rng_key, logpdf, position, log_prob):
    """Moves the chains by one step using the Random Walk Metropolis algorithm.
    Attributes
    ----------
    rng_key: jax.random.PRNGKey
      Key for the pseudo random number generator.
    logpdf: function
      Returns the log-probability of the model given a position.
    position: jnp.ndarray, shape (n_dims,)
      The starting position.
    log_prob: float
      The log probability at the starting position.
    Returns
    -------
    Tuple
        The next positions of the chains along with their log probability.
    """
    key1, key2 = jax.random.split(rng_key)
    move_proposal = jax.random.normal(key1, shape=position.shape) * 0.1
    proposal = position + move_proposal
    proposal_log_prob = logpdf(proposal)

    log_uniform = jnp.log(jax.random.uniform(key2))
    do_accept = log_uniform < proposal_log_prob - log_prob

    position = jnp.where(do_accept, proposal, position)
    log_prob = jnp.where(do_accept, proposal_log_prob, log_prob)
    return position, log_prob


@partial(jax.jit, static_argnums=(1, 2))
def rw_metropolis_sampler(rng_key, n_samples, logpdf, initial_position):

    def mh_update_sol2(i, state):
        key, positions, log_prob = state
        _, key = jax.random.split(key)
        new_position, new_log_prob = rw_metropolis_kernel(key, logpdf, positions[i-1], log_prob)
        positions=positions.at[i].set(new_position)
        return (key, positions, new_log_prob)


    logp = logpdf(initial_position)
    all_positions = jnp.zeros((n_samples,)+initial_position.shape) + initial_position
    initial_state = (rng_key,all_positions, logp)
    rng_key, all_positions, log_prob = jax.lax.fori_loop(1, n_samples, 
                                                 mh_update_sol2, 
                                                 initial_state)
    
    
    return rng_key, all_positions, log_prob

