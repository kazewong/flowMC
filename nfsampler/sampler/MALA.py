import jax
import jax.numpy as jnp
from jax import grad
from functools import partial

@partial(jax.jit, static_argnums=(1, 2))
def mala_kernel(rng_key, logpdf, d_logpdf, position, log_prob, kernal_size=0.1):

    key1, key2 = jax.random.split(rng_key)
    proposal = position + kernal_size * d_logpdf(position)
    proposal += kernal_size * jnp.sqrt(2/kernal_size) * jax.random.normal(key1, shape=position.shape)
    ratio = logpdf(proposal) - logpdf(position)
    ratio -= ((position - proposal - kernal_size * d_logpdf(proposal)) ** 2 / (4 * kernal_size)).sum()
    ratio += ((proposal - position - kernal_size * d_logpdf(position)) ** 2 / (4 * kernal_size)).sum()
    proposal_log_prob = logpdf(proposal)

    log_uniform = jnp.log(jax.random.uniform(key2))
    do_accept = log_uniform < ratio

    position = jnp.where(do_accept, proposal, position)
    log_prob = jnp.where(do_accept, proposal_log_prob, log_prob)
    return position, log_prob


@partial(jax.jit, static_argnums=(1, 2, 3))
def mala_sampler(rng_key, n_samples, logpdf, d_logpdf, initial_position, kernal_size=0.1):

    def mh_update_sol2(i, state):
        key, positions, log_prob = state
        _, key = jax.random.split(key)
        new_position, new_log_prob = mala_kernel(key, logpdf, d_logpdf, positions[i-1], log_prob, kernal_size)
        positions=positions.at[i].set(new_position)
        return (key, positions, new_log_prob)


    logp = logpdf(initial_position)
    all_positions = jnp.zeros((n_samples,)+initial_position.shape) + initial_position
    initial_state = (rng_key,all_positions, logp)
    rng_key, all_positions, log_prob = jax.lax.fori_loop(1, n_samples, 
                                                 mh_update_sol2, 
                                                 initial_state)
    
    
    return rng_key, all_positions, log_prob

