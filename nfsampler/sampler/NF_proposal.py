import jax
import jax.numpy as jnp
from functools import partial
from jax import random, jit, vmap


def nf_metropolis_kernel(rng_key, proposal_position, initial_position, proposal_pdf, proposal_nf_pdf, initial_pdf, initial_nf_pdf):

    rng_key, subkeys = random.split(rng_key,2)
    ratio = (proposal_pdf - initial_pdf) - (proposal_nf_pdf - initial_nf_pdf)
    ratio = jnp.exp(ratio)
    u = jax.random.uniform(subkeys, ratio.shape)
    do_accept = u < ratio
    position = jnp.where(do_accept, proposal_position, initial_position)
    log_prob = jnp.where(do_accept, proposal_pdf, initial_pdf)
    log_prob_nf = jnp.where(do_accept, proposal_nf_pdf, initial_nf_pdf)
    return position, log_prob, log_prob_nf

nf_metropolis_kernel = vmap(jit(nf_metropolis_kernel))

def nf_metropolis_sampler(rng_key, n_samples, nf_model, nf_param, target_pdf, initial_position):

    def mh_update_sol2(i, state):
        key, positions, log_prob, log_prob_nf = state
        key, *sub_key = jax.random.split(key, positions.shape[1]+1)
        sub_key = jnp.array(sub_key)
        new_position, new_log_prob, new_log_prob_nf = nf_metropolis_kernel(sub_key, proposal_position[i], positions[i-1], log_pdf_proposal[i], log_pdf_nf_proposal[i], log_prob, log_prob_nf)
        positions=positions.at[i].set(new_position)
        return (key, positions, new_log_prob, new_log_prob_nf)

    rng_key, *subkeys = random.split(rng_key,3)
    all_positions = jnp.zeros((n_samples,)+initial_position.shape) + initial_position
    proposal_position = nf_model.apply({'params': nf_param}, subkeys[0], initial_position.shape[0]*n_samples, nf_param, method=nf_model.sample)[0]


    log_pdf_nf_proposal = nf_model.apply({'params': nf_param}, proposal_position, method=nf_model.log_prob)
    log_pdf_nf_initial = nf_model.apply({'params': nf_param}, initial_position, method=nf_model.log_prob)
    log_pdf_proposal = target_pdf(proposal_position)
    log_pdf_initial = target_pdf(initial_position)

    proposal_position = proposal_position.reshape(n_samples, initial_position.shape[0], initial_position.shape[1])
    log_pdf_nf_proposal = log_pdf_nf_proposal.reshape(n_samples, initial_position.shape[0])
    log_pdf_proposal = log_pdf_proposal.reshape(n_samples, initial_position.shape[0])
    initial_state = (subkeys[1], all_positions, log_pdf_initial, log_pdf_nf_initial)
    rng_key, all_positions, log_prob, log_prob_nf = jax.lax.fori_loop(1, n_samples, 
                                                 mh_update_sol2, 
                                                 initial_state)
    all_positions = all_positions.swapaxes(0,1)
    return rng_key, all_positions, log_prob, log_prob_nf
