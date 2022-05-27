import jax
import jax.numpy as jnp
from jax import random, jit, vmap


def nf_metropolis_kernel(rng_key, proposal_position, initial_position,
                         proposal_pdf, proposal_nf_pdf, initial_pdf, initial_nf_pdf):

    rng_key, subkeys = random.split(rng_key,2)
    ratio = (proposal_pdf - initial_pdf) - (proposal_nf_pdf - initial_nf_pdf)
    ratio = jnp.exp(ratio)
    u = jax.random.uniform(subkeys, ratio.shape)
    do_accept = u < ratio
    position = jnp.where(do_accept, proposal_position, initial_position)
    log_prob = jnp.where(do_accept, proposal_pdf, initial_pdf)
    log_prob_nf = jnp.where(do_accept, proposal_nf_pdf, initial_nf_pdf)
    return position, log_prob, log_prob_nf, do_accept

nf_metropolis_kernel = vmap(jit(nf_metropolis_kernel))

def nf_metropolis_sampler(rng_key, n_steps, nf_model, nf_param, target_pdf,
                          initial_position):
    """
    Returns:
        rng_key: current state of random key
        all_positions (n_steps, dim): all the positions of the chain
        log_prob (): log probability at the end of the chain
        log_prob_nf (): log probability at the end of the chain
        acceptance (n_steps, ): acceptance table of the chains
    """

    def mh_update_sol2(i, state):
        key, positions, log_prob, log_prob_nf, acceptance = state
        key, *sub_key = jax.random.split(key, positions.shape[1]+1)
        sub_key = jnp.array(sub_key)
        new_position, new_log_prob, new_log_prob_nf, do_accept = nf_metropolis_kernel(
            sub_key, proposal_position[i], positions[i-1], log_pdf_proposal[i],
            log_pdf_nf_proposal[i], log_prob[i-1], log_prob_nf[i-1]
            )
        positions = positions.at[i].set(new_position)
        log_prob = log_prob.at[i].set(new_log_prob)
        log_prob_nf = log_prob_nf.at[i].set(new_log_prob_nf)

        acceptance = acceptance.at[i].set(do_accept)

        return (key, positions, log_prob, log_prob_nf, acceptance)

    rng_key, *subkeys = random.split(rng_key,3)



    proposal_position = nf_model.apply({'params': nf_param}, subkeys[0],
                                       initial_position.shape[0]*n_steps,
                                       nf_param, method=nf_model.sample)[0]

    log_pdf_nf_proposal = nf_model.apply({'params': nf_param},
                                         proposal_position,
                                         method=nf_model.log_prob)

    log_pdf_nf_initial = nf_model.apply({'params': nf_param}, initial_position,
                                        method=nf_model.log_prob)
    
    log_pdf_proposal = target_pdf(proposal_position)
    log_pdf_initial = target_pdf(initial_position)


    proposal_position = proposal_position.reshape(n_steps,
                                                  initial_position.shape[0],
                                                  initial_position.shape[1])
    log_pdf_nf_proposal = log_pdf_nf_proposal.reshape(n_steps,
                                                      initial_position.shape[0])
    log_pdf_proposal = log_pdf_proposal.reshape(n_steps, initial_position.shape[0])
    
    all_positions = jnp.zeros((n_steps,) + initial_position.shape) + \
        initial_position
    all_logp = jnp.zeros((n_steps, initial_position.shape[0])) + log_pdf_initial
    all_logp_nf = jnp.zeros((n_steps, initial_position.shape[0])) + log_pdf_nf_initial
    acceptance = jnp.zeros((n_steps, initial_position.shape[0]))


    initial_state = (subkeys[1], all_positions, all_logp,
                    all_logp_nf, acceptance)
    rng_key, all_positions, all_logp, all_logp_nf, acceptance = jax.lax.fori_loop(1, n_steps, 
                                                 mh_update_sol2, 
                                                 initial_state)
    all_positions = all_positions.swapaxes(0, 1)
    all_logp = all_logp.swapaxes(0, 1)
    all_logp_nf = all_logp_nf.swapaxes(0, 1)
    acceptance = acceptance.swapaxes(0, 1)
    
    return rng_key, all_positions, all_logp, all_logp_nf, acceptance
