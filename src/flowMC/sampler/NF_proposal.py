from typing import Tuple
import jax
import jax.numpy as jnp
from jax import random, jit, vmap
from tqdm import tqdm
from flowMC.nfmodel.base import NFModel
from jaxtyping import Array, PRNGKeyArray, PyTree
from typing import Callable
n_sample_max = 100000

def nf_metropolis_kernel(
    rng_key: jax.random.PRNGKey,
    proposal_position: jnp.ndarray,
    initial_position: jnp.ndarray,
    log_proposal_pdf: jnp.ndarray,
    log_proposal_nf_pdf: jnp.ndarray,
    log_initial_pdf: jnp.ndarray,
    log_initial_nf_pdf: jnp.ndarray,
):

    """
    A one-step global sampling kernel for a given normalizing flow model.

    Args:
        rng_key: Jax PRNGKey.
        proposal_position: Proposed positions, shape (Ndim).
        initial_position: Initial positions, shape (Ndim).
        log_proposal_pdf: Log-pdf value evaluate using the target function at the proposal position, shape (Ndim).
        log_proposal_nf_pdf: Log-pdf value evaluate using the normalizing flow model at the proposal position, shape (Ndim).
        log_initial_pdf: Log-pdf value evaluate using the target function at the initial position, shape (Ndim).
        log_initial_nf_pdf: Log-pdf value evaluate using the normalizing flow model at the initial position, shape (Ndim).

    Returns:
        position: New positions, shape (Ndim).
        log_prob: Pdf value evaluate using the target function at the new position, shape (Ndim).
        log_prob_nf: Pdf value evaluate using the normalizing flow model at the new position, shape (Ndim).
        do_accept: Acceptance boolean, shape (Ndim).
    """

    rng_key, subkeys = random.split(rng_key, 2)
    ratio = (log_proposal_pdf - log_initial_pdf) - (log_proposal_nf_pdf - log_initial_nf_pdf)
    u = jnp.log(jax.random.uniform(subkeys, ratio.shape))
    do_accept = u < ratio
    position = jnp.where(do_accept, proposal_position, initial_position)
    log_prob = jnp.where(do_accept, log_proposal_pdf, log_initial_pdf)
    log_prob_nf = jnp.where(do_accept, log_proposal_nf_pdf, log_initial_nf_pdf)
    return position, log_prob, log_prob_nf, do_accept

nf_metropolis_kernel = vmap(nf_metropolis_kernel)


@jax.jit
def nf_metropolis_update(i: int, state: Tuple):
    """
    A multistep global sampling kernel for a given normalizing flow model.

    Args:
        i: Number of current iteration.
        state: A tuple containing the current state of the sampler.
    """
    (
        key,
        positions,
        proposal,
        log_prob,
        log_prob_nf,
        log_prob_proposal,
        log_prob_nf_proposal,
        acceptance,
    ) = state
    key, *sub_key = jax.random.split(key, positions.shape[1] + 1)
    sub_key = jnp.array(sub_key)
    new_position, new_log_prob, new_log_prob_nf, do_accept = nf_metropolis_kernel(
        sub_key,
        proposal[i],
        positions[i - 1],
        log_prob_proposal[i],
        log_prob_nf_proposal[i],
        log_prob[i - 1],
        log_prob_nf[i - 1],
    )
    positions = positions.at[i].set(new_position)
    log_prob = log_prob.at[i].set(new_log_prob)
    log_prob_nf = log_prob_nf.at[i].set(new_log_prob_nf)

    acceptance = acceptance.at[i].set(do_accept)

    return (
        key,
        positions,
        proposal,
        log_prob,
        log_prob_nf,
        log_prob_proposal,
        log_prob_nf_proposal,
        acceptance,
    )


def nf_metropolis_sampler(nf_model: NFModel,
                        rng_key: PRNGKeyArray,
                        n_steps: int,
                        target_pdf: Callable,
                        initial_position: Array,
                        data: PyTree,
):
    r""" Normalizing flow Metropolis sampler.

    Args:
        nf_model: Normalizing flow model.
        rng_key: Jax PRNGKey.
        n_steps: Number of steps to run the sampler.
        target_pdf: Target pdf function.
        initial_position: Initial position of the sampler.
        data: Data to be passed to the target pdf function.
    """

    rng_key, *subkeys = random.split(rng_key, 3)

    total_sample = initial_position.shape[0] * n_steps

    log_pdf_nf_initial = nf_model.log_prob(initial_position)
    log_pdf_initial = target_pdf(initial_position, data)

    if total_sample > n_sample_max:
        proposal_position = jnp.zeros((total_sample, initial_position.shape[-1]))
        log_pdf_nf_proposal = jnp.zeros((total_sample,))
        log_pdf_proposal = jnp.zeros((total_sample,))
        local_key, subkey = random.split(subkeys[0], 2)
        for i in tqdm(
            range(total_sample // n_sample_max),
            desc="Sampling Globally",
            miniters=(total_sample // n_sample_max) // 10,
        ):
            local_samples = nf_model.sample(subkey, n_sample_max)
            proposal_position = proposal_position.at[
                i * n_sample_max : (i + 1) * n_sample_max
            ].set(local_samples)
            log_pdf_nf_proposal = log_pdf_nf_proposal.at[
                i * n_sample_max : (i + 1) * n_sample_max
            ].set(nf_model.log_prob(local_samples))
            log_pdf_proposal = log_pdf_proposal.at[
                i * n_sample_max : (i + 1) * n_sample_max
            ].set(target_pdf(local_samples, data))
            local_key, subkey = random.split(local_key, 2)

    else:
        proposal_position = nf_model.sample(subkeys[0], total_sample)
        log_pdf_nf_proposal = nf_model.log_prob(proposal_position)
        log_pdf_proposal = target_pdf(proposal_position, data)

    proposal_position = proposal_position.reshape(
        n_steps, initial_position.shape[0], initial_position.shape[1]
    )
    log_pdf_nf_proposal = log_pdf_nf_proposal.reshape(
        n_steps, initial_position.shape[0]
    )
    log_pdf_proposal = log_pdf_proposal.reshape(n_steps, initial_position.shape[0])

    all_positions = (
        jnp.zeros((n_steps,) + initial_position.shape) + initial_position
    )
    all_logp = jnp.zeros((n_steps, initial_position.shape[0])) + log_pdf_initial
    all_logp_nf = (
        jnp.zeros((n_steps, initial_position.shape[0])) + log_pdf_nf_initial
    )
    acceptance = jnp.zeros((n_steps, initial_position.shape[0]))

    initial_state = (
        subkeys[1],
        all_positions,
        proposal_position,
        all_logp,
        all_logp_nf,
        log_pdf_proposal,
        log_pdf_nf_proposal,
        acceptance,
    )
    (
        rng_key,
        all_positions,
        proposal_position,
        all_logp,
        all_logp_nf,
        log_pdf_proposal,
        log_pdf_nf_proposal,
        acceptance,
    ) = jax.lax.fori_loop(1, n_steps, nf_metropolis_update, initial_state)
    all_positions = all_positions.swapaxes(0, 1)
    all_logp = all_logp.swapaxes(0, 1)
    all_logp_nf = all_logp_nf.swapaxes(0, 1)
    acceptance = acceptance.swapaxes(0, 1)

    return rng_key, all_positions, all_logp, all_logp_nf, acceptance