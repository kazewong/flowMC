from typing import Tuple
import jax
import jax.numpy as jnp
from jax import random, jit, vmap
from tqdm import tqdm

n_sample_max = 100000


def nf_metropolis_kernel(
    rng_key: jax.random.PRNGKey,
    proposal_position: jnp.ndarray,
    initial_position: jnp.ndarray,
    proposal_pdf: jnp.ndarray,
    proposal_nf_pdf: jnp.ndarray,
    initial_pdf: jnp.ndarray,
    initial_nf_pdf: jnp.ndarray,
):

    """
    A one-step global sampling kernel for a given normalizing flow model.

    Args:
        rng_key: Jax PRNGKey.
        proposal_position: Proposed positions, shape (Ndim).
        initial_position: Initial positions, shape (Ndim).
        proposal_pdf: Pdf value evaluate using the target function at the proposal position, shape (Ndim).
        proposal_nf_pdf: Pdf value evaluate using the normalizing flow model at the proposal position, shape (Ndim).
        initial_pdf: Pdf value evaluate using the target function at the initial position, shape (Ndim).
        initial_nf_pdf: Pdf value evaluate using the normalizing flow model at the initial position, shape (Ndim).

    Returns:
        position: New positions, shape (Ndim).
        log_prob: Pdf value evaluate using the target function at the new position, shape (Ndim).
        log_prob_nf: Pdf value evaluate using the normalizing flow model at the new position, shape (Ndim).
        do_accept: Acceptance boolean, shape (Ndim).
    """

    rng_key, subkeys = random.split(rng_key, 2)
    ratio = (proposal_pdf - initial_pdf) - (proposal_nf_pdf - initial_nf_pdf)
    u = jnp.log(jax.random.uniform(subkeys, ratio.shape))
    do_accept = u < ratio
    position = jnp.where(do_accept, proposal_position, initial_position)
    log_prob = jnp.where(do_accept, proposal_pdf, initial_pdf)
    log_prob_nf = jnp.where(do_accept, proposal_nf_pdf, initial_nf_pdf)
    return position, log_prob, log_prob_nf, do_accept


nf_metropolis_kernel = vmap(nf_metropolis_kernel)


@jax.jit
def nf_metropolis_update(i: int, state: Tuple):

    """
    A multistep global sampling kernel for a given normalizing flow model.

    Args:
        i: Number of iteration.
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


def make_nf_metropolis_sampler(nf_model):

    """
    A function make the normalizing flow sampler for a given normalizing flow model.

    Args:
        nf_model: A normalizing flow model.
        
    Returns:
        nf_metropolis_sampler: A normalizing flow sampler that has the following signature:

            Args:
                rng_key: Jax PRNGKey.
                n_steps: Number of steps.
                nf_param: Normalizing flow parameters.
                nf_variables: Normalizing flow variables.
                target_pdf: Target pdf.
                initial_position: Initial position.

            Returns:
                rng_key: Current state of random key
                all_positions: All positions of the chain
                all_logp: Log probability of the target function at all positions
                all_logp_nf: Log probability of the normalizing flow model at all positions
                acceptance: Acceptance boolean

    """

    @jax.jit
    def eval_nf_logprob(position, nf_params, nf_variables):
        return nf_model.apply(
            {"params": nf_params, "variables": nf_variables},
            position,
            method=nf_model.log_prob,
        )

    def sample_nf(rng_key, n_samples, nf_params, nf_variables):
        return nf_model.apply(
            {"params": nf_params, "variables": nf_variables},
            rng_key,
            n_samples,
            method=nf_model.sample,
        )

    sample_nf = jax.jit(sample_nf, static_argnums=(1))

    def nf_metropolis_sampler(
        rng_key, n_steps, nf_param, nf_variables, target_pdf, initial_position
    ):
        rng_key, *subkeys = random.split(rng_key, 3)

        total_sample = initial_position.shape[0] * n_steps

        log_pdf_nf_initial = eval_nf_logprob(
            initial_position, nf_param, nf_variables
        )
        log_pdf_initial = target_pdf(initial_position)

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
                local_samples = sample_nf(subkey, n_sample_max, nf_param, nf_variables)
                proposal_position = proposal_position.at[
                    i * n_sample_max : (i + 1) * n_sample_max
                ].set(local_samples)
                log_pdf_nf_proposal = log_pdf_nf_proposal.at[
                    i * n_sample_max : (i + 1) * n_sample_max
                ].set(eval_nf_logprob(local_samples, nf_param, nf_variables))
                log_pdf_proposal = log_pdf_proposal.at[
                    i * n_sample_max : (i + 1) * n_sample_max
                ].set(target_pdf(local_samples))
                local_key, subkey = random.split(local_key, 2)

        else:
            proposal_position = sample_nf(
                subkeys[0], total_sample, nf_param, nf_variables
            )
            log_pdf_nf_proposal = eval_nf_logprob(
                proposal_position, nf_param, nf_variables
            )
            log_pdf_proposal = target_pdf(proposal_position)

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

    return nf_metropolis_sampler
