import jax
import jax.numpy as jnp
from jax.scipy.stats import multivariate_normal
from tqdm import tqdm


def make_mMALA_kernel(logpdf, d_logpdf, d2_logpdf, dt):
    def mMALA_kernel(rng_key, position, log_prob):

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

        n_dim = position.shape[-1]
        d_log_current = d_logpdf(position)
        d2_current = d2_logpdf(position)
        d2_current = jax.lax.cond(
            jnp.any(jnp.isnan(d2_current)), lambda: jnp.eye(n_dim), lambda: d2_current
        )
        d2_current = jnp.linalg.inv(d2_current)

        dt_current_sqrt = dt * jnp.sqrt(d2_current)

        proposal = position + jnp.dot((dt * d2_current) ** 2, d_log_current) / 2
        proposal += jnp.dot(
            dt_current_sqrt, jax.random.normal(key1, shape=position.shape)
        )
        proposal_log_prob = logpdf(proposal)
        d2_proposal = d2_logpdf(proposal)
        d2_proposal = jax.lax.cond(
            jnp.any(jnp.isnan(d2_proposal)), lambda: jnp.eye(n_dim), lambda: d2_proposal
        )
        d2_proposal = jnp.linalg.inv(d2_proposal)

        ratio = proposal_log_prob - logpdf(position)
        ratio -= multivariate_normal.logpdf(
            position,
            proposal + jnp.dot(d2_proposal * dt * dt, d_logpdf(proposal)) / 2,
            d2_proposal * dt * dt,
        )
        ratio += multivariate_normal.logpdf(
            proposal,
            position + jnp.dot(d2_current * dt * dt, d_log_current) / 2,
            d2_current * dt * dt,
        )

        log_uniform = jnp.log(jax.random.uniform(key2))
        do_accept = log_uniform < ratio

        position = jnp.where(do_accept, proposal, position)
        log_prob = jnp.where(do_accept, proposal_log_prob, log_prob)
        return position, log_prob, do_accept

    return mMALA_kernel


def make_mMALA_update(logpdf, d_logpdf, d2_logpdf, dt):
    mMALA_kernel = make_mMALA_kernel(logpdf, d_logpdf, d2_logpdf, dt)

    def mMALA_update(i, state):
        key, positions, log_prob, acceptance = state
        _, key = jax.random.split(key)
        new_position, new_log_prob, do_accept = mMALA_kernel(
            key, positions[i - 1], log_prob[i - 1]
        )
        positions = positions.at[i].set(new_position)
        log_prob = log_prob.at[i].set(new_log_prob)
        acceptance = acceptance.at[i].set(do_accept)
        return (key, positions, log_prob, acceptance)

    mMALA_update = jax.vmap(mMALA_update, in_axes=(None, (0, 0, 0, 0)))
    mMALA_kernel_vec = jax.vmap(mMALA_kernel, in_axes=(0, 0, 0))
    logpdf = jax.vmap(logpdf)
    d_logpdf = jax.vmap(d_logpdf)

    # Apperantly jitting after vmap will make compilation much slower.
    # Output the kernel, logpdf, and dlogpdf for warmup jitting.
    # Apperantly passing in a warmed up function will still trigger recompilation.
    # so the warmup need to be done with the output function

    return mMALA_update, mMALA_kernel_vec, logpdf, d_logpdf


def make_MALA_sampler(logpdf, dt=1e-5, jit=False):

    d_logpdf = jax.grad(logpdf)
    d2_logpdf = jax.hessian(logpdf)
    mMALA_update, mk, lp, dlp = make_mMALA_update(logpdf, d_logpdf, d2_logpdf, dt)
    # Somehow if I define the function inside the other function,
    # I think it doesn't use the cache and recompile everytime.
    if jit:
        mMALA_update = jax.jit(mMALA_update)
        mk = jax.jit(mk)
        lp = jax.jit(lp)
        dlp = jax.jit(dlp)
        d2lp = jax.jit(d2lp)

    def mMALA_sampler(rng_key, n_steps, initial_position):

        """
        Metropolis-adjusted Langevin algorithm sampler.
        This function do n step with the mMALA kernel.

        Args:
            rng_key (n_chains, 2): random key for the sampler
            n_steps (int): number of local steps
            logpdf (function): log-density function
            d_logpdf (function): gradient of log-density function
            initial_position (n_chains, n_dim): initial position of the chain
            dt (float): step size of the mMALA step

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
        state = (rng_key, all_positions, all_logp, acceptance)
        for i in tqdm(
            range(1, n_steps), desc="Sampling Locally", miniters=int(n_steps / 10)
        ):
            state = mMALA_update(i, state)
        return state

    return mMALA_sampler, mMALA_update, mk, lp, dlp
