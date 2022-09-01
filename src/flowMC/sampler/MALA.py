import jax
import jax.numpy as jnp
from jax.scipy.stats import multivariate_normal
from tqdm import tqdm

def make_mala_kernel(logpdf, d_logpdf, dt, M=None):

    if M != None:
        dt  = dt*jnp.sqrt(M)
    def mala_kernel(rng_key, position, log_prob):

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

        d_log_current = d_logpdf(position)
        proposal = position + jnp.dot(dt*dt,d_log_current)/2
        proposal +=  jnp.dot(dt,jax.random.normal(key1, shape=position.shape))
        proposal_log_prob = logpdf(proposal)

        ratio = proposal_log_prob - logpdf(position)
        ratio -= multivariate_normal.logpdf(position, proposal+jnp.dot(dt*dt,d_logpdf(proposal))/2,dt)
        ratio += multivariate_normal.logpdf(proposal, position+jnp.dot(dt*dt,d_log_current)/2,dt)
        
        log_uniform = jnp.log(jax.random.uniform(key2))
        do_accept = log_uniform < ratio

        position = jnp.where(do_accept, proposal, position)
        log_prob = jnp.where(do_accept, proposal_log_prob, log_prob)
        return position, log_prob, do_accept
    return mala_kernel


def make_mala_update(logpdf, d_logpdf, dt, M=None):
    mala_kernel = make_mala_kernel(logpdf, d_logpdf, dt, M)
    def mala_update(i,state):
        key, positions, log_prob, acceptance = state
        _, key = jax.random.split(key)
        new_position, new_log_prob, do_accept = mala_kernel(key, positions[i-1],
                                                                log_prob[i-1])
        positions = positions.at[i].set(new_position)
        log_prob = log_prob.at[i].set(new_log_prob)
        acceptance = acceptance.at[i].set(do_accept)
        return (key, positions, log_prob, acceptance)

    mala_update = jax.vmap(mala_update, in_axes=(None,(0,0,0,0)))
    mala_kernel_vec = jax.vmap(mala_kernel, in_axes=(0, 0, 0))
    logpdf = jax.vmap(logpdf)
    d_logpdf = jax.vmap(d_logpdf)

    # Apperantly jitting after vmap will make compilation much slower.
    # Output the kernel, logpdf, and dlogpdf for warmup jitting.
    # Apperantly passing in a warmed up function will still trigger recompilation.
    # so the warmup need to be done with the output function

    return mala_update, mala_kernel_vec, logpdf, d_logpdf

def make_mala_sampler(logpdf, d_logpdf, dt=1e-5, jit=False, M=None):
    mala_update, mk, lp, dlp = make_mala_update(logpdf, d_logpdf, dt, M)
    # Somehow if I define the function inside the other function,
    # I think it doesn't use the cache and recompile everytime.
    if jit:
        mala_update = jax.jit(mala_update)
        mk = jax.jit(mk)
        lp = jax.jit(lp)
        dlp = jax.jit(dlp)

    def mala_sampler(rng_key, n_steps, initial_position):

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
        acceptance = jnp.zeros((n_chains,n_steps,))
        all_positions = jnp.zeros((n_chains, n_steps,)+initial_position.shape[-1:]) + initial_position[:,None]
        all_logp = jnp.zeros((n_chains,n_steps,)) + logp[:,None]
        state = (rng_key, all_positions, all_logp, acceptance)
        for i in tqdm(range(1, n_steps),desc='Sampling Locally',miniters=int(n_steps/10)):
            state = mala_update(i, state)
        return state

    return mala_sampler, mala_update, mk, lp, dlp

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
