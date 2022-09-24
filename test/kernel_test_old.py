from flowMC.nfmodel.realNVP import RealNVP
import jax
import jax.numpy as jnp  # JAX NumPy
from flowMC.utils import Sampler, initialize_rng_keys
from jax.scipy.special import logsumexp
import numpy as np


from flowMC.nfmodel.utils import *


def mala_kernel(rng_key, logpdf, d_logpdf, position, log_prob, kernal_size=0.1):

    key1, key2 = jax.random.split(rng_key)
    proposal = position + kernal_size * d_logpdf(position)
    proposal += (
        kernal_size
        * jnp.sqrt(2 / kernal_size)
        * jax.random.normal(key1, shape=position.shape)
    )
    ratio = logpdf(proposal) - logpdf(position)
    ratio -= (
        (position - proposal - kernal_size * d_logpdf(proposal)) ** 2
        / (4 * kernal_size)
    ).sum()
    ratio += (
        (proposal - position - kernal_size * d_logpdf(position)) ** 2
        / (4 * kernal_size)
    ).sum()
    proposal_log_prob = logpdf(proposal)

    log_uniform = jnp.log(jax.random.uniform(key2))
    do_accept = log_uniform < ratio

    position = jnp.where(do_accept, proposal, position)
    log_prob = jnp.where(do_accept, proposal_log_prob, log_prob)
    return position, log_prob, do_accept.astype(jnp.int8)


mala_kernel_vec = jax.vmap(mala_kernel, in_axes=(0, None, None, 0, 0, None))


def mala_sampler(
    rng_key, n_samples, logpdf, d_logpdf, initial_position, kernal_size=0.1
):
    def mh_update_sol2(i, state):
        key, positions, log_prob, acceptance = state
        _, key = jax.random.split(key)
        new_position, new_log_prob, accept_local = mala_kernel(
            key, logpdf, d_logpdf, positions[i - 1], log_prob, kernal_size
        )
        positions = positions.at[i].set(new_position)
        acceptance += accept_local
        return (key, positions, new_log_prob, acceptance)

    logp = logpdf(initial_position)
    acceptance = jnp.zeros(logp.shape)
    all_positions = jnp.zeros((n_samples,) + initial_position.shape) + initial_position
    initial_state = (rng_key, all_positions, logp, acceptance)
    rng_key, all_positions, log_prob, acceptance = jax.lax.fori_loop(
        1, n_samples, mh_update_sol2, initial_state
    )

    return rng_key, all_positions, log_prob, acceptance / n_samples


def dual_moon_pe(x):
    """
    Term 2 and 3 separate the distriubiotn and smear it along the first and second dimension
    """
    term1 = 0.5 * ((jnp.linalg.norm(x) - 2) / 0.1) ** 2
    term2 = -0.5 * ((x[:1] + jnp.array([-3.0, 3.0])) / 0.8) ** 2
    term3 = -0.5 * ((x[1:2] + jnp.array([-3.0, 3.0])) / 0.6) ** 2
    return -(term1 - logsumexp(term2) - logsumexp(term3))


d_dual_moon = jax.jit(jax.grad(dual_moon_pe))
d_dual_moon_vec = jax.jit(jax.vmap(jax.grad(dual_moon_pe)))
dual_moon_pe_vec = jax.jit(jax.vmap(dual_moon_pe))


n_dim = 5
n_samples = 20
nf_samples = 100
n_chains = 100
learning_rate = 0.01
momentum = 0.9
num_epochs = 100
batch_size = 1000


config = {}
config["n_dim"] = 5
config["n_loop"] = 5
config["n_local_steps"] = 20
config["n_global_steps"] = 100
config["n_chains"] = 100
config["learning_rate"] = 0.01
config["momentum"] = 0.9
config["num_epochs"] = 100
config["batch_size"] = 1000
config["stepsize"] = 0.01

print("Preparing RNG keys")
rng_key_set = initialize_rng_keys(config["n_chains"], seed=42)

print("Initializing MCMC model and normalizing flow model.")

initial_position = jax.random.normal(
    rng_key_set[0], shape=(config["n_chains"], config["n_dim"])
)


model = RealNVP(10, config["n_dim"], 64, 1)
run_mcmc = jax.vmap(mala_sampler, in_axes=(0, None, None, None, 0, None), out_axes=0)

# print("Initializing sampler class")

# nf_sampler = Sampler(rng_key_set, config, model, run_mcmc, dual_moon_pe, d_dual_moon)

# print("Sampling")

# chains, nf_samples = nf_sampler.sample(initial_position)

# import corner
# import matplotlib.pyplot as plt

# # Plot one chain to show the jump
# plt.plot(chains[70,:,0],chains[70,:,1])
# plt.show()
# plt.close()

# # Plot all chains
# corner.corner(chains.reshape(-1,n_dim), labels=["$x_1$", "$x_2$", "$x_3$", "$x_4$", "$x_5$"])
