from flowMC.sampler.HMC import HMC
from flowMC.utils.PRNG_keys import initialize_rng_keys
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

def dual_moon_pe(x):
    """
    Term 2 and 3 separate the distribution and smear it along the first and second dimension
    """
    term1 = 0.5 * ((jnp.linalg.norm(x) - 2) / 0.1) ** 2
    term2 = -0.5 * ((x[:1] + jnp.array([-3.0, 3.0])) / 0.8) ** 2
    term3 = -0.5 * ((x[1:2] + jnp.array([-3.0, 3.0])) / 0.6) ** 2
    return -(term1 - logsumexp(term2) - logsumexp(term3))

n_dim = 5
n_chains = 15
step_size = 0.1
n_leapfrog = 10

rng_key_set = initialize_rng_keys(n_chains, seed=42)

initial_position = jax.random.normal(rng_key_set[0], shape=(n_chains, n_dim)) * 1

HMC = HMC(dual_moon_pe, False, {"step_size": step_size,"n_leapfrog": n_leapfrog})

initial_Ham = HMC.get_initial_hamiltonian(rng_key_set[0], initial_position[0])

HMC_kernel = HMC.make_hmc_kernel()

print(HMC_kernel(rng_key_set[0], initial_position[0], initial_Ham))