import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
from jaxtyping import Array, Float

from flowMC.resource.local_kernel.Gaussian_random_walk import GaussianRandomWalk
from flowMC.strategy.take_steps import TakeSerialSteps
from flowMC.resource.buffers import Buffer
from flowMC.resource.logPDF import LogPDF
from flowMC.Sampler import Sampler


def dual_moon_pe(x: Float[Array, " n_dims"], data: dict):
    """
    Term 2 and 3 separate the distribution
    and smear it along the first and second dimension
    """
    print("compile count")
    term1 = 0.5 * ((jnp.linalg.norm(x - data["data"]) - 2) / 0.1) ** 2
    term2 = -0.5 * ((x[:1] + jnp.array([-3.0, 3.0])) / 0.8) ** 2
    term3 = -0.5 * ((x[1:2] + jnp.array([-3.0, 3.0])) / 0.6) ** 2
    return -(term1 - logsumexp(term2) - logsumexp(term3))


# Test parameters
n_dims = 5
n_chains = 2
n_local_steps = 3
n_global_steps = 3
step_size = 0.1

data = {"data": jnp.arange(5)}

# Initialize random key and position
rng_key = jax.random.PRNGKey(43)
rng_key, subkey = jax.random.split(rng_key)
initial_position = jax.random.normal(subkey, shape=(n_chains, n_dims)) * 1

# Define resources
RWMCMC_sampler = GaussianRandomWalk(step_size=step_size)
positions = Buffer("positions", (n_chains, n_local_steps, n_dims), 1)
log_prob = Buffer("log_prob", (n_chains, n_local_steps), 1)
acceptance = Buffer("acceptance", (n_chains, n_local_steps), 1)

# Initialize normalizing flow model
rng_key, subkey = jax.random.split(rng_key)

resource = {
    "logpdf": LogPDF(dual_moon_pe, n_dims=n_dims),
    "positions": positions,
    "log_prob": log_prob,
    "acceptance": acceptance,
    "RWMCMC": RWMCMC_sampler,
}

# Define strategy
strategy = TakeSerialSteps(
    "logpdf",
    kernel_name="RWMCMC",
    buffer_names=["positions", "log_prob", "acceptance"],
    n_steps=n_local_steps,
)

print("Initializing sampler class")

# Initialize and run sampler
nf_sampler = Sampler(
    n_dim=n_dims,
    n_chains=n_chains,
    rng_key=rng_key,
    resources=resource,
    strategies={"take_steps": strategy},
    strategy_order=["take_steps"],
)

nf_sampler.sample(initial_position, data)
