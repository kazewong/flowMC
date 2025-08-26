import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
from jaxtyping import Array, Float

from flowMC.resource.buffers import Buffer
from flowMC.resource.kernel.MALA import MALA
from flowMC.resource.logPDF import LogPDF
from flowMC.resource.states import State
from flowMC.Sampler import Sampler
from flowMC.strategy.take_steps import TakeSerialSteps


def dual_moon_pe(x: Float[Array, "n_dims"], data: dict):
    """
    Term 2 and 3 separate the distribution
    and smear it along the first and second dimension
    """
    print("compile count")
    term1 = 0.5 * ((jnp.linalg.norm(x - data["data"]) - 2) / 0.1) ** 2
    term2 = -0.5 * ((x[:1] + jnp.array([-3.0, 3.0])) / 0.8) ** 2
    term3 = -0.5 * ((x[1:2] + jnp.array([-3.0, 3.0])) / 0.6) ** 2
    return -(term1 - logsumexp(term2) - logsumexp(term3))


n_dims = 5
n_chains = 15
n_local_steps = 30
step_size = 0.01

data = {"data": jnp.arange(5)}

rng_key = jax.random.PRNGKey(42)
rng_key, subkey = jax.random.split(rng_key)
initial_position = jax.random.normal(subkey, shape=(n_chains, n_dims)) * 1

# Defining resources

MALA_Sampler = MALA(step_size=step_size)
positions = Buffer("positions", (n_chains, n_local_steps, n_dims), 1)
log_prob = Buffer("log_prob", (n_chains, n_local_steps), 1)
acceptance = Buffer("acceptance", (n_chains, n_local_steps), 1)
sampler_state = State(
    {
        "positions": "positions",
        "log_prob": "log_prob",
        "acceptance": "acceptance",
    },
    name="sampler_state",
)

resource = {
    "positions": positions,
    "log_prob": log_prob,
    "acceptance": acceptance,
    "MALA": MALA_Sampler,
    "logpdf": LogPDF(dual_moon_pe, n_dims=n_dims),
    "sampler_state": sampler_state,
}

# Defining strategy

strategy = TakeSerialSteps(
    "logpdf",
    kernel_name="MALA",
    state_name="sampler_state",
    buffer_names=["positions", "log_prob", "acceptance"],
    n_steps=n_local_steps,
)

nf_sampler = Sampler(
    n_dim=n_dims,
    n_chains=n_chains,
    rng_key=rng_key,
    resources=resource,
    strategies={"take_steps": strategy},
    strategy_order=[("take_steps", "testing_phase")],
)

nf_sampler.sample(initial_position, data)
