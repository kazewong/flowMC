from flowMC.sampler.MALA import MALA
from flowMC.sampler.flowHMC import flowHMC
from flowMC.utils.PRNG_keys import initialize_rng_keys
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
from flowMC.nfmodel.rqSpline import MaskedCouplingRQSpline
from flowMC.sampler.Sampler import Sampler


def log_posterior(x, data):
    """
    Term 2 and 3 separate the distribution and smear it along the first and second dimension
    """
    print("compile count")
    term1 = 0.5 * ((jnp.linalg.norm(x - data) - 2) / 0.1) ** 2
    term2 = -0.5 * ((x[:1] + jnp.array([-3.0, 3.0])) / 0.8) ** 2
    term3 = -0.5 * ((x[1:2] + jnp.array([-3.0, 3.0])) / 0.6) ** 2
    return -(term1 - logsumexp(term2) - logsumexp(term3))


n_dim = 8
n_chains = 15
n_local_steps = 30
step_size = 0.1
n_leapfrog = 3

data = jnp.arange(n_dim)

rng_key_set = initialize_rng_keys(n_chains, seed=42)

initial_position = jax.random.normal(rng_key_set[0], shape=(n_chains, n_dim)) * 1
model = MaskedCouplingRQSpline(n_dim, 4, [32, 32], 4, PRNGKeyArray(10))

local_sampler = MALA(log_posterior, True, {"step_size": step_size})

step_size = 0.01
flowHMC_sampler = flowHMC(
    log_posterior,
    True,
    model,
    params={
        "step_size": step_size,
        "n_leapfrog": n_leapfrog,
        "condition_matrix": jnp.eye(n_dim),
    },
)

n_steps = 50
rng_key, *subkeys = jax.random.split(PRNGKeyArray(0), 3)

n_chains = initial_position.shape[0]
n_dim = initial_position.shape[-1]
log_prob_initial = flowHMC_sampler.logpdf_vmap(initial_position, data)

proposal_position, proposal_metric = flowHMC_sampler.sample_flow(
    subkeys[0], initial_position, n_steps
)

initial_PE = flowHMC_sampler.logpdf_vmap(initial_position, data)

momentum = jax.random.normal(subkeys[0], shape=initial_position.shape)

nf_sampler = Sampler(
    n_dim,
    rng_key_set,
    data,
    local_sampler,
    model,
    n_local_steps=50,
    n_global_steps=50,
    n_epochs=30,
    learning_rate=1e-2,
    batch_size=10000,
    n_chains=n_chains,
    global_sampler=flowHMC_sampler,
    verbose=True,
)

nf_sampler.sample(initial_position, data)
