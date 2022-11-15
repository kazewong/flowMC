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
n_local_steps = 30
step_size = 0.1
n_leapfrog = 10

rng_key_set = initialize_rng_keys(n_chains, seed=42)

initial_position = jax.random.normal(rng_key_set[0], shape=(n_chains, n_dim)) * 1

HMC = HMC(dual_moon_pe, True, {"step_size": step_size,"n_leapfrog": n_leapfrog, "inverse_metric": jnp.ones(n_dim)})

initial_Ham = jax.vmap(HMC.get_initial_hamiltonian, in_axes=(0, 0, None))(rng_key_set[1], initial_position, HMC.params)

HMC_kernel = HMC.make_kernel()

print(HMC_kernel(rng_key_set[0], initial_position[0], initial_Ham[0], HMC.params))

HMC_update = HMC.make_update()
HMC_update = jax.vmap(HMC_update, in_axes = (None, (0, 0, 0, 0, None)), out_axes=(0, 0, 0, 0, None))

initial_position = jnp.repeat(initial_position[:,None], n_local_steps, 1)
initial_Ham = jnp.repeat(initial_Ham[:,None], n_local_steps, 1)

state = (rng_key_set[1], initial_position, initial_Ham, jnp.zeros((n_chains, n_local_steps,1)), HMC.params)


HMC_update(1, state)

HMC_sampler = HMC.make_sampler()

state = HMC_sampler(rng_key_set[1], n_local_steps, initial_position[:, 0])


from flowMC.nfmodel.rqSpline import RQSpline
from flowMC.sampler.Sampler import Sampler

n_dim = 5
n_chains = 2
n_local_steps = 3
n_global_steps = 3
step_size = 0.1
n_loop_training = 2
n_loop_production = 2

rng_key_set = initialize_rng_keys(n_chains, seed=42)

initial_position = jax.random.normal(rng_key_set[0], shape=(n_chains, n_dim)) * 1

model = RQSpline(n_dim, 4, [32, 32], 8)

print("Initializing sampler class")

nf_sampler = Sampler(
    n_dim,
    rng_key_set,
    HMC,
    dual_moon_pe,
    model   ,
    n_loop_training=n_loop_training,
    n_loop_production=n_loop_production,
    n_local_steps=n_local_steps,
    n_global_steps=n_global_steps,
    n_chains=n_chains,
    use_global=False,
)

nf_sampler.sample(initial_position)