import jax
import jax.numpy as jnp

from flowMC.resource.nf_model.rqSpline import MaskedCouplingRQSpline
from flowMC.resource.nf_model.NF_proposal import NFProposal
from flowMC.resource.local_kernel.MALA import MALA
from flowMC.resource.buffers import Buffer
from flowMC.resource.optimizer import Optimizer

from flowMC.strategy.take_steps import TakeSerialSteps, TakeGroupSteps
from flowMC.strategy.train_model import TrainModel

from flowMC.Sampler import Sampler


def log_posterior(x, data: dict):
    return -0.5 * jnp.sum((x - data['data']) ** 2)



n_dim = 2
n_steps = 25
n_chains = 10
data = {'data':jnp.arange(n_dim)}

rng_key = jax.random.PRNGKey(42)
rng_key, subkey = jax.random.split(rng_key)
initial_position = jax.random.normal(subkey, shape=(n_chains, n_dim)) * 1

positions = Buffer("positions", n_chains, n_steps, n_dim)
log_prob = Buffer("log_prob", n_chains, n_steps, 1)
local_accs = Buffer("local_accs", n_chains, n_steps, 1)
global_accs = Buffer("global_accs", n_chains, n_steps, 1)
local_sampler = MALA(step_size=1e-1)
rng_key, subkey = jax.random.split(rng_key)
model = MaskedCouplingRQSpline(n_dim, 3, [64, 64], 8, subkey)
global_sampler = NFProposal(model)
optimizer = Optimizer(model=model)

resources = {
    "positions": positions,
    "log_prob": log_prob,
    "local_accs": local_accs,
    "global_accs": global_accs,
    "model": model,
    "optimizer": optimizer,
}

local_step = TakeSerialSteps(log_posterior, local_sampler, ["positions", "log_prob", "local_accs"], n_steps)
train_model = TrainModel("model", "positions", "optimizer", n_epochs=10, batch_size=10000)
global_step = TakeGroupSteps(log_posterior, global_sampler, ["positions", "log_prob", "global_accs"], n_steps)

strategy = [local_step, train_model, global_step]

nf_sampler = Sampler(
    n_dim,
    n_chains,
    log_posterior,
    rng_key,
    resources,
    strategy,
)

nf_sampler.sample(initial_position, data)
# chains, log_prob, local_accs, global_accs = nf_sampler.get_sampler_state().values()
