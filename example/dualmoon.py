import jax
import jax.numpy as jnp  # JAX NumPy
from jax.scipy.special import logsumexp
import numpy as np

from flowMC.nfmodel.rqSpline import MaskedCouplingRQSpline
from flowMC.sampler.MALA import MALA
from flowMC.sampler.Sampler import Sampler
from flowMC.utils.PRNG_keys import initialize_rng_keys

import corner
import matplotlib.pyplot as plt


def target_dualmoon(x, data):
    """
    Term 2 and 3 separate the distribution and smear it
    along the first and second dimension
    """
    print("compile count")
    term1 = 0.5 * ((jnp.linalg.norm(x - data) - 2) / 0.1) ** 2
    term2 = -0.5 * ((x[:1] + jnp.array([-3.0, 3.0])) / 0.8) ** 2
    term3 = -0.5 * ((x[1:2] + jnp.array([-3.0, 3.0])) / 0.6) ** 2
    return -(term1 - logsumexp(term2) - logsumexp(term3))


n_dim = 5
n_chains = 20
n_loop_training = 5
n_loop_production = 5
n_local_steps = 100
n_global_steps = 100
learning_rate = 0.001
momentum = 0.9
num_epochs = 30
batch_size = 10000

data = jnp.zeros(n_dim)

rng_key_set = initialize_rng_keys(n_chains, 42)
model = MaskedCouplingRQSpline(n_dim, 4, [32, 32], 8, PRNGKeyArray(10))

initial_position = jax.random.normal(rng_key_set[0], shape=(n_chains, n_dim)) * 1

MALA_Sampler = MALA(target_dualmoon, True, {"step_size": 0.1})

print("Initializing sampler class")

nf_sampler = Sampler(
    n_dim,
    rng_key_set,
    jnp.zeros(5),
    MALA_Sampler,
    model,
    n_loop_training=n_loop_training,
    n_loop_production=n_loop_production,
    n_local_steps=n_local_steps,
    n_global_steps=n_global_steps,
    n_chains=n_chains,
    n_epochs=num_epochs,
    learning_rate=learning_rate,
    momentum=momentum,
    batch_size=batch_size,
    use_global=True,
)

nf_sampler.sample(initial_position, data)
summary = nf_sampler.get_sampler_state(training=True)
chains, log_prob, local_accs, global_accs, loss_vals = summary.values()
nf_samples = nf_sampler.sample_flow(10000)

print(
    "chains shape: ",
    chains.shape,
    "local_accs shape: ",
    local_accs.shape,
    "global_accs shape: ",
    global_accs.shape,
)

chains = np.array(chains)
nf_samples = np.array(nf_samples[1])
loss_vals = np.array(loss_vals)

# Plot one chain to show the jump
plt.figure(figsize=(6, 6))
axs = [plt.subplot(2, 2, i + 1) for i in range(4)]
plt.sca(axs[0])
plt.title("2 chains")
plt.plot(chains[0, :, 0], chains[0, :, 1], alpha=0.5)
plt.plot(chains[1, :, 0], chains[1, :, 1], alpha=0.5)
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")

plt.sca(axs[1])
plt.title("NF loss")
plt.plot(loss_vals.reshape(-1))
plt.xlabel("iteration")

plt.sca(axs[2])
plt.title("Local Acceptance")
plt.plot(local_accs.mean(0))
plt.xlabel("iteration")

plt.sca(axs[3])
plt.title("Global Acceptance")
plt.plot(global_accs.mean(0))
plt.xlabel("iteration")
plt.tight_layout()
plt.show(block=False)

# Plot all chains
figure = corner.corner(
    chains.reshape(-1, n_dim), labels=["$x_1$", "$x_2$", "$x_3$", "$x_4$", "$x_5$"]
)
figure.set_size_inches(7, 7)
figure.suptitle("Visualize samples")
plt.show(block=False)

# Plot Nf samples
figure = corner.corner(nf_samples, labels=["$x_1$", "$x_2$", "$x_3$", "$x_4$", "$x_5$"])
figure.set_size_inches(7, 7)
figure.suptitle("Visualize NF samples")
plt.show()
