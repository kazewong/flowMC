import jax
import jax.numpy as jnp  # JAX NumPy
import numpy as np
from scipy.stats import norm

from flowMC.nfmodel.realNVP import RealNVP
from flowMC.sampler.Gaussian_random_walk import rw_metropolis_sampler
from flowMC.sampler.Sampler import Sampler
from flowMC.utils.PRNG_keys import initialize_rng_keys
from flowMC.utils.PythonFunctionWrap import wrap_python_log_prob_fn

@wrap_python_log_prob_fn
def neal_funnel(x):
    y_pdf = norm.logpdf(x[0], loc=0, scale=3)
    x_pdf = norm.logpdf(x[1:], loc=0, scale=np.exp(x[0] / 2))
    return y_pdf + np.sum(x_pdf)


n_dim = 5
n_chains = 20
n_loop_training = 5
n_loop_production = 5
n_local_steps = 20
n_global_steps = 100
n_chains = 100
learning_rate = 0.01
momentum = 0.9
num_epochs = 100
batch_size = 1000

print("Preparing RNG keys")
rng_key_set = initialize_rng_keys(n_chains, seed=42)

print("Initializing MCMC model and normalizing flow model.")

initial_position = jax.random.normal(
    rng_key_set[0], shape=(n_chains, n_dim)
)  # (n_dim, n_chains)


# model = RQSpline(n_dim, 10, [128, 128], 8)
# run_mcmc = jax.vmap(rw_metropolis_sampler, in_axes=(0, None, None, 0), out_axes=0)

# print("Initializing sampler class")
# local_sampler_caller = lambda x: make_mala_sampler(x, jit=True)

print("Initializing sampler class")

nf_sampler = Sampler(
    n_dim,
    rng_key_set,
    local_sampler_caller,
    {'dt':1e-1},
    neal_funnel,
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
print("Sampling")

nf_sampler.sample(initial_position)
chains, log_prob, local_accs, global_accs, loss_vals = nf_sampler.get_sampler_state()
nf_samples = nf_sampler.sample_flow()

chains = np.array(chains)
nf_samples = np.array(nf_samples[1])
loss_vals = np.array(loss_vals)

import corner
import matplotlib.pyplot as plt

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
plt.savefig('temp_nonjaxlikelihood.png')

# Plot all chains
figure = corner.corner(
    chains.reshape(-1, n_dim), labels=["$x_1$", "$x_2$", "$x_3$", "$x_4$", "$x_5$"]
)
figure.set_size_inches(7, 7)
figure.suptitle("Visualize samples")
plt.show(block=False)
plt.savefig('temp_nonjaxlikelihood_1.png')

# Plot Nf samples
figure = corner.corner(nf_samples, labels=["$x_1$", "$x_2$", "$x_3$", "$x_4$", "$x_5$"])
figure.set_size_inches(7, 7)
figure.suptitle("Visualize NF samples")
plt.show(block=False)
plt.savefig('temp_nonjaxlikelihood_2.png')