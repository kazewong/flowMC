import jax
import numpy as np
import jax.numpy as jnp
from scipy.stats import norm
from flowMC.nfmodel.rqSpline import MaskedCouplingRQSpline
from flowMC.sampler.Gaussian_random_walk import GaussianRandomWalk

from flowMC.sampler.Sampler import Sampler
from flowMC.utils.PRNG_keys import initialize_rng_keys
from flowMC.utils.PythonFunctionWrap import wrap_python_log_prob_fn

"""
The purpose of this example is to demonstate this is doable with the code, but here are some honest warnings:
1. It is probably not going to be as fast as you think/hope it to be, because of the communication between host and device.
2. You cannot jit and grad through the likelihood function, so only random walks gaussian proposal is supported.
3. Making this work with other parallelization scheme such as MPI may be tricky because of Jax.
4. Your code won't run on GPU.

So when you say your code is too much to rewrite in Jax but you still want to use flowMC, ask yourself these questions:
1. How long would it take to rewrite your code in Jax? Weeks? Months? Years? If it is just a couple months, maybe one should really consider doing it.
2. Can I recast the problem in a way that I can have a jax likelihood? For example can you do some surrogate modeling to replace the likelihood?

If the answer to any of these two questions is yes, then you should probably do it. If the answer to both of these questions is no, then maybe consider some other alternative such as PocoMC.
"""


@wrap_python_log_prob_fn
def neal_funnel(x):
    y_pdf = norm.logpdf(x["params"][0], loc=0, scale=3)
    x_pdf = norm.logpdf(x["params"][1:], loc=0, scale=np.exp(x["params"][0] / 2))
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

data = jnp.zeros(n_dim)

rng_key_set = initialize_rng_keys(n_chains, 42)
model = MaskedCouplingRQSpline(n_dim, 4, [32, 32], 8, jax.random.PRNGKey(10))

initial_position = jax.random.normal(rng_key_set[0], shape=(n_chains, n_dim)) * 1

RW_Sampler = GaussianRandomWalk(neal_funnel, False, {"step_size": 0.1})


print("Initializing sampler class")

nf_sampler = Sampler(
    n_dim,
    rng_key_set,
    jnp.zeros(5),
    RW_Sampler,
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