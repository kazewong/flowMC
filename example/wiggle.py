from nfsampler.nfmodel.realNVP import RealNVP
from nfsampler.sampler.MALA import mala_sampler
import jax
import jax.numpy as jnp                # JAX NumPy
from nfsampler.utils import Sampler, initialize_rng_keys
from jax.scipy.special import logsumexp
import numpy as np  

from nfsampler.nfmodel.utils import *
import matplotlib.pyplot as plt


def log_prob(x):
    """
    Wiggle-distribution 
    """
    mean = jnp.array([0, 6])
    centered_x = x[0] - mean - jnp.sin(5 * x[1] / 5)
    log_prob = - 0.5 * centered_x @ jnp.eye(2) @ centered_x.T
    log_prob -= 0.5 * (jnp.linalg.norm(x) - 5) ** 2 / 8
    return log_prob

def plot_2d_level(log_prob, dim=2, x_min=-10, x_max=10,
                  y_min=None, y_max=None,
                  n_points=1000, ax=None, title=''):
    """
    Args:
    model (RealNVP_MLP or MoG): must have a .sample and .U method
    """
    # if 
    x_range = np.linspace(x_min, x_max, n_points)
    if y_min is None:
        y_range = x_range
    else:
        y_range = np.linspace(y_min, y_max, n_points)

    grid = np.meshgrid(x_range, y_range)
    xys = np.stack(grid).reshape(2, n_points ** 2).T
    if dim > 2:
        blu = np.zeros(n_points ** 2, dim)
        blu[:, 0:2] = xys
        xys = blu

    log_probs = jax.vmap(log_prob)(xys).reshape(n_points, n_points)
    
    if ax is None:
        plt.figure()
    else:
        plt.sca(ax)
    plt.contourf(x_range, y_range, np.exp(log_probs), 10)
    # plt.axis('off')
    plt.colorbar()
    plt.title(title)
    return log_probs

d_log_prob = jax.grad(log_prob)

config = {}
config['n_dim'] = 2
config['n_loop'] = 10
config['n_local_steps'] = 20
config['n_global_steps'] = 100
config['n_chains'] = 100
config['learning_rate'] = 0.01
config['momentum'] = 0.9
config['num_epochs'] = 10
config['batch_size'] = 1000
config['stepsize'] = 0.01
config['logging'] = True

print("Preparing RNG keys")
rng_key_set = initialize_rng_keys(config['n_chains'],seed=42)

print("Initializing MCMC model and normalizing flow model.")

initial_position = jax.random.normal(rng_key_set[0],shape=(config['n_chains'],config['n_dim'])) #(n_dim, n_chains)


model = RealNVP(10, config['n_dim'], 64, 1)
run_mcmc = jax.vmap(mala_sampler, in_axes=(0, None, None, None, 0, None),
                    out_axes=0)

print("Initializing sampler class")

nf_sampler = Sampler(rng_key_set, config, model, run_mcmc, log_prob, d_log_prob)

print("Sampling")

chains, nf_samples = nf_sampler.sample(initial_position)

import corner
import matplotlib.pyplot as plt

# Plot one chain to show the jump
plot_2d_level(log_prob)
plt.plot(chains[70,:,0],chains[70,:,1])
plt.show(block=False)

# Plot all chains
corner.corner(chains.reshape(-1, config['n_dim']), labels=["$x_1$", "$x_2$"])
plt.show(block=False)
