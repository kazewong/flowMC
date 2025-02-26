
# Quick Start


## Installation

The recommended way to install flowMC is using pip

```
pip install flowMC
```

This will install the latest stable release and its dependencies.
flowMC is based on [JAX](https://github.com/google/jax) and [Flax](https://github.com/google/flax).
By default, installing flowMC will automatically install JAX and Flax available on [PyPI](https://pypi.org/).
JAX does not install GPU support by default.
If you want to use GPU with JAX, you need to install JAX with GPU support according to [their document](https://github.com/google/jax#pip-installation-gpu-cuda-installed-via-pip-easier).
At the time of writing this documentation page, this is the command to install JAX with GPU support:

```
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

If you want to install the latest version of flowMC, you can clone this repo and install it locally:

```
git clone https://github.com/kazewong/flowMC.git
cd flowMC
pip install -e .
```
## Basic Usage


To sample a N dimensional Gaussian, you would do something like:

``` python
import jax
import jax.numpy as jnp
from flowMC.Sampler import Sampler
from flowMC.resource_strategy_bundles import RQSpline_MALA_Bundle

# Defining the log posterior

def log_posterior(x, data: dict):
    return -0.5 * jnp.sum((x - data["data"]) ** 2)

# Declaring hyperparameters

n_dims = 2
n_local_steps = 10
n_global_steps = 10
n_training_loops = 3
n_production_loops = 3
n_epochs = 10
n_chains = 10
rq_spline_hidden_units = [64, 64]
rq_spline_n_bins = 8
rq_spline_n_layers = 3
data = {"data": jnp.arange(n_dims).astype(jnp.float32)}

rng_key = jax.random.PRNGKey(42)
rng_key, subkey = jax.random.split(rng_key)
initial_position = jax.random.normal(subkey, shape=(n_chains, n_dims)) * 1

# Initializing the strategy bundle

rng_key, subkey = jax.random.split(rng_key)
bundle = RQSpline_MALA_Bundle(
    subkey,
    n_chains,
    n_dims,
    log_posterior,
    n_local_steps,
    n_global_steps,
    n_training_loops,
    n_production_loops,
    n_epochs,
    rq_spline_hidden_units=rq_spline_hidden_units,
    rq_spline_n_bins=rq_spline_n_bins,
    rq_spline_n_layers=rq_spline_n_layers,
)

# Run the sampler

nf_sampler = Sampler(
    n_dims,
    n_chains,
    rng_key,
    resource_strategy_bundles=bundle,
)

nf_sampler.sample(initial_position, data)
```

For more examples, there is a series of tutorials in the tutorials directory, we recommend you walk through the [Dual moon example](../tutorials/dualmoon) as a starting point.

In the ideal case, the only three things you will have to do are:

1. Write down the log-probability density function you want to sample in the form of `log_p(x)`, where `x` is the vector of variables of interest,
2. Give the sampler the initial position of your n_chains,
3. Choose your sampling strategy and initialize the sampler.

While this [configuration guide](#configuration_guide-section-top) can help in configuring the sampler, here are the two important bits you need to think about before using this package:

## 1. Write the likelihood function in JAX

While the package does support non-JAX likelihood function, it is highly recommended that you write your likelihood function in JAX for performance reasons.

If your likelihood is fully defined in [JAX](https://github.com/google/jax), there are a couple benefits that compound with each other:

1. JAX allows you to access the gradient of the likelihood function with respect to the parameters of the model through automatic differentiation. Having access to the gradient allows the use of gradient-based local sampler such as Metropolis-adjusted Langevin algorithm (MALA) and Hamiltonian Monte Carlo (HMC). These algorithms allow the sampler to handle high dimensional problems, and is often more efficient than the gradient-free local sampler such as Metropolis-Hastings.
2. JAX uses [XLA](https://www.tensorflow.org/xla) to compile your code not only into machine code but also in a way that is more optimized for accelerators such as GPUs and TPUs. Having multiple MCMC chains helps speed up the training of the normalizing flow. Accelerators such as GPUs and TPUs provide parallel computing solutions that are more scalable compared to CPUs.

Being able to run many chains in parallel helps training the normalizing flow model.

## 2. Start the chains wisely

For the global sampler to be effective, the normalizing flow needs to learn where there is mass in the target distribution. Once the flow overlaps with the target, non-local jumps will start to be accepted and the MCMC chains will mix quickly.

As the flow learns from the chains, starting the chains in regions of interest will speed up the convergence of the algorithm. If these regions are not known, a good rule of thumb is to start from random draws from the prior provided the prior is spread enough to cover high density regions of the posterior.
