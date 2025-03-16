
# Quick Start


## Installation

The recommended way to install flowMC is using pip

```
pip install flowMC
```

This will install the latest stable release and its dependencies.
flowMC is based on [JAX](https://github.com/google/jax) and [Equinox](https://github.com/patrick-kidger/equinox).
By default, installing flowMC will automatically install JAX and Equinox available on [PyPI](https://pypi.org/).
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

If you have [uv](https://docs.astral.sh/uv/) installed, you can also install the latest version of flowMC by running:

```
uv sync
```

once you have cloned the repo.


## Basic Usage


To sample an N dimensional Gaussian, you would do something like:

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

# Initializing the strategy bundle
rng_key = jax.random.PRNGKey(42)
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

rng_key, subkey = jax.random.split(rng_key)
initial_position = jax.random.normal(subkey, shape=(n_chains, n_dims)) * 1
nf_sampler = Sampler(
    n_dims,
    n_chains,
    rng_key,
    resource_strategy_bundles=bundle,
)

nf_sampler.sample(initial_position, data)
```

In the ideal case, the only three things you will have to do are:

1. Write down the log-probability density function you want to sample in the form of `log_p(x)`, where `x` is the vector of variables of interest,
2. Choose your sampling strategy and hyperparameters,
3. Give the sampler the initial position of your chains and start sampling.

Given the scripts above, you can start playing with the sampler and see how it behaves. Below is a more detail description of `flowMC` and some guiding principles when using `flowMC`.

## Anatomy of flowMC

Prior to version 0.4.0, `flowMC` was a package that was designed to execute the algorithm detailed in [this paper](https://arxiv.org/pdf/2105.12603). Since then the community has tried applying `flowMC` to different problems. While there were some successes, there are also limiting factors in terms of performance. One of the biggest issues `flowMC` faced is the fact that the global-local sampling algorithm were baked into the top level `Sampler` API, which means `flowMC` can only use the exact algorithm described in the paper. What if the users want to use a different model? Or run some optimization steps during the sampling stage? Or apply annealing? These are either impossible or not very intuitive in `flowMC` prior to version 0.4.0.

Seeing this limitation, we redesigned the middle level API of `flowMC` while keeping the top level API as similar as possible. This guide aims to describe the different components of `flowMC` and how they interact with each other, and give users who want to extend `flowMC` to optimize for their specific problems a starting point on what could be useful to change. This also acts as a rule of thumb for users who want to use `flowMC` as a black box and interact with internal components through hyperparameters only.

### Target distribution

The target distribution should be defined as a log-probability density function, which follows the following function signature:

```python
def target_log_prob_fn(x: Float[Array, "n_dims"], data: dict[str, Any]) -> Float:
    ...
    return log_prob
```

The `target_log_prob_fn` should take in a `Float[Array, "n_dims"]` array `x` and a dictionary `data` that contains any additional data that the target distribution depends on. The function should return a scalar `Float` that is the log-probability density of the target distribution at `x`.

To ensure the target distribution is well-defined and performant, you should also check whether the function is behaving as expected when `jax.jit` and `jax.grad` are applied to it. 

### Sampler

On the top level, the `Sampler` class is a thin wrapper on top of the resource-strategy pair (defined below) that provides a couple of extra functionality. The `Sampler` class manages the resources and strategies, as well as run-related parameters such as where would the resources be stored if the user decides to serialize the resources.

```python
nf_sampler = Sampler(
    n_dims,
    n_chains,
    rng_key,
    # you can either supply the resources and strategies directly,
    # which is prioritized over the resource-strategy bundles
    resources=resources,
    strategies=strategies,
    # or you can supply the resource-strategy bundles
    resource_strategy_bundles=bundle,
)
```

The main loop of `Sampler` is pretty straight forward after initialization: Given the available resources, iterate through the list of strategies, which each takes the resources, perform some actions (such as taking local steps or training a normalizing flow), and return the updated resources. In the current implementation, the `Sampler` simply goes through the list of strategy, but in the future we are planning to more flexible main loop such as automatic stopping based on some criteria.

### Resource and Strategy

At the core of the new `flowMC` API are the resource and strategy interfaces. Broadly speaking resources are similar to a data class, and strategies are similar to functions.
**Resources** store some attribute and can be manipulated, but should not have too many methods associated with it. For example, a buffer that stores the sampling results is a resource, a MALA kernel is a resource, and a normalizing flow model is a resource. **Strategies** are functions that take in resources and return updated resources. For example, taking a local step requires two kinds of resources: a proposal distribution and the buffer where the samples are stored. Examples of strategies are taking a local step, training a normalizing flow, and running an optimization step.

If you are initializing the sources and strategies directly, you can do something like:

```python
resources = {
    "buffer": Buffer(name, n_chains, n_steps, n_dims),
    "proposal": MALA(step_size),
    "model": NormalizingFlow(model_parameters),
}

strategies = [
    Strategy1(),
    Strategy2(),
]
```

The reason for this separation is to allow users to compose different strategies together. For example, the user may want to update the parameters of a proposal kernel like MALA with the local information from a normalizing flow model. Instead of hard coding this functionality to associate with either the MALA kernel or the normalizing flow model, the current API allows the user to define a strategy that takes in both the MALA kernel and the normalizing flow model, and update the MALA kernel with the information from the normalizing flow model. This separate the concern of intermixing different components of the algorithm and make experimenting with new strategies more manageable.

Since this API is designed for users who are willing to look into the guts of `flowMC` and experiment with different strategies, the main question to ask is whether a new data structure/functionality should be a resource or a strategy. While there is no hard rules for such implementation other than conforming to the individual base classes, a good rule of thumb is to ask whether the new data structure/functionality is something that should be updated by other strategies. If the answer is yes, then it should be a resource. If the answer is no, then it should be a strategy.

One extra criteria that decides whether an implementation should be a resource or a strategy is whether the implementation is compatible with `jax`'s transformation. Resource should be compatible with `jit`, and strategy is not required to be compatible with `jit`. An example to illustrate the difference is a training loop contains for-looping over a number of epochs and logging the metadata, which is usually not necessary to be jitted, so this should be a strategy. A neural network and its main functions needs to run efficiently on GPU no matter in sampling or training, so it should be a resource.

You can find the hyper-parameters of a resource, a strategy, or a resource-strategy bundles in the API docs.

## Guiding principles

### Write the likelihood function in JAX

If your likelihood is fully defined in [JAX](https://github.com/google/jax), there are a couple benefits that compound with each other:

1. JAX allows you to access the gradient of the likelihood function with respect to the parameters of the model through automatic differentiation. Having access to the gradient allows the use of gradient-based local sampler such as Metropolis-adjusted Langevin algorithm (MALA) and Hamiltonian Monte Carlo (HMC). These algorithms allow the sampler to handle high dimensional problems, and is often more efficient than the gradient-free local sampler such as Metropolis-Hastings.
2. JAX uses [XLA](https://www.tensorflow.org/xla) to compile your code not only into machine code but also in a way that is more optimized for accelerators such as GPUs and TPUs. Having multiple MCMC chains helps speed up the training of the normalizing flow. Accelerators such as GPUs and TPUs provide parallel computing solutions that are more scalable compared to CPUs.

Since version 0.4.0, we made the design choice of removing support for likelihood functions incompatible with `jax` transformations. The reason is that `flowMC` is designed to leverage GPU acceleration and machine learning methods to solve complex problems. If a developer decides to use `flowMC` to try to solve their problem, it is also a good time to consider rewriting their legacy code base in `jax`, which on its own could provide a significant speedup. Instead of letting people off the hook by allowing non-jax compatible likelihood functions, we decided to enforce the use of `jax` to encourage users to take advantage of its benefits.

### Parallelize whenever you can

One should center their choice of resource and strategy around leveraging parallelization. This is reflected by the fact that `n_chains` is a required parameter for the `Sampler` class. The reason for this is `flowMC` is designed to solve problems with complex geometry using adaptive sampling method such as training a normalizing flow alongside with a local proposal, which benefit tremendously from having multiple chains running in parallel.
