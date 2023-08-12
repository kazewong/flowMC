
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


``` 
import jax
import jax.numpy as jnp
from flowMC.nfmodel.rqSpline import MaskedCouplingRQSpline
from flowMC.sampler.MALA import MALA
from flowMC.sampler.Sampler import Sampler
from flowMC.utils.PRNG_keys import initialize_rng_keys
from flowMC.nfmodel.utils import *

def log_posterior(x, data):
    return -0.5 * jnp.sum((x-data) ** 2)

data = jnp.arange(5)

n_dim = 5
n_chains = 10

rng_key_set = initialize_rng_keys(n_chains, seed=42)
initial_position = jax.random.normal(rng_key_set[0], shape=(n_chains, n_dim)) * 1
model = MaskedCouplingRQSpline(n_dim, 3, [64, 64], 8, jax.random.PRNGKey(21))
step_size = 1e-1
local_sampler = MALA(log_posterior, True, {"step_size": step_size})

nf_sampler = Sampler(n_dim,
                    rng_key_set,
                    jnp.arange(n_dim),
                    local_sampler,
                    model,
                    n_local_steps = 50,
                    n_global_steps = 50,
                    n_epochs = 30,
                    learning_rate = 1e-2,
                    batch_size = 1000,
                    n_chains = n_chains)

nf_sampler.sample(initial_position, data)
chains,log_prob,local_accs, global_accs = nf_sampler.get_sampler_state().values()
```

For more examples, have a look at the [tutorials](https://github.com/kazewong/flowMC/tree/main/example) on GitHub.
In particular, currently the best engineered test case is [dualmoon.py](https://github.com/kazewong/flowMC/blob/main/example/dualmoon.py).

In the ideal case, the only three things you will have to do are:

1. Write down the log-probability density function you want to sample in the form of `log_p(x)`, where `x` is the vector of variables of interest,
2. Give the sampler the initial position of your n_chains,
3. Configure the sampler and normalizing flow model.

While this [configuration guide](#configuration_guide-section-top) can help in configuring the sampler, here are the two important bits you need to think about before using this package:

## 1. Write the likelihood function in JAX

While the package does support non-JAX likelihood function, it is highly recommended that you write your likelihood function in JAX. This entails writing matrix operations using `jax.numpy` and making sure the code remains *functional* as in *functional programming*.

If your likelihood is fully defined in [JAX](https://github.com/google/jax), there are a couple benefits that compound with each other:

1. JAX allows you to access the gradient of the likelihood function with respect to the parameters of the model through automatic differentiation. Having access to the gradient allows the use of gradient-based local sampler such as Metropolis-adjusted Langevin algorithm (MALA) and Hamiltonian Monte Carlo (HMC). These algorithms allow the sampler to handle high dimensional problems, and is often more efficient than the gradient-free local sampler such as Metropolis-Hastings.
2. JAX uses [XLA](https://www.tensorflow.org/xla) to compile your code not only into machine code but also in a way that is more optimized for accelerators such as GPUs and TPUs. Having multiple MCMC chains helps speed up the training of the normalizing flow. Accelerators such as GPUs and TPUs provide parallel computing solutions that are more scalable compared to CPUs.

Being able to run many chains in parallel helps training the normalizing flow model.

## 2. Start the chains wisely

For the global sampler to be effective, the normalizing flow needs to learn where there is mass in the target distribution. Once the flow overlaps with the target, non-local jumps will start to be accepted and the MCMC chains will mix quickly.

As the flow learns from the chains, starting the chains in regions of interest will speed up the convergence of the algorithm. If these regions are not known, a good rule of thumb is to start from random draws from the prior provided the prior is spread enough to cover high density regions of the posterior.