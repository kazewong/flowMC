.. _quickstart-section-top:

Quick Start
============

Installation
------------

The recommended way to install FlowMC is using pip

.. code-block::

    pip install flowMC

This will install the latest stable release and its dependencies.
FlowMC is based on `Jax <https://github.com/google/jax>`_ and `Flax <https://github.com/google/flax>`_.
By default, installing flowMC will automatically install Jax and Flax available on `PyPI <https://pypi.org/>`_.
Jax does not install GPU support by default.
If you want to use GPU with Jax, you need to install Jax with GPU support according to `their document <pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html>`_.
At the time of writing this documentation page, this is the command to install Jax with GPU support:

.. code-block::

    pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html


Basic Usage
-----------

A minimum example to sample a N dimensional Gaussian, you would do something like:

.. code-block:: python

    import jax
    import jax.numpy as jnp
    from flowMC.nfmodel.realNVP import RealNVP
    from flowMC.sampler.MALA import mala_sampler
    from flowMC.sampler.Sampler import Sampler
    from flowMC.utils.PRNG_keys import initialize_rng_keys
    from flowMC.nfmodel.utils import *

    def log_prob(x):
        return -0.5 * jnp.sum(x ** 2)

    d_logp = jax.grad(log_prob)

    n_dim = 5
    n_chains = 10

    rng_key_set = initialize_rng_keys(n_chains, seed=42)
    initial_position = jax.random.normal(rng_key_set[0], shape=(n_chains, n_dim)) * 1

    model = RealNVP(10, n_dim, 64, 1)
    run_mcmc = jax.vmap(mala_sampler, in_axes=(0, None, None, None, 0, None), out_axes=0)

    nf_sampler = Sampler(n_dim, rng_key_set, model, run_mcmc,
                        log_prob,
                        n_chains=n_chains,
                        d_likelihood=d_logp,)

    nf_sampler.sample(initial_position)

For more realistic test case, see the examples on `github <https://github.com/kazewong/FlowMC/tree/main/example>`_.
In particular, currently the best engineered test case is `dualmoon.py <https://github.com/kazewong/FlowMC/blob/main/example/dualmoon.py>`_.

In the ideal case, the only two things you will have to do is:

#. Write down the function you want to sample in the form of :code:`p(x)`, where :code:`p` is the log probability density function and :code:`x` is the vector of variables of interest.
#. Configure the sampler.

