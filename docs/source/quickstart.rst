.. _quickstart-section-top:

Quick Start
============

Installation
------------

The recommended way to install flowMC is using pip

.. code-block::

    pip install flowMC

This will install the latest stable release and its dependencies.
flowMC is based on `Jax <https://github.com/google/jax>`_ and `Flax <https://github.com/google/flax>`_.
By default, installing flowMC will automatically install Jax and Flax available on `PyPI <https://pypi.org/>`_.
Jax does not install GPU support by default.
If you want to use GPU with Jax, you need to install Jax with GPU support according to `their document <pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html>`_.
At the time of writing this documentation page, this is the command to install Jax with GPU support:

.. code-block::

    pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html


If you want to install the latest version of flowMC, you can clone this repo and install it locally:

.. code-block::

    git clone https://github.com/kazewong/flowMC.git
    cd flowMC
    pip install -e .

Basic Usage
-----------

A minimum example to sample a N dimensional Gaussian, you would do something like:

.. code-block:: python

    import jax
    import jax.numpy as jnp
    from flowMC.nfmodel.rqSpline import RQSpline
    from flowMC.sampler.MALA import make_mala_sampler
    from flowMC.sampler.Sampler import Sampler
    from flowMC.utils.PRNG_keys import initialize_rng_keys
    from flowMC.nfmodel.utils import *

    def posterior(x):
        return -0.5 * jnp.sum(x ** 2)

    n_dim = 5
    n_chains = 10

    rng_key_set = initialize_rng_keys(n_chains, seed=42)
    initial_position = jax.random.normal(rng_key_set[0], shape=(n_chains, n_dim)) * 1
    model = RQSpline(n_dim, 3, [64, 64], 8)
    local_sampler_caller = lambda x: make_mala_sampler(x, jit=True)
    sampler_params = {'dt': 5e-1}

    nf_sampler = Sampler(n_dim,
                        rng_key_set,
                        local_sampler_caller,
                        sampler_params,
                        posterior,
                        model,
                        n_local_steps = 50,
                        n_global_steps = 50,
                        n_epochs = 30,
                        learning_rate = 1e-2,
                        batch_size = 1000,
                        n_chains = n_chains)

    nf_sampler.sample(initial_position)
    chains,log_prob,local_accs, global_accs = nf_sampler.get_sampler_state().values()

For more realistic test case, see the examples on `github <https://github.com/kazewong/flowMC/tree/main/example>`_.
In particular, currently the best engineered test case is `dualmoon.py <https://github.com/kazewong/flowMC/blob/main/example/dualmoon.py>`_.

In the ideal case, the only three things you will have to do are:

#. Write down the function you want to sample in the form of :code:`p(x)`, where :code:`p` is the log probability density function and :code:`x` is the vector of variables of interest.
#. Configure the sampler.
#. Give the sampler the initial position of your n_chains

While this guide can help in configuring the sampler, here are the two important bits you need to think about before using this package:

1. You better write the likelihood function in JAX
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

While the package does support non-Jax likelihood function, it is highly recommended that you write your likelihood function in JAX.

If your likelihood is fully defined in Jax, there are a couple benefits that compound with each other:
#. Jax allows you to access the gradient of the likelihood function with respect to the parameters of the model through automatic differentiation.
   Having access to the gradient allows the use of gradient-based local sampler such as Metropolis-adjusted Langevin algorithm (MALA) and Hamiltonian Monte Carlo (HMC).
   These algorithms allow the sampler to handle high dimensional problems, and is often more efficient than the gradient-free local sampler such as Metropolis-Hastings.
#. Jax uses `XLA <https://www.tensorflow.org/xla>`_ to compile your code not only into machine code but also in a way that is more optimized for accelerators such as GPUs and TPUs.
   Having multiple MCMC chains helps speed up the training of the normalizing flow. Accelerators such as GPUs and TPUs provide parallel computing solutions that are more scalable compared to CPUs.



Being able to run many chains in parallel helps training the normalizing flow model.

1. Starting at the right position might help training and sampling.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^