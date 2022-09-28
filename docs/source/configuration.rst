.. _configuration_guide-section-top:

Configuration Guide
===================

This page contains information about the most important hyperparameters which affect the behavior of the sampler.

Essential arguments
-------------------

``n_dim``

The dimension of the problem, the sampler would bug if ``n_dim`` does not match the input dimension of your likelihood function

``rng_key_set``

The set of Jax generated PRNG_keys.

``local_sampler``

Specific local sampler you want to use.

``sampler_params``

Specific parameters from a particular local sampler.

``likelihood``

Target function you want to sample.

``nf_model``

Specific normalizing flow model you want to use.

Optional arguments that you might want to tune
----------------------------------------------

``use_global``

Whether to use global sampler or not. Default is ``True``.
Turning off global sampler will also disable to training phase.
This is useful when you want to test whether the local sampler is behaving normally.
In production quality runs, you probably always want to use the global sampler since it improves convergence significantly.

``n_chains``

Number of parallel chains to run. Default is ``20``.
Within your memory bandwidth and without oversubscribing your computational device, you should use as many chains as possible.
Our method benefits tremendously from parallelization.

``n_loop_training``

Number of local-global sample loop to run during training phase. Default is ``3``.


``n_loop_production``

Number of local-global sample loop to run during production phase. Default is ``3``.
This is similar to ``n_loop_training``, the only difference is during the production loop,
the normalizing flow model will not be updated in order to maintain detail balance.

``n_local_steps``

Number of local steps to run during the local sampling step. Default is ``50``.

``n_global_steps``

Number of global steps to run during the global sampling step. Default is ``50``.

``n_epochs``

Number of epochs to run during the training phase. Default is ``30``.

``learning_rate``

Learning rate for the Adam optimizer. Default is ``1e-2``.

``max_samples``

Maximum number of samples used to training the normalizing flow model. Default is ``10000``.

``batch_size``

Batch size for training the normalizing flow model. Default is ``10000``.



Only-if-you-know-what-you-are-doing arguments
---------------------------------------------

``keep_quantile``


flowMC.sampler.Sampler module
-----------------------------

.. automodule:: flowMC.sampler.Sampler
   :members:
   :noindex:

