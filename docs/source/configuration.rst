.. _configuration_guide-section-top:

Configuration Guide
===================

This page contains information about the most important hyperparameters which affect the behavior of the sampler.


Essential arguments
-------------------

``n_dim``

The dimension of the problem, the sampler would bug if ``n_dim`` does 

``rng_key_set``

The set of Jax generated PRNG_keys,

``local_sampler``

``sampler_params``

``likelihood``

``nf_model``

Optional arguments that you might want to tune
----------------------------------------------

``use_global``

``n_chains``

``n_loop_training``

``n_loop_production``

``n_local_steps``

``n_global_steps``

``n_epochs``

``learning_rate``

``max_samples``

``batch_size``



Only-if-you-know-what-you-are-doing arguments
---------------------------------------------

``keep_quantile``


flowMC.sampler.Sampler module
-----------------------------

.. automodule:: flowMC.sampler.Sampler
   :members:
   :noindex:

