.. _tuning_guide-section-top:

Tuning Guide
============

This page contains information about what knobs can you tune when the sampler doesn't work as expected.



===============================    ===================================
Name                               Description
===============================    ===================================
likelihood                         Likelihood function to use.
local_sampler                      Local sampler to use.
d_likelihood                       Gradient of the likelihood function.
rng_keys_nf                        RNG keys for the normalizing flow.
rng_keys_mcmc                      RNG keys for the MCMC.
n_dim                              Dimension of the sampling problem.
n_loop                             Number of sampling loops.
n_local_steps                      Number of local steps in each sampling loop.
n_global_steps                     Number of global steps in each sampling loop.
n_chains                           Number of parallel chains.
stepsize                           Stepsize of the local sampler.
n_epochs                           Number of epochs in training the normalizing flow model.
n_nf_samples                       Number of samples drawn in each global sampling loop.
learning_rate                      Learning rate to use when training the normalizing flow model.
momentum                           Momentum to use when training the normalizing flow model.
batch_size                         Batch size to use when training the normalizing flow model.
logging                            Whether we log or not.
===============================    ===================================

Sampler class arguments
-----------------------

likelihood 


Local sampler parameters
------------------------

Normalizing flow parameters
---------------------------

flowMC.sampler.Sampler module
-----------------------------

.. automodule:: flowMC.sampler.Sampler
   :members:
   :noindex:

