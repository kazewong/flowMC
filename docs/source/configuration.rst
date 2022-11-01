.. _configuration_guide-section-top:

Configuration Guide
===================

This page contains information about the most important hyperparameters which affect the behavior of the sampler.

Essential arguments
-------------------

n_dim
^^^^^

The dimension of the problem, the sampler would bug if ``n_dim`` does not match the input dimension of your likelihood function

rng_key_set
^^^^^^^^^^^^^

The set of Jax generated PRNG_keys.

local_sampler
^^^^^^^^^^^^^

Specific local sampler you want to use.

sampler_params
^^^^^^^^^^^^^^

Specific parameters from a particular local sampler.

likelihood
^^^^^^^^^^

Target function you want to sample.

nf_model
^^^^^^^^

Specific normalizing flow model you want to use.

Optional arguments that you might want to tune
----------------------------------------------

use_global
^^^^^^^^^^

Whether to use global sampler or not. Default is ``True``.
Turning off global sampler will also disable to training phase.
This is useful when you want to test whether the local sampler is behaving normally.
In production quality runs, you probably always want to use the global sampler since it improves convergence significantly.

n_chains
^^^^^^^^

Number of parallel chains to run. Default is ``20``.
Within your memory bandwidth and without oversubscribing your computational device, you should use as many chains as possible.
Our method benefits tremendously from parallelization.

n_loop_training
^^^^^^^^^^^^^^^

Number of local-global sample loop to run during training phase. Default is ``3``.

n_loop_production
^^^^^^^^^^^^^^^^^

Number of local-global sample loop to run during production phase. Default is ``3``.
This is similar to ``n_loop_training``, the only difference is during the production loop,
the normalizing flow model will not be updated in order to maintain detail balance.

n_local_steps
^^^^^^^^^^^^^

Number of local steps to run during the local sampling step. Default is ``50``.

n_global_steps
^^^^^^^^^^^^^^

Number of global steps to run during the global sampling step. Default is ``50``.

n_epochs
^^^^^^^^

Number of epochs to run during the training phase. Default is ``30``.
The higher this number, the better the flow performs, at the cost of increasing the training time.

learning_rate
^^^^^^^^^^^^^

Learning rate for the Adam optimizer. Default is ``1e-2``.

max_samples
^^^^^^^^^^^

Maximum number of samples used to training the normalizing flow model. Default is ``10000``.
If the total number of samples is more than this parameters, only up to this parameters of samples will be fed into the normalizing flow model.
The chains dimension has priority over step dimension, meaning the sampler will try to take at least one sample from each chains before going to previous steps to retrieve more samples.
One usually choose this number base on the memory capacity of the device.
If the number is larger than the memory bandwidth of your device, each training loop will take longer to finish.
On the otherhand, the training time will not be affect if the entire dataset can fit on your device.
If this number is too small, it means only the most recent samples are used in the training.
This may cause the normalizing flow model to forget about the global landscape too quickly, leading to mode collapse.


batch_size
^^^^^^^^^^

Batch size for training the normalizing flow model. Default is ``10000``.
Using large batch size speeds up the training since the training time is determined by the number of batched backward passes.
Unlikely typical deep learning use case, since our training dataset is continuously evolving, we do not really have to worry about overfitting.
Therefore, using larger batch size is usually better.
The rule of thumb here is: within memory and computational bandwith, choose the largest number that would not decrease the training time.


Only-if-you-know-what-you-are-doing arguments
---------------------------------------------

``keep_quantile``


flowMC.sampler.Sampler module
-----------------------------

.. automodule:: flowMC.sampler.Sampler
   :members:
   :noindex:

