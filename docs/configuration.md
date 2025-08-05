Configuration Guide
===================

This page contains information about the most important hyperparameters which affect the behavior of the sampler.


| Essential                         | Optional                                  | Advanced                            |
| --------------------------------- | ----------------------------------------- | ----------------------------------- |
| [`n_dim`](#n_dim)                 | [`use_global`](#use_global)               | [`keep_quantile`](#keep_quantile)   |
| [`rng_key_set`](#rng_key_set)     | [`n_chains`](#n_chains)                   | [`momenutum`](#momenutum)           |
| [`local_sampler`](#local_sampler) | [`n_loop_training`](#n_loop_training)     | [`nf_variable`](#nf_variable)       |
| [`data`](#data)                   | [`n_loop_production`](#n_loop_production) | [`local_autotune`](#local_autotune) |
| [`nf_model`](#nf_model)           | [`n_local_steps`](#n_local_steps)         | [`train_thinning`](#train_thinning) |
|                                   | [`n_global_steps`](#n_global_steps)       |                                     |
|                                   | [`n_epochs`](#n_epochs)                   |                                     |
|                                   | [`learning_rate`](#learning_rate)         |                                     |
|                                   | [`max_samples`](#max_samples)             |                                     |
|                                   | [`batch_size`](#batch_size)               |                                     |
|                                   | [`verbose`](#verbose)                     |                                     |

   


Essential arguments
-------------------

## [n_dim](#n_dim)

The dimension of the problem, the sampler would bug if `n_dim` does not match the input dimension of your likelihood function

## [rng_key_set](#rng_key_set)

The set of Jax generated PRNG_keys.

## [data](#data)

The data you want to sample from. This is used to precompile the kernels used during the sampling.
Note that you keep the shape of the data consistent between runs, otherwise it would trigger recompilation.
If your likelihood does not take any data arguments, simply put it as None should work.

## [local_sampler](#local_sampler)
Specific local sampler you want to use.

## [nf_model](#nf_model)
Specific normalizing flow model you want to use.

Optional arguments that you might want to tune
----------------------------------------------

## [use_global](#use_global)
Whether to use global sampler or not. Default is ``True``.
Turning off global sampler will also disable to training phase.
This is useful when you want to test whether the local sampler is behaving normally or perform an ablation study on the benefits of the global sampler.
In production quality runs, you probably always want to use the global sampler since it improves convergence significantly.

## [n_chains](#n_chains)
Number of parallel chains to run. Default is ``20``.
Within your memory bandwidth and without oversubscribing your computational device, you should use as many chains as possible.
The method benefits tremendously from parallelization.

## [n_loop_training](#n_loop_training)
Number of local-global sample loop to run during training phase. Default is ``3``.

## [n_loop_production](#n_loop_production)
Number of local-global sample loop to run during production phase. Default is ``3``.
This is similar to ``n_loop_training``, the only difference is during the production loop, the normalizing flow model will not be updated anymore. This saves computation time once the flow is sufficiently trained to power global moves. As the MCMC stops being adaptive, detailed balance is also restored in the Metropolis-Hastings steps and the traditional diagnostic of MCMC convergence can be applied safely.


## [n_local_steps](#n_local_steps)
Number of local steps to run during the local sampling phase. Default is ``50``.

## [n_global_steps](#n_global_steps)
Number of global steps to run during the global sampling phase. Default is ``50``.

## [n_epochs](#n_epochs)
Number of epochs to run during the training phase. Default is ``30``.
The higher this number, the better the flow performs, at the cost of increasing the training time.

## [learning_rate](#learning_rate)
Learning rate for the Adam optimizer. Default is ``1e-2``.

## [max_samples](#max_samples)
Maximum number of samples used to training the normalizing flow model. Default is ``10000``.

If the total number of obtained samples along the chains is more than ``max_samples`` when getting to a training phase, only a subsample of the most recent steps of the chains of size ``max_samples`` will be used for training.
.. The chains dimension has priority over step dimension, meaning the sampler will try to take at least one sample from each chain before going to previous steps to retrieve more samples.
One usually choose this number base on the memory capacity of the device.
If the number is larger than the memory bandwidth of your device, each training loop will take longer to finish.
On the other hand, the training time will not be affect if the entire dataset can fit on your device.
If this number is small only the most recent samples are used in the training.
This may cause the normalizing flow model to forget about some features of the global landscape that were not visited recently by the chains. For instance, it can lead to mode collapse.

## [batch_size](#batch_size)
Batch size for training the normalizing flow model. Default is ``10000``.
Using large batch size speeds up the training since the training time is determined by the number of batched backward passes.
Unlike typical deep learning use case, since our training dataset is continuously evolving, we do not really have to worry about overfitting.
Therefore, using larger batch size is usually better.
The rule of thumb here is: within memory and computational bandwith, choose the largest number that would not increase the training time.

## [keep_quantile](#keep_quantile)
Dictionary with keys ``params`` and ``variables`` allowing to use the model trained during a previous run of the NFSampler. These variables can be retrieved from the ``NFSampler.state`` after a run. An exemple is provided in :ref:`tutorials`.

## [verbose](#verbose)
Whether to print out extra info during the inference. Default is ``False``.



Only-if-you-know-what-you-are-doing arguments
---------------------------------------------


## [keep_quantile](#keep_quantile)

## [momentum](#momentum)

## [nf_variable](#nf_variable)

## [local_autotune](#local_autotune)

## [train_thinning](#train_thinning)

Thinning factors for data used to train the normalizing flow.
Given we only keep ``max_samples`` Samples, only the newest ``max_samples/n_chains`` in each chain are used for training the normalizing flow.
This thinning factor keep every ``train_thinning`` samples in each chain.
The larger the number, the less correlated each samples in each chain are.
When ``max_samples*train_thining/n_chains > n_local_steps``, samples generated from different global training loops are used in training the new normalizing flow.
This reduces the possibility of mode collapse since the algorithms had access to samples generated before the mode collapse if it would have happened.

This API is still experimental and might be combined with other hyperparameters into one big tuning parameters later.
