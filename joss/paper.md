---
title: 'FlowMC'
tags:
  - Python
  - Bayesian Inference 
  - Machine Learning
  - Jax
  - 
authors:
  - name: Kaze W. K. Wong
    orcid: 0000-0001-8432-7788
    equal-contrib: true
    affiliation: 1 
  - name:  Marylou Gabrié
    orcid: 0000-0002-5989-1018
    equal-contrib: true 
    affiliation: 2
  - name: Dan Foreman-Mackey
    orcid: 0000-0002-9328-5652
    affiliation: 1
affiliations:
 - name: Center for Computational Astrophysics, Flatiron Institute, New York, NY 10010, US
   index: 1
 - name: Institution Name, Country
date: 30 September 2022
bibliography: paper.bib
---

# Summary

`FlowMC` is a Python library for accelerated Markov Chain Monte Carlo (MCMC) building on top of `Jax` and `Flax`.
At its core, `FlowMC` uses a local sampler such as Metropolis-adjusted Langevin algorithm (MALA) and normalizing flow models in tandem to efficiently sample posterior distributions with non-trivial geometry,
such as multimodal distributions and distributions with local correlations.
While multiple chains of the local sampler generate samples over the region of interest in the target parameter space, the package uses these samples to train a normalizing flow model, then use it to propose global jumps across the parameter space.
The key features of `FlowMC` are summarized in the following list:

## Key features

- Since `FlowMC` is built on top of `Jax`, it supports gradient-based sampler such as MALA and Hamiltonian Monte Carlo (HMC) through automatic differentiation.
- `FlowMC` uses state-of-the-art normalizing flow models such as rational quadratic spline (RQS) for the global sampler, which is very efficient in capturing local features with relatively short training time.
- Use of accelerators such as GPUs and TPUs are natively supported. The code also supports the use of multiple accelerators with SIMD parallelism.
- By default, Just-in-time (JIT) compilation are used to further speed up the sampling process. 
- We provide a simple black box interface for the users who want to use `FlowMC` by its default parameters, at the same time provide an extensive guide explaining trade-off while tuning the sampler parameters.

The tight integration of all the above features makes `FlowMC` a highly performant yet simple-to-use package for Bayesian inference.

# Statement of need

***Gradient-based sampler***
Models in many scientific fields are growing more complex to capture complicated physical processes.
One common way to increase the complexity of a model is to introduce more parameters.
This increases the flexibility in the model, but it makes downstream data analysis tasks such as parameter estimation more challenging since introducing new parameters increase the dimension of problem.
In a high dimensional space, sampling methods which leverage gradient information of the target distribution such as MALA and HMC are shown to be more efficient in proposing new samples with higher acceptance rate.
`FlowMC` supports gradient-based samplers such as MALA and HMC through automatic differentiation with `Jax`.
The computational cost of obtaining the gradient information in this way is often about the same order as evaluating the target function itself,
which makes the extra computational costs in computing the gradient information usually a favorable trade-off for the increased efficiency in sampling.


***Learned reparameterization with normalizing flow***
While gradient-based sampler such as MALA and HMC are powerful in decorrelating random variables with a problem, their capability are limited to global correlation.
Posterior distribution of many real-world problems can have non-trivial geometry such as multi-modality and local correlation, which could drastically slow down the convergence of the sampler.
To address this problem, we combine gradient-based sampler with normalizing flow, which is a class of generative model that can learn the geometry of the posterior distribution, as the proposal distribution.
As individual chains are exploring their local neighbor, multiple chains can be combined and fed to the normalizing flow, such that the normalizing flow learn the global landscape of the posterior distribution.
Since we are only using the normalizing flow as a proposal distribution, the entire algorithm is still essentially a MCMC method, meaning one assess the robustness of the inference result using diagnostics one would use to assess other MCMC methods.
This means we do not have to worry about validation of the normalizing flow model, which is a common problem in deep learning.
The normalizing flow is trained in parallel to the sampling process, so no pre-training is required.
The mathematical detail of the method are explained in (cite)

***Use of Accelerator***
Modern accelerators such as GPU and TPU are designed to execute dense computation in parallel.
Due to the sequential nature of MCMC, a common approach in leveraging accelerators is to run multiple chains in parallel, then combine their results to obtain the posterior distribution.
However, large portion of the computation comes from the burn-in phase, and simply by parallelizing over many chains do not help speed up the burn-in.
To fully leverage the benefit from having many chains, ensemble methods such as (Cite) are often implemented.
This comes with its own set of challenges, and implementing such class of methods on accelerators require careful consideration.
Because the benefit from accelerators is not clear ahead of time and the hefty cost of implementation, 
there are not many MCMC libraries that are designed to take advantage on accelerators.
Since `FlowMC` is built on top of `Jax`, it supports the use of accelerators by default.
Users can write codes in the same way as they would do on a CPU, and the library will automatically detect the available accelerators and use them in run time.
Furthermore, the library leverage Just-In-Time compilations to further improve the performance of the sampler.

***Simplicity and extensibility***
Since we anticipate most of the users would like to spend most of their time building model instead of optimize the performance of the sampler,
we provide a black-box interface with a few tuning parameters for users who intend to use `FlowMC` without too much customization on the sampler side.
The only inputs we require from the users are the log-likelihood function, the log-prior function, and initial position of the chains.
<!-- Mention something related to auto tune -->

While we provide a high-level API for most of the users, the code is also designed to be extensible.
<!-- Say something about extensibility like custom proposal -->

# Acknowledgements

# References
