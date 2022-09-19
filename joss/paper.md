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
affiliations:
 - name: Center for Computational Astrophysics, Flatiron Institute, New York, NY 10010, US
   index: 1
 - name: Institution Name, Country
date: 30 September 2022
bibliography: paper.bib
---

# Summary

`FlowMC` is a Python library for accelerated Markov Chain Monte Carlo (MCMC) building on top of `Jax` and `Flax`.


# Key features

- `FlowMC` by default employ gradient-based sampler such as Metropolis-adjusted Langevin algorithm (MALA)
- Use of accelerators such as GPU and TPU are natively supported. The code also supports the use of multiple accelerators with SIMD parallelism.
- `FlowMC` provides an interface to train normalizing flow models using `Flax`.
- We provide a simple blackbox interface for the users who want to use `FlowMC` by its default parameters, at the same time provide an extensive guide explaining trade-off while tuning the sampler parameters.
- We keep the library relatively lightweight and extensible. We provide examples of how to combine `FlowMC` with other libraries such as `harmonics`

# Statement of need

***Gradient-based sampler***
Models in many scientific fields are growing more complex with more tuning parameters

***Learned reparameterization with normalizing flow***
While gradient-based sampler such as MALA and HMC are powerful in decorrelating random variables with a problem, their capability are limited to global correlation.
Posterior distribution of many real-world problems can have non-trivial geometry such as multi-modality and local correlation, which could drastically slow down the convergence of the sampler.
To address this problem, we combine gradient-based sampler with normalizing flow, which is a class of generative model that can learn the geometry of the posterior distribution, as the proposal distribution.
The normalizing flow is trained in parallel to the sampling process, so no pre-training is required.

***Use of Accelerator***
Modern accelerators such as GPU and TPU are designed to execute dense computation in parallel.
Due to the sequential nature of MCMC, a common approach in leveraging accelerators is to run multiple chains in parallel, then combine their results to obtain the posterior distribution.
However, large portion of the computation comes from the burn-in phase, and simply by parallelizing over many chains do not help speed up the burn-in.
To fully leverage the benefit from having many chains, ensemble methods such as (Cite) are often implemented.
This comes with its own set of challenges, and implementing such class of methods on accelerators require careful consideration.
Because the benefit from accelerators is not clear ahead of time and the hefty cost of implementation, 
there are not many MCMC libraries that are designed to take advantage on accelerators.

***Simplicity and extensibility***


# Acknowledgements

# References
