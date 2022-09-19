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

Models in many scientific fields are growing more complex with more tuning parameters

Modern accelerators such as GPU and TPU are designed to execute dense computation in parallel.


# Acknowledgements

# References
