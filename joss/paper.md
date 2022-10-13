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
    # equal-contrib: true
    affiliation: 1 
  - name:  Marylou Gabrié
    orcid: 0000-0002-5989-1018
    # equal-contrib: true 
    affiliation: "2, 3"
  - name: Dan Foreman-Mackey
    orcid: 0000-0002-9328-5652
    affiliation: 1
affiliations:
  - name: Center for Computational Astrophysics, Flatiron Institute, New York, NY 10010, US
    index: 1
  - name: École Polytechnique, Palaiseau 91120, France
    index: 2
  - name: Center for Computational Mathematics, Flatiron Institute, New York, NY 10010, US
    index: 3
date: 30 September 2022
bibliography: paper.bib
---

# Summary

Across scientific fields, the modelling of increasingly complex physical processes requires more flexible models. Yet the estimation of models'parameters becomes more challenging as the dimension of the parameter space grows. A common strategy to explore parameter space is to sample through a Markov Chain Monte Carlo (MCMC). Yet even MCMC methods can struggle to fully represent the parameter space when only relying on local updates.

`FlowMC` is a Python library for accelerated Markov Chain Monte Carlo (MCMC) leveraging deep generative modelling built on top of machine learning libraries `Jax` and `Flax`. At its core, `FlowMC` uses a local sampler and a learnable global sampler in tandem to efficiently sample posterior distributions with non-trivial geometry, such as multimodal distributions and distributions with local correlations. While multiple chains of the local sampler generate samples over the region of interest in the target parameter space, the package uses these samples to train a normalizing flow model, then use it to propose global jumps across the parameter space.

The key features of `FlowMC` are summarized in the following list:

## Key features

- Since `FlowMC` is built on top of `Jax`, it supports gradient-based sampler such as MALA and Hamiltonian Monte Carlo (HMC) through automatic differentiation.
- `FlowMC` uses state-of-the-art normalizing flow models such as rational quadratic spline (RQS) for the global sampler, which is very efficient in capturing local features with relatively short training time.
- Use of accelerators such as GPUs and TPUs are natively supported. The code also supports the use of multiple accelerators with SIMD parallelism.
- By default, Just-in-time (JIT) compilations are used to further speed up the sampling process. 
- We provide a simple black box interface for the users who want to use `FlowMC` by its default parameters, yet provide at the same time an extensive guide explaining trade-offs while tuning the sampler parameters.

The tight integration of all the above features makes `FlowMC` a highly performant yet simple-to-use package for statistical inference.

# Statement of need

Bayesian inference requires to compute expectations with respect to the posterior distribution on the parameters $\theta$ after collecting the observations $\mathcal{D}$. This posterior is given by 

$$
p(\theta|\mathcal{D}) = \frac{\ell(\mathcal{D}|\theta) p_0(\theta)}{Z(\mathcal{D})}  
$$

where $\ell(\mathcal{D}|\theta)$ is the likelihood induced by the model,  $p_0(\theta)$ the prior on the parameters and  $Z(\mathcal{D})$ the model evidence. 
As soon as the dimension of $\theta$ exceeds 3 or 4, it is necessary to resort to a robust sampling strategy such as a MCMC. Drastic gains in computational efficiency can be obtained by a careful selection of the MCMC transition kernel which can be assisted by machine learning methods and libraries.  

***Gradient-based sampler***
In a high dimensional space, sampling methods which leverage gradient information of the target distribution are shown to be efficient by proposing new samples likely to be accepted.
`FlowMC` supports gradient-based samplers such as MALA and HMC through automatic differentiation with `Jax`.
The computational cost of obtaining a gradient in this way is often of the same order as evaluating the target function itself, making gradient-based samplers compare usually favorably to random walks with respect to the efficiency/accuracy trade-off.

***Learned transition kernels with normalizing flow***
Posterior distribution of many real-world problems have non-trivial geometry such as multi-modality and local correlation, which could drastically slow down the convergence of the sampler only based on gradient information.
To address this problem, we combine a gradient-based sampler with a normalizing flow, which is a class of generative model `[@Papamakarios2019; @Kobyzev2021]`, that is trained to mimic the posterior distribution and used as a proposal a Metropolis-Hastings step. Variant of this idea have been explored in the past few years (e.g.`[@Albergo2019; @Hoffman2019; @Gabrie2021]` and references there in).
Despite the growing interest for these methods few accessible implementations for non-experts already exist and none of them propose GPU and TPU. Namely, a version of the NeuTra sampler `[@Hoffman2019]` available in Pyro `[@bingham2019pyro]` and the PocoMC package `[@Karamanis2022]` are both CPU bounded.

`FlowMC` implements the proposition of `[@Gabrie2021a]`. 
As individual chains are exploring their local neighborhood through gradient-based MCMC steps, multiple chains can be combined and fed to the normalizing flow so it can learn the global landscape of the posterior distribution. In turn, the chains can be propagated with a Metropolis-Hastings kernel using the normalizing flow to propose globally in the parameter space. The cycle of local sampling, normalizing flow tuning and global sampling is repeated until convergence of the chains.
The entire algorithm belongs to the class of adaptive MCMCs `[@Andrieu2008]` collecting information from the chains previous steps to simultaneously improve the transition kernel. 
Usual MCMC diagnostics can be applied to asses the robustness of the inference results without worrying about the validation of the normalizing flow model, which is a common problem in deep learning. 
If further sampling from the posterior is necessary, the flow trained during a previous can be reused without further training. 
The mathematical detail of the method are explained in `[@Gabrie2021a]`.

***Use of Accelerator***
Modern accelerators such as GPU and TPU are designed to execute dense computation in parallel.
Due to the sequential nature of MCMC, a common approach in leveraging accelerators is to run multiple chains in parallel, then combine their results to obtain the posterior distribution.
However, large portion of the computation comes from the burn-in phase, and simply by parallelizing over many chains do not help speed up the burn-in.
To fully leverage the benefit from having many chains, ensemble methods such as (Cite) are often implemented.
This comes with its own set of challenges, and implementing such class of methods on accelerators require careful consideration.
<!-- Because the benefit from accelerators is not clear ahead of time and the hefty cost of implementation, 
there are not many MCMC libraries that are designed to take advantage on accelerators. -->
Since `FlowMC` is built on top of `Jax`, it supports the use of accelerators by default.
Users can write codes in the same way as they would do on a CPU, and the library will automatically detect the available accelerators and use them in run time.
Furthermore, the library leverage Just-In-Time compilations to further improve the performance of the sampler.

***Simplicity and extensibility***
Since we anticipate most of the users would like to spend most of their time building model instead of optimize the performance of the sampler,
we provide a black-box interface with a few tuning parameters for users who intend to use `FlowMC` without too much customization on the sampler side.
The only inputs we require from the users are the log-likelihood function, the log-prior function, and initial position of the chains.
On top of the black-box interface, the package offers automatic tuning for the local samplers, in order to reduce the number of hyperparameters the users have to manage.

While we provide a high-level API for most of the users, the code is also designed to be extensible. In particular, custom local and global sampling kernels can be integrated in the `sampler` module. 
<!-- Say something about extensibility like custom proposal -->

# Acknowledgements
M.G. acknowledges support from Hi!Paris.
# References
