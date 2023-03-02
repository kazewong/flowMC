---
title: 'flowMC: Normalizing flow enhanced sampling package for probabilistic inference in JAX'
tags:
  - Python
  - Bayesian Inference 
  - Machine Learning
  - JAX
authors:
  - name: Kaze W. K. Wong
    orcid: 0000-0001-8432-7788
    # equal-contrib: true
    affiliation: 1 
  - name:  Marylou Gabrié
    orcid: 0000-0002-5989-1018
    # equal-contrib: true 
    affiliation: "2, 3"
  - name: Daniel Foreman-Mackey
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
Across scientific fields, the Bayesian framework is used to account for uncertainties in inference [@ellison2004bayesian; @lancaster2004introduction; @von2011bayesian]. However, for models with more than a few parameters, exact inference is intractable. A frequently used strategy is to approximately sample the posterior distribution on a model's parameters with a Markov chain Monte Carlo (MCMC) method. Yet conventional MCMC methods relying on local updates can take a prohibitive time to converge when posterior distributions have complex geometries (see e.g., @rubinstein_simulation_2017).

`flowMC` is a Python library implementing accelerated MCMC leveraging deep generative modelling as proposed by @Gabrie2022, built on top of the machine learning libraries JAX [@jax2018github] and Flax [@flax2020github]. At its core, `flowMC` uses a combination of Metropolis-Hastings Markov kernels using local and global proposed moves. While multiple chains are run using local-update Markov kernels to generate approximate samples over the region of interest in the target parameter space, these samples are used to train a normalizing flow (NF) model to approximate the samples' density. The NF is then used in an independent Metropolis-Hastings kernel to propose global jumps across the parameter space. The `flowMC` sampler can handle non-trivial geometry, such as multimodal distributions and distributions with local correlations. 

The key features of `flowMC` are summarized in the following list:

- Since `flowMC` is built on top of JAX, it supports gradient-based samplers through automatic differentiation such as the Metropolis-adjusted Langevin algorithm (MALA) and Hamiltonian Monte Carlo (HMC).
- `flowMC` uses state-of-the-art NF models such as rational quadratic splines (RQS) to power its global proposals. These models are efficient in capturing important features within a relatively short training time.
- Use of accelerators such as graphics processing units (GPUs) and tensor processing units (TPUs) are natively supported. The code also supports the use of multiple accelerators with SIMD parallelism.
- By default, just-in-time (JIT) compilations are used to further accelerate the sampling process. 
- We provide a simple black box interface for the users who want to use `flowMC` by its default parameters, yet provide at the same time an extensive guide explaining trade-offs while tuning the sampler parameters.

The tight integration of all the above features makes `flowMC` a highly performant yet simple-to-use package for statistical inference.

# Statement of need

Bayesian inference requires computing expectations with respect to a posterior distribution on parameters $\theta$ after collecting observations $\mathcal{D}$. This posterior is given by 

$$
p(\theta|\mathcal{D}) = \frac{\ell(\mathcal{D}|\theta) p_0(\theta)}{Z(\mathcal{D})},  
$$

where $\ell(\mathcal{D}|\theta)$ is the likelihood induced by the model, $p_0(\theta)$ the prior on the parameters and  $Z(\mathcal{D})$ the model evidence. 
For parameter space with more than a few dimensions, it is necessary to resort to a robust sampling strategy such as MCMC. Drastic gains in computational efficiency can be obtained by a careful selection of the MCMC transition kernel which can be assisted by machine learning methods and libraries.  

***Gradient-based sampler***
In a high dimensional space, sampling methods which leverage gradient information of the target distribution aid by proposing new samples likely to be accepted.
`flowMC` supports gradient-based samplers such as MALA and HMC through automatic differentiation with JAX.


***Learned transition kernels with NFs***
When the posterior distribution has a non-trivial geometry, such as multiple modes or spatially dependent correlation structures (e.g, [@neal2003slice]), samplers based on local updates are inefficient.
To address this problem, `flowMC` also uses a generative model, namely a NF [@Papamakarios2019; @Kobyzev2021], that is trained to mimic the posterior distribution and used as a proposal in Metropolis-Hastings MCMC steps. Variants of this idea have been explored in the past few years [e.g., @Parno2018; @Albergo2019; @Hoffman2019].
Despite the growing interest in these methods, few accessible implementations for practitioners exist, especially with GPU and TPU support. Notably, a version of the NeuTra sampler [@Hoffman2019] is available in Pyro [@bingham2019pyro], and the PocoMC package [@Karamanis2022] implements a version of sequential Monte Carlo (SMC), including NFs.

`flowMC` implements the method proposed by @Gabrie2022. 
As individual chains explore their local neighborhood through gradient-based MCMC steps, multiple chains train the NF to learn the global landscape of the posterior distribution. In turn, the chains can be propagated with a Metropolis-Hastings kernel using the NF to propose globally in the parameter space. The cycle of local sampling, NF tuning, and global sampling is repeated until obtaining chains of the desired length.
The entire algorithm belongs to the class of adaptive MCMC methods [@Andrieu2008], collecting information from the chains' previous steps to simultaneously improve the transition kernel. 
Usual MCMC diagnostics can be applied to assess the robustness of the inference results, thereby avoiding the common concern of validating the NF model. 
If further sampling from the posterior is necessary, the flow trained during a previous run can be reused without further training. 
The mathematical details of the method are explained in [@Gabrie2022; @Gabrie2021a].

***Use of accelerators***
`flowMC` is built on top of JAX, which supports the use of GPU and TPU accelerators by default.
Users can write code the same way as they would on a CPU, and the library will automatically detect the available accelerators and use them at run time.
Furthermore, the library leverages JIT compilations to further improve the performance of the sampler.

***Simplicity and extensibility***
<!-- Since we anticipate most of the users would like to spend most of their time building model instead of optimize the performance of the sampler, -->
We provide a black-box interface with a few tuning parameters for users who intend to use `flowMC` without too much customization on the sampler side.
The only inputs we require from the users are a function to evaluate the logarithm of the (unnormalized) density of the posterior distribution of interest and the initial position of the chains.
On top of the black-box interface, the package offers automatic tuning for the local samplers to reduce the number of hyperparameters users need to manage.

While we provide a high-level interface suitable for most practitioners, the code is also designed to be extensible. Researchers with knowledge of more appropriate local and/or global sampling kernels for their application can integrate the kernels in the sampler module. 

# Acknowledgements
M.G. acknowledges funding from Hi! PARIS.

# References
