# FlowMC

**Normalizing-flow enhanced sampling package for probabilistic inference**

<a href="https://FlowMC.readthedocs.io/en/latest/?">
<img src="https://badgen.net/badge/Read/the doc/blue" alt="doc"/>
</a>
<a href="https://github.com/kazewong/FlowMC/blob/Packaging/LICENSE">
<img src="https://badgen.net/badge/License/MIT/blue" alt="doc"/>
</a>


FlowMC is a Jax-based python package for normalizing-flow enhanced Markov chain Monte Carlo (MCMC) sampling.
The code is open source under MIT license, and it is under active development.

- Just-in-time compilation is supported.
- Native support for GPU acceleration.
- Suit for problems with multi-modality.
- Minimal tuning.

# Installation 

Our package is still in development stage, so it has not reached the official PyPi index yet.
To install our package, run the following command:

```
pip install -i https://test.pypi.org/simple/ FlowMC
```

## Requirements

    * Python 3.8+
    * Jax
    * Jaxlib
    * Flax



# Attribution

A Jax implementation of methods described in: 
> *Efficient Bayesian Sampling Using Normalizing Flows to Assist Markov Chain Monte Carlo Methods* Gabrié M., Rotskoff G. M., Vanden-Eijnden E. - ICML INNF+ workshop 2021 - [pdf](https://openreview.net/pdf?id=mvtooHbjOwx)

> *Adaptive Monte Carlo augmented with normalizing flows.*
Gabrié M., Rotskoff G. M., Vanden-Eijnden E. - PNAS 2022 - [doi](https://www.pnas.org/doi/10.1073/pnas.2109420119), [arxiv](https://arxiv.org/abs/2105.12603)

 
