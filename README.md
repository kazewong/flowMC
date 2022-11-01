# FlowMC

**Normalizing-flow enhanced sampling package for probabilistic inference**

<a href="https://flowmc.readthedocs.io/en/latest/">
<img src="https://badgen.net/badge/Read/the doc/blue" alt="doc"/>
</a>
<a href="https://github.com/kazewong/FlowMC/blob/Packaging/LICENSE">
<img src="https://badgen.net/badge/License/MIT/blue" alt="doc"/>
</a>

![flowMC_logo](./docs//source/logo_0810.png)

FlowMC is a Jax-based python package for normalizing-flow enhanced Markov chain Monte Carlo (MCMC) sampling.
The code is open source under MIT license, and it is under active development.

- Just-in-time compilation is supported.
- Native support for GPU acceleration.
- Suit for problems with multi-modality.
- Minimal tuning.

# Installation 

The simplest way to install the package is to do it through pip

```
pip install flowMC
```

This will install the latest stable release and its dependencies.
flowMC is based on [Jax](https://github.com/google/jax) and [Flax](https://github.com/google/flax).
By default, installing flowMC will automatically install Jax and Flax available on [PyPI](https://pypi.org).
Jax does not install GPU support by default.
If you want to use GPU with Jax, you need to install Jax with GPU support according to their document.
At the time of writing this documentation page, this is the command to install Jax with GPU support:

```
pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

If you want to install the latest version of flowMC, you can clone this repo and install it locally:

```
git clone https://github.com/kazewong/flowMC.git
cd flowMC
pip install -e .
```

## Requirements

Here is a list of packages we use in the main library

    * Python 3.8+
    * Jax
    * Jaxlib
    * Flax

To visualize the inference results in the examples, we requrie the following packages in addtion to the above:

    * matplotlib
    * corner
    * arviz



# Attribution

A Jax implementation of methods described in: 
> *Efficient Bayesian Sampling Using Normalizing Flows to Assist Markov Chain Monte Carlo Methods* Gabrié M., Rotskoff G. M., Vanden-Eijnden E. - ICML INNF+ workshop 2021 - [pdf](https://openreview.net/pdf?id=mvtooHbjOwx)

> *Adaptive Monte Carlo augmented with normalizing flows.*
Gabrié M., Rotskoff G. M., Vanden-Eijnden E. - PNAS 2022 - [doi](https://www.pnas.org/doi/10.1073/pnas.2109420119), [arxiv](https://arxiv.org/abs/2105.12603)

 
