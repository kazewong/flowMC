# flowMC

**Normalizing-flow enhanced sampling package for probabilistic inference**

<a href="https://flowmc.readthedocs.io/en/main/">
<img src="https://readthedocs.org/projects/flowmc/badge/?version=main&style=flat-square" alt="doc"/>
</a>
<a href="https://github.com/kazewong/flowMC/blob/Packaging/LICENSE">
<img src="https://img.shields.io/badge/License-MIT-blue?style=flat-square" alt="doc"/>
</a>
<a href='https://coveralls.io/github/kazewong/flowMC?branch=main'><img src='https://img.shields.io/coverallsCoverage/github/kazewong/flowMC?style=flat-square' alt='Coverage Status' /></a>

![flowMC_logo](./docs/logo_0810.png)

flowMC is a Jax-based python package for normalizing-flow enhanced Markov chain Monte Carlo (MCMC) sampling.
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
flowMC is based on [Jax](https://github.com/google/jax) and [Equinox](https://github.com/patrick-kidger/equinox).
By default, installing flowMC will automatically install Jax and Equinox available on [PyPI](https://pypi.org).
Jax does not install GPU support by default.
If you want to use GPU with Jax, you need to install Jax with GPU support according to their document.
At the time of writing this documentation page, this is the command to install Jax with GPU support:

```
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

If you want to install the latest version of flowMC, you can clone this repo and install it locally:

```
git clone https://github.com/kazewong/flowMC.git
cd flowMC
pip install -e .
```

## Requirements

Here is a list of packages we use in the main library

    * Python 3.9+
    * Jax
    * Jaxlib
    * equinox

To visualize the inference results in the examples, we requrie the following packages in addtion to the above:

    * matplotlib
    * corner
    * arviz

The test suite is based on pytest. To run the tests, one needs to install `pytest` and run `pytest` at the root directory of this repo.

# Attribution

If you used `flowMC` in your research, we would really appericiate it if you could at least cite the following papers:

```
@article{Wong:2022xvh,
    author = "Wong, Kaze W. k. and Gabri\'e, Marylou and Foreman-Mackey, Daniel",
    title = "{flowMC: Normalizing flow enhanced sampling package for probabilistic inference in JAX}",
    eprint = "2211.06397",
    archivePrefix = "arXiv",
    primaryClass = "astro-ph.IM",
    doi = "10.21105/joss.05021",
    journal = "J. Open Source Softw.",
    volume = "8",
    number = "83",
    pages = "5021",
    year = "2023"
}

@article{Gabrie:2021tlu,
    author = "Gabri\'e, Marylou and Rotskoff, Grant M. and Vanden-Eijnden, Eric",
    title = "{Adaptive Monte Carlo augmented with normalizing flows}",
    eprint = "2105.12603",
    archivePrefix = "arXiv",
    primaryClass = "physics.data-an",
    doi = "10.1073/pnas.2109420119",
    journal = "Proc. Nat. Acad. Sci.",
    volume = "119",
    number = "10",
    pages = "e2109420119",
    year = "2022"
}
```

This will help `flowMC` getting more recognition, and the main benefit *for you* is this means the `flowMC` community will grow and it will be continuously improved. If you believe in the magic of open-source software, please support us by attributing our software in your work.


`flowMC` is a Jax implementation of methods described in: 
> *Efficient Bayesian Sampling Using Normalizing Flows to Assist Markov Chain Monte Carlo Methods* Gabrié M., Rotskoff G. M., Vanden-Eijnden E. - ICML INNF+ workshop 2021 - [pdf](https://openreview.net/pdf?id=mvtooHbjOwx)

> *Adaptive Monte Carlo augmented with normalizing flows.*
Gabrié M., Rotskoff G. M., Vanden-Eijnden E. - PNAS 2022 - [doi](https://www.pnas.org/doi/10.1073/pnas.2109420119), [arxiv](https://arxiv.org/abs/2105.12603)

 
