# flowMC

**Normalizing-flow enhanced sampling package for probabilistic inference**

<a href="https://flowmc.readthedocs.io/en/main/">
<img src="https://readthedocs.org/projects/flowmc/badge/?version=main&style=flat-square" alt="doc"/>
</a>
<a href="https://github.com/kazewong/flowMC/blob/Packaging/LICENSE">
<img src="https://img.shields.io/badge/License-MIT-blue?style=flat-square" alt="doc"/>
</a>
<a href='https://coveralls.io/github/kazewong/flowMC?branch=main'><img src='https://img.shields.io/coverallsCoverage/github/kazewong/flowMC?style=flat-square' alt='Coverage Status' /></a>

> [!WARNING]
> Note that `flowMC` has not reached v1.0.0, meaning the API could subject to changes. In general, the higher level the API, the less likely it is going to change. However, intermediate level API such as the resource strategy interface could subject to major revision for performance concerns.

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
By default this install the CPU version of Jax. If you have a GPU and want to use it, you can install the GPU version of Jax by running:

```
pip install flowMC[cuda]
```

If you want to install the latest version of flowMC, you can clone this repo and install it locally:

```
git clone https://github.com/kazewong/flowMC.git
cd flowMC
pip install -e .
```

There are a couple more extras that you can install with flowMC, including:
- `flowMC[docs]`: Install the documentation dependencies.
- `flowMC[codeqa]`: Install the code quality dependencies.
- `flowMC[visualize]`: Install the visualization dependencies.

On top of `pip` installation, we highly encourage you to use [uv](https://docs.astral.sh/uv/) to manage your python environment. Once you clone the repo, you can run `uv sync` to create a virtual environment with all the dependencies installed.
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

## Contributors

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="http://secondearths.sakura.ne.jp/en/index.html"><img src="https://avatars.githubusercontent.com/u/15956904?v=4?s=100" width="100px;" alt="Hajime Kawahara"/><br /><sub><b>Hajime Kawahara</b></sub></a><br /><a href="https://github.com/kazewong/flowMC/issues?q=author%3AHajimeKawahara" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/daniel-dodd"><img src="https://avatars.githubusercontent.com/u/68821880?v=4?s=100" width="100px;" alt="Daniel Dodd"/><br /><sub><b>Daniel Dodd</b></sub></a><br /><a href="https://github.com/kazewong/flowMC/commits?author=daniel-dodd" title="Documentation">📖</a> <a href="https://github.com/kazewong/flowMC/pulls?q=is%3Apr+reviewed-by%3Adaniel-dodd" title="Reviewed Pull Requests">👀</a> <a href="https://github.com/kazewong/flowMC/commits?author=daniel-dodd" title="Tests">⚠️</a> <a href="https://github.com/kazewong/flowMC/issues?q=author%3Adaniel-dodd" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="http://matt-graham.github.io"><img src="https://avatars.githubusercontent.com/u/6746980?v=4?s=100" width="100px;" alt="Matt Graham"/><br /><sub><b>Matt Graham</b></sub></a><br /><a href="https://github.com/kazewong/flowMC/issues?q=author%3Amatt-graham" title="Bug reports">🐛</a> <a href="https://github.com/kazewong/flowMC/commits?author=matt-graham" title="Tests">⚠️</a> <a href="https://github.com/kazewong/flowMC/pulls?q=is%3Apr+reviewed-by%3Amatt-graham" title="Reviewed Pull Requests">👀</a> <a href="https://github.com/kazewong/flowMC/commits?author=matt-graham" title="Documentation">📖</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://www.kaze-wong.com/"><img src="https://avatars.githubusercontent.com/u/8803931?v=4?s=100" width="100px;" alt="Kaze Wong"/><br /><sub><b>Kaze Wong</b></sub></a><br /><a href="https://github.com/kazewong/flowMC/issues?q=author%3Akazewong" title="Bug reports">🐛</a> <a href="#blog-kazewong" title="Blogposts">📝</a> <a href="https://github.com/kazewong/flowMC/commits?author=kazewong" title="Code">💻</a> <a href="#content-kazewong" title="Content">🖋</a> <a href="https://github.com/kazewong/flowMC/commits?author=kazewong" title="Documentation">📖</a> <a href="#example-kazewong" title="Examples">💡</a> <a href="#infra-kazewong" title="Infrastructure (Hosting, Build-Tools, etc)">🚇</a> <a href="#maintenance-kazewong" title="Maintenance">🚧</a> <a href="#research-kazewong" title="Research">🔬</a> <a href="https://github.com/kazewong/flowMC/pulls?q=is%3Apr+reviewed-by%3Akazewong" title="Reviewed Pull Requests">👀</a> <a href="https://github.com/kazewong/flowMC/commits?author=kazewong" title="Tests">⚠️</a> <a href="#tutorial-kazewong" title="Tutorials">✅</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://marylou-gabrie.github.io/"><img src="https://avatars.githubusercontent.com/u/11092071?v=4?s=100" width="100px;" alt="Marylou Gabrié"/><br /><sub><b>Marylou Gabrié</b></sub></a><br /><a href="https://github.com/kazewong/flowMC/issues?q=author%3Amarylou-gabrie" title="Bug reports">🐛</a> <a href="https://github.com/kazewong/flowMC/commits?author=marylou-gabrie" title="Code">💻</a> <a href="#content-marylou-gabrie" title="Content">🖋</a> <a href="https://github.com/kazewong/flowMC/commits?author=marylou-gabrie" title="Documentation">📖</a> <a href="#example-marylou-gabrie" title="Examples">💡</a> <a href="#maintenance-marylou-gabrie" title="Maintenance">🚧</a> <a href="#research-marylou-gabrie" title="Research">🔬</a> <a href="https://github.com/kazewong/flowMC/commits?author=marylou-gabrie" title="Tests">⚠️</a> <a href="#tutorial-marylou-gabrie" title="Tutorials">✅</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/Qazalbash"><img src="https://avatars.githubusercontent.com/u/62182585?v=4?s=100" width="100px;" alt="Meesum Qazalbash"/><br /><sub><b>Meesum Qazalbash</b></sub></a><br /><a href="https://github.com/kazewong/flowMC/commits?author=Qazalbash" title="Code">💻</a> <a href="#maintenance-Qazalbash" title="Maintenance">🚧</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/thomasckng"><img src="https://avatars.githubusercontent.com/u/97585527?v=4?s=100" width="100px;" alt="Thomas Ng"/><br /><sub><b>Thomas Ng</b></sub></a><br /><a href="https://github.com/kazewong/flowMC/commits?author=thomasckng" title="Code">💻</a> <a href="#maintenance-thomasckng" title="Maintenance">🚧</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/tedwards2412"><img src="https://avatars.githubusercontent.com/u/6105841?v=4?s=100" width="100px;" alt="Thomas Edwards"/><br /><sub><b>Thomas Edwards</b></sub></a><br /><a href="https://github.com/kazewong/flowMC/issues?q=author%3Atedwards2412" title="Bug reports">🐛</a> <a href="https://github.com/kazewong/flowMC/commits?author=tedwards2412" title="Code">💻</a></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

 
