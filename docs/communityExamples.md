# Community examples

The core design philosophy of flowMC is to stay lean and simple, so we decided to leave most of the use case-specific optimization to the users. Hence, we keep most of the flowMC internals away from most of the users,
so the only interfaces with flowMC are really defining your likelihood and tuning the sampler parameters exposed on the top level.
That said, it would be useful to have references to see how to use/tune flowMC for different use cases, therefore in this page we host a number of community examples that are contributed by the users.
If you find flowMC useful, please consider contributing your example to this page. This will help other users (and perhaps your future students) to get started quickly.

## Examples

- [jim - A JAX-based gravitational-wave inference toolkit](https://github.com/kazewong/jim)
- [Bayeux - Stitching together models and samplers](https://github.com/jax-ml/bayeux)
  - [Colab example](https://colab.research.google.com/drive/1-PhneVVik5GUq6w2HlKOsqvus13ZLaBH?usp=sharing)
- [Markovian Flow Matching: Accelerating MCMC with Continuous Normalizing Flows](https://arxiv.org/pdf/2405.14392)