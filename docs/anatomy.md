Prior to version 0.4.0, `flowMC` was a package that was designed to execute the algorithm detailed in [this paper](https://arxiv.org/pdf/2105.12603). Since then the community has tried applying `flowMC` to different problems. While there were some successes, there are also limiting factors in terms of performance. One of the biggest issues `flowMC` faced is the fact that the global-local sampling algorithm were baked into the top level `Sampler` API, which means `flowMC` can only use the exact algorithm described in the paper. What if the users want to use a different model? Or run some optimization steps during the sampling stage? Or apply annealing? These are either impossible or not very intuitive in `flowMC` prior to version 0.4.0.

Seeing this limitation, we redesigned the middle level API of `flowMC` while keeping the top level API as similar as possible. This guide aims to describe the different components of `flowMC` and how they interact with each other, and give users who want to extend `flowMC` to optimize for their specific problems a starting point on what could be useful to change. This also acts as a rule of thumb for users who want to use `flowMC` as a black box and interact with internal components through hyperparameters only.

# Target distribution

The target distribution should be defined as a log-probability density function, which follows the following function signature:

```python
def target_log_prob_fn(x: Float[Array, "n_dims"], data: dict[str, Any]) -> Float:
    ...
    return log_prob
```

The `target_log_prob_fn` should take in a `Float[Array, "n_dims"]` array `x` and a dictionary `data` that contains any additional data that the target distribution depends on. The function should return a scalar `Float` that is the log-probability density of the target distribution at `x`.

To ensure the target distribution is well-defined and performant, you should also check whether the function is behaving as expected when `jax.jit` and `jax.grad` are applied to it. 

# Resource-strategy bundle

# Sampler