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

# Sampler

On the top level, the `Sampler` class is a thin wrapper on top of the resource-strategy pair (defined below) that provides a couple of extra functionality. The `Sampler` class manages the resources and strategies, as well as run-related parameters such as where would the resources be stored if the user decides to serialize the resources.

The main loop of `Sampler` is pretty straight forward after initialization: Given the available resources, iterate through the list of strategies, which each takes the resources, perform some actions (such as taking local steps or training a normalizing flow), and return the updated resources. In the current implementation, the `Sampler` simply goes through the list of strategy, but in the future we are planning to more flexible main loop such as automatic stopping based on some criteria.

# Resource and Strategy

At the core of the new `flowMC` API are the resource and strategy interfaces. Broadly speaking resources are similar to a data class, and strategies are similar to functions.
**Resources** store some attribute and can be manipulated, but should not have too many methods associated with it. For example, a buffer that stores the sampling results is a resource, a MALA kernel is a resource, and a normalizing flow model is a resource. **Strategies** are functions that take in resources and return updated resources. For example, taking a local step requires two kinds of resources: a proposal distribution and the buffer where the samples are stored. Examples of strategies are taking a local step, training a normalizing flow, and running an optimization step.

The reason for this separation is to allow users to compose different strategies together. For example, the user may want to update the parameters of a proposal kernel like MALA with the local information from a normalizing flow model. Instead of hard coding this functionality to associate with either the MALA kernel or the normalizing flow model, the current API allows the user to define a strategy that takes in both the MALA kernel and the normalizing flow model, and update the MALA kernel with the information from the normalizing flow model. This separate the concern of intermixing different components of the algorithm and make experimenting with new strategies more manageable.

Since this API is designed for users who are willing to look into the guts of `flowMC` and experiment with different strategies, the main question to ask is whether a new data structure/functionality should be a resource or a strategy. While there is no hard rules for such implementation other than conforming to the individual base classes, a good rule of thumb is to ask whether the new data structure/functionality is something that should be updated by other strategies. If the answer is yes, then it should be a resource. If the answer is no, then it should be a strategy.

One extra criteria that decides whether an implementation should be a resource or a strategy is whether the implementation is compatible with `jax`'s transformation. Resource should be compatible with `jit`, and strategy is not required to be compatible with `jit`. An example to illustrate the difference is a training loop contains for-looping over a number of epochs and logging the metadata, which is usually not necessary to be jitted, so this should be a strategy. A neural network and its main functions needs to run efficiently on GPU no matter in sampling or training, so it should be a resource.

# Scope of flowMC

There are a few design choices we made for this version of `flowMC`, which are reflected on the top level API:

1. One should center their choice of resource and strategy around leveraging parallelization. This is reflected by the fact that `n_chains` is a required parameter for the `Sampler` class. The reason for this is `flowMC` is designed to solve problems with complex geometry using adaptive sampling method such as training a normalizing flow alongside with a local proposal. This requires 