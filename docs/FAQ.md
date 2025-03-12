FAQ
===

**My local sampler is not accepting**

This usually means you are setting the step size of the local samplerto be too big.
Try reducing the step size in your local sampler.

Alternative, this could also mean your sampler is proposing in region where the likelihood is ill-defined (i.e. NaN either in likelihood or its derivative if you are using a gradient-based local sampler).
It is worth making sure your likelihood is well-defined within your range of prior.

**In order for my local sampler to accept, I have to choose a very small step size, which makes my chain very correlated.**

This usually indicate some of your parameters are much better measured than others.
Since taking a small step in those directions will already change your likelihood value by a lot, the exploration power of the local sampler in other parameters are limited by those which are well measured.
Currently, we support different step size for different parameters, which you can tune to see whether that improves the situation or not.
If you know the scale of each parameter ahead of time, reparameterizing them to maintain roughly equal scale across parameters also helps.

**My global sample's loss is exploding/not decreasing**

This usually means your learning rate used for training the normalizing flow is too large.
Try reducing the learning rate by a factor of ten.

Another reason for a flat loss is your local sampler is not accepting at all.
This is a bit rarer since this means your data used to train the normalizing flow is just your prior, which the normalizing flow should still be able to learn.

**The sampler is stuck a bit until it starts sampling**

If you use the option ``Jit`` in constructing the local sampler, the code will compile your code to speed up the execution.
The sampler is not really stuck, but it is compiling the code. Depending on how you code up your likelihood function, the compilation can take a while.
If you don't want to wait, you can set ``Jit=False``, which would increase the sampling time.

**The compilation is slow**

If you have a likelihood with many lines, Jax will take a long time to compile the code.
Jax is known to be slow in compilation, especially if your computational graph uses some sort of loop that call a function many times.
While we cannot fundamentally get rid of the problem, [using a jax.lax.scan](https://docs.kidger.site/equinox/tricks/#low-overhead-training-loops) is usually how we deal with it.
