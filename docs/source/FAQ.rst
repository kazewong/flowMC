.. _FAQ-section-top:

FAQ
===

**My local sampler is not accepting**

**My global sample's loss is exploding/not decreasing**

**The sampler is stuck a bit until it starts sampling**

If you turn on 

**The compilation is slow**

If you have a likelihood with many lines, Jax will take a long time to compile the code.
Jax is known to be slow in compilation, especially if your computational graph uses some sort of loop that call a function many times.
While we cannot fundamentally get rid of the problem, here are some tips in alleviating the symptoms:


**Why do you need to call all these making methods instead of using a class?**