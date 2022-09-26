flowMC
=====================================

**Normalizing-flow enhanced sampling package for probabilistic inference**


.. image:: logo_0810.png

.. image:: https://badgen.net/badge/Read/the doc/blue
   :target: https://flowMC.readthedocs.io/en/latest/
.. image:: https://badgen.net/badge/License/MIT/blue
   :target: https//github.com/kazewong/flowMC/blob/Packaging/LICENSE



flowMC is a Jax-based python package for normalizing-flow enhanced Markov chain Monte Carlo (MCMC) sampling.
The code is open source under MIT license, and it is under active development.

- Just-in-time compilation is supported.
- Native support for GPU acceleration.
- Suit for problems with multi-modality.


Five steps to use flowMC's guide
================================

#. You will find basic information such as installation and a quick guide in the :ref:`quickstart-section-top`.
#. We give more information about tuning parameters of our sampler in the :ref:`configuration-section-top`.
#. In :ref:`analysis-section-top`, we give some examples of common diagnostics one can use in understanding the performance of our sampler.
#. We list some examples in :ref:`example-section-top` so users can see whether there is a similar use case they can adopt their code quickly.
#. Finally, the auto-generated API doc is available in the :ref:`api-section-top`.

User guide
----------

.. toctree::
   :maxdepth: 2
   :caption: Contents
   
   quickstart
   configuration
   analysis
   examples
   FAQ
   api/modules
   
