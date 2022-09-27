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
#. We give more information about tuning parameters of our sampler in the :ref:`tuning_guide-section-top`.
#. In :ref:`tutorials`, we have a set of more pedagogical notebooks that will give you a better understanding of the package infrastructure.
#. We list some community examples in :ref:`example-section-top`, so users can see whether there is a similar use case they can adopt their code quickly.
#. Finally, we have a list of frequently asked questions in :ref:`FAQ-section-top`.

User guide
----------

.. toctree::
   :maxdepth: 2
   :caption: Contents
   
   quickstart
   configuration
   examples
   FAQ
   api/modules
   
.. toctree::
   :maxdepth: 1
   :caption: Tutorials
   :name: tutorials

   tutorials/analyzingChains
   tutorials/localKernels
   tutorials/normalizingFlow