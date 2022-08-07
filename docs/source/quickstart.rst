.. _quickstart-section-top:

Quick Start
============

Installation
------------

The recommended way to install FlowMC is using pip

.. code-block::

    pip install flowMC

This will install the latest stable release and its dependencies.
FlowMC is based on `Jax <https://github.com/google/jax>`_ and `Flax <https://github.com/google/flax>`_.
By default, installing flowMC will automatically install Jax and Flax available on `PyPI <https://pypi.org/>`_.
Jax does not install GPU support by default.
If you want to use GPU with Jax, you need to install Jax with GPU support according to `their document <pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html>`_.
At the time of writing this documentation page, this is the command to install Jax with GPU support:

.. code-block::

    pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html


Basic Usage
-----------

