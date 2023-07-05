.. pxmcmc documentation master file, created by
   sphinx-quickstart on Thu May 20 10:32:37 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

PXMCMC
==================================

High-dimensional imaging inverse problems arise in many fields, including astrophysics, geophysics and medical imaging.
They involve recovering the pixels of an image of, for example, the inside of a human body from attenuated X-rays.
Proximal Markov Chain Monte Carlo algorithms can be used for sampling high-dimensional parameter spaces where the posterior distribution is non-differentiable, for example when using a sparse prior.
The `proximity operator <https://en.wikipedia.org/wiki/Proximal_operator>`_ is used instead of the gradient to efficiently navigate the parameter space.
The algorithms implemented here were first introduced in `Pereyra (2016) <https://link.springer.com/article/10.1007/s11222-015-9567-4>`_, modifying the gradient-based Langevin MCMC.

This is a python package for performing proximal MCMC.
It contains the MCMC methods and base classes for building different forward operators and priors as needed, as well as routines for calculating simple uncertainties based on the MCMC chains.
Example scripts are also provided. 

Installation
============

Available on PyPI

.. code-block:: bash

    $ pip install pxmcmc

Installation is currently managed by `poetry <https://python-poetry.org/>`_ to handle dependencies when installing from source

.. code-block:: bash

    $ git clone https://github.com/auggiemarignier/pxmcmc.git
    $ cd pxmcmc
    $ poetry install
    $ source <venv>/bin/activate

where :code:`<venv>` will depend on your :code:`poetry` configuration.

.. toctree::
   :maxdepth: 1
   :caption: Modules

   modulesrst/mcmc
   modulesrst/forward
   modulesrst/measurements
   modulesrst/transforms
   modulesrst/prior
   modulesrst/uncertainty
   modulesrst/saving
   modulesrst/plotting
   modulesrst/utils

.. toctree::
   :maxdepth: 1
   :caption: Examples

   examplesrst/quickstart
   examplesrst/customops

.. toctree::
   :maxdepth: 1
   :caption: About

   aboutrst/LICENSE
   aboutrst/CITATION
   aboutrst/CONTRIBUTING
   

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
