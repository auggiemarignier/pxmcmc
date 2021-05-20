.. pxmcmc documentation master file, created by
   sphinx-quickstart on Thu May 20 10:32:37 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

PXMCMC
==================================

A python package for performing proximal Markov Chain Monte Carlo.  This package contains the MCMC methods and base classes for building different forward operators and priors as needed, as well as routines for calculating simple uncertainties based on the MCMC chains.  Example scripts are also provided.

Installation
============

Installation is currently managed by `poetry <https://python-poetry.org/>`_ to handle dependencies when installing from source

.. code-block:: bash

    $ git clone https://github.com/auggiemarignier/pxmcmc.git
    $ cd pxmcmc
    $ poetry install
    $ source <venv>/bin/install

where :code:`<venv>` will depend on your :code:`poetry` configuration.

.. todo::

   Will be made available on PyPI to be pip installable.


.. toctree::
   :maxdepth: 2
   :caption: Modules:

   modulesrst/mcmc
   modulesrst/forward
   modulesrst/measurements
   modulesrst/transforms
   modulesrst/prior
   modulesrst/uncertainty
   modulesrst/saving
   modulesrst/plotting
   modulesrst/utils
   

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
