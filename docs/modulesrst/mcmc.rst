MCMC
====

This module implements various proximal MCMC algorithms.  The base class :class:`.PxMCMC` implements some basic (mostly private) functions that should be common to all algorithms.  When creating a new algorithm, a class should inherit the :class:`.PxMCMC` class and implement at least the :meth:`run` method.

.. automodule:: mcmc
   :members: 
