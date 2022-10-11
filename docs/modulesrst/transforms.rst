
Transforms
============

The :class:`.Transform` holds all the forward, inverse and adjoint transforms between an image space and some other (e.g. Fourier) space.  Users should use this class to wrap their own transformations to ensure they work with the :class:`forward.ForwardOperator` and :class:`mcmc.PxMCMC` classes.  The :class:`.SphericalWaveletTransform` class is an example.

.. automodule:: transforms
   :members: 
