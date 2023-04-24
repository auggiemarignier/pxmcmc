---
title: 'PxMCMC: A Python package for proximal Markov Chain Monte Carlo'
tags:
  - Python
  - MCMC
  - imaging
  - geophysics
  - astrophysics
authors:
  - name: Augustin Marignier
    orcid: 0000-0001-6778-1399
    corresponding: true
    affiliation: "1, 2, 3" # (Multiple affiliations must be quoted)
affiliations:
 - name: Mullard Space Science Laboratory, University College London, UK
   index: 1
 - name: Department of Earth Sciences, University College London, UK
   index: 2
 - name: Research School of Earth Sciences, Australian National University, Australia
   index: 3
date: 24 April 2023
bibliography: paper.bib
---

# Summary

Markov Chain Monte Carlo (MCMC) methods form the dominant set of algorithms for Bayesian inference.
The appeal of MCMC in the physical sciences is that it produces a set of samples from the posterior distribution of model parameters given the available data, integrating any prior information one may have about the parameters and providing a fully flexible way to quantify uncertainty.
However, it is well known that in high dimensions (many model parameters) standard MCMC struggles to converge to a solution due the exponentially large space to be sampled.
This led to the development of gradient-based MCMC algorithms, which use the gradient of the posterior distribution to efficiently navigate the parameter space.
While this allows MCMC to scale to high dimensions, it restricts the form of the posterior to be continuously differentiable.
Certain forms of prior information used in imaging problems, such as sparsity, use a non-smooth prior distribution, thus gradient-based MCMC cannot be used for these inference problems.
Proximal MCMC leverages the proximity mapping operator `[@Moreau1962]`, a form of generalised gradient, used in convex optimisation problems to efficiently navigate non-smooth parameter spaces.

# Statement of need

High-dimensional imaging inverse problems arise in many fields, including astrophysics, geophysics and medical imaging.
They involve recovering the pixels of an image of, for example, the inside of a human body from attenuated X-rays.
For applications where the data may be incomplete, as is often the case in geophysical and astrophysical imaging, compressed sensing `[@Donoho2006; @Candes2011]` has demonstrated that sparsity in a particular basis (typically wavelets) can be used to accurately recover signals from an underdetermined system..
In a Bayesian setting, sparse priors come in the form of the non-differentiable Laplace distribution, resulting in the need for proximal mappings for optimisation problems `[@Moreau1962; @Parikh2014]`.
The use of proximal operators in MCMC was first proposed by `@Pereyra2016`, modifying the gradient-based Langevin MCMC, and has since been used in astrophysical and geophysical applications (e.g. `@Cai2018; @Price2019; @Marignier2023`).

MCMC methods already have popular implementations.
For example, gradient-based Hamiltonian Monte Carlo is implemented in `STAN` `[@Stan]`, and `emcee` `[@Foreman-Mackey2013]` is a Python implementation of the affine-invariant ensemble sampler MCMC `[@Goodman2010]` popular in the astrophysics community.
To the author's knowledge, however, there exists no Python implementation of proximal MCMC readily available.
`PxMCMC` is a Python package implementing proximal algorithms from `@Pereyra2019` and `@Pereyra2020`.
The class-based API abstracts out the main components of MCMC into interoperable classes, thereby allowing users to implement their own forward models (physical model) and priors, and even their own MCMC sampler if desired.
Originally developed to solve inverse imaging problems defined on spherical domains `[@Marignier2023]`, the package provides priors to promote sparsity in a spherical wavelet domain using transforms from the `S2LET` package `[@Leistedt2013]`.
Examples provided in the package include a common problem in global seismic tomography and a full-sky cosmological mass-mapping problem, the details of which can be found in `@Marignier2023`.

# Acknowledgements

We acknowledge contributions from Brigitta Sipocz, Syrtis Major, and Semyeong
Oh, and support from Kathryn Johnston during the genesis of this project.

# References