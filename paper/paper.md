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
Proximal MCMC leverages the proximity mapping operator, a form of generalised gradient, used in convex optimisation problems to efficiently navigate non-smooth parameter spaces.

# Statement of need

`Gala` is an Astropy-affiliated Python package for galactic dynamics. Python
enables wrapping low-level languages (e.g., C) for speed without losing
flexibility or ease-of-use in the user-interface. The API for `Gala` was
designed to provide a class-based and user-friendly interface to fast (C or
Cython-optimized) implementations of common operations such as gravitational
potential and force evaluation, orbit integration, dynamical transformations,
and chaos indicators for nonlinear dynamics. `Gala` also relies heavily on and
interfaces well with the implementations of physical units and astronomical
coordinate systems in the `Astropy` package [@astropy] (`astropy.units` and
`astropy.coordinates`).

`Gala` was designed to be used by both astronomical researchers and by
students in courses on gravitational dynamics or astronomy. It has already been
used in a number of scientific publications [@Pearson:2017] and has also been
used in graduate courses on Galactic dynamics to, e.g., provide interactive
visualizations of textbook material [@Binney:2008]. The combination of speed,
design, and support for Astropy functionality in `Gala` will enable exciting
scientific explorations of forthcoming data releases from the *Gaia* mission
[@gaia] by students and experts alike.

# Mathematics

Single dollars ($) are required for inline mathematics e.g. $f(x) = e^{\pi/x}$

Double dollars make self-standing equations:

$$\Theta(x) = \left\{\begin{array}{l}
0\textrm{ if } x < 0\cr
1\textrm{ else}
\end{array}\right.$$

You can also use plain \LaTeX for equations
\begin{equation}\label{eq:fourier}
\hat f(\omega) = \int_{-\infty}^{\infty} f(x) e^{i\omega x} dx
\end{equation}
and refer to \autoref{eq:fourier} from text.

# Citations

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

If you want to cite a software repository URL (e.g. something on GitHub without a preferred
citation) then you can do it with the example BibTeX entry below for @fidgit.

For a quick reference, the following citation commands can be used:
- `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"

# Figures

Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.

Figure sizes can be customized by adding an optional second parameter:
![Caption for example figure.](figure.png){ width=20% }

# Acknowledgements

We acknowledge contributions from Brigitta Sipocz, Syrtis Major, and Semyeong
Oh, and support from Kathryn Johnston during the genesis of this project.

# References