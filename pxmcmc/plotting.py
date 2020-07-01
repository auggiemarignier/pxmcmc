import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from scipy.stats import laplace

from pxmcmc.utils import suppress_stdout


def mollview(image, figsize=(10, 8), **kwargs):
    fig = plt.figure(num=50, figsize=figsize)  # this figure number thing is a bit hacky...
    with suppress_stdout():
        hp.mollview(image, fig=50, **kwargs)
        hp.graticule(30)
    return fig


def plot_evolution(logposteriors, L2s, L1s, figsize=(10, 8)):
    fig = plt.figure(figsize=figsize)
    plt.subplot(3, 1, 1)
    plt.plot(-logposteriors)
    plt.yscale("log")
    plt.ylabel("-log(posterior)")

    plt.subplot(3, 1, 2)
    plt.plot(L2s)
    plt.yscale("log")
    plt.ylabel("L2")

    plt.subplot(3, 1, 3)
    plt.plot(L1s)
    plt.yscale("log")
    plt.ylabel("L1")
    return fig


def plot_chain_sample(X, figsize=(10, 8)):
    fig = plt.figure(figsize=figsize)
    plt.subplot(2, 1, 1)
    plt.plot(X.real)
    plt.subplot(2, 1, 2)
    plt.plot(X.imag)
    return fig


def plot_posterior_marginals(
    chain, els, L, B, J_min, xrange=(-0.25, 0.25), figsize=(20, 20)
):
    from pxmcmc.utils import get_parameter_from_chain, wavelet_basis

    basis = wavelet_basis(L, B, J_min)
    nb = basis.shape[1]
    fig = plt.figure(figsize=figsize)
    x = np.linspace(*xrange)
    for i, el in enumerate(els):
        for base in range(nb):
            print(f"\r{i},{base}", end="")
            X = [
                get_parameter_from_chain(chain, L, base, el, em).real
                for em in range(-el, el + 1)
            ]
            ax = fig.add_subplot(nb, len(els), base * len(els) + i + 1)
            ax.hist(X)

            ax1 = ax.twinx()
            ax1.plot(x, laplace.pdf(x), c="red")
            ax1.set_ylim([0, 1])
            ax1.tick_params(axis="y", colors="red")
    return fig


def plot_basis_els(
    chain, L, B, J_min, ylim=None, figsize=(20, 20), realpart=True, inflate_mads=1
):
    from pxmcmc.utils import get_parameter_from_chain, wavelet_basis

    basis = wavelet_basis(L, B, J_min)
    fig = plt.figure(figsize=figsize)
    for b, base in enumerate(basis.T):
        medians = np.zeros(L)
        mads = np.zeros(L)
        for el in range(L):
            try:
                if realpart:
                    X = get_parameter_from_chain(chain, L, b, el, 0).real
                else:
                    X = get_parameter_from_chain(chain, L, b, el, 0).imag

            except AssertionError:
                continue
            median = np.median(X)
            mad = np.mean(np.abs(X - median))
            medians[el] = median
            mads[el] = mad * inflate_mads

        ax = fig.add_subplot(basis.shape[1], 1, b + 1)
        ax.plot(medians.real if realpart else medians.imag, c="blue")
        ax.fill_between(
            np.arange(L), medians + mads, medians - mads, alpha=0.5, color="blue"
        )
        ax.set_ylim(ylim)
        ax.set_xlim([0, L])
        ax.tick_params(axis="y", colors="blue")

        base_l0s = [base[l ** 2 + l].real for l in range(L)]
        base_l0s /= np.max(base_l0s)
        ax1 = ax.twinx()
        ax1.plot(base_l0s, c="red")
        ax1.set_ylim([0, 1])
        ax1.tick_params(axis="y", colors="red")
    return fig


def plot_meds_mads(medians, mads, L, B, J_min, figsize=(10, 8)):
    from pxmcmc.utils import wavelet_basis

    basis = wavelet_basis(L, B, J_min)
    nb = basis.shape[1]

    nparams = len(medians)
    fig = plt.figure(figsize=figsize)
    plt.scatter(np.arange(nparams), medians, s=0.5)
    plt.errorbar(np.arange(nparams), medians, yerr=mads, elinewidth=0.5)
    wvltbndrs = [n * nparams // nb for n in range(1, nb)]
    for bndr in wvltbndrs:
        plt.axvline(bndr, ls="--", c="k", zorder=0, alpha=0.5)
    return fig
