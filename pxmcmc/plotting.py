import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from matplotlib import cm
import copy
import pyssht
import pys2let

try:
    from cartopy.crs import Mollweide
except ModuleNotFoundError:
    print("cartopy not found.  Cannot plot coasts.")

from pxmcmc.utils import suppress_stdout, _multires_bandlimits


def plot_map(
    f,
    title=None,
    cbar=True,
    cmap="turbo",
    vmin=None,
    vmax=None,
    cbar_label="",
    oversample=True,
    centre0=False,
    coasts=False,
):
    """
    Plots a single `MW <https://arxiv.org/abs/1110.6298>`_ sampled spherical map.

    :param array f: MW sampled image.  Shape :math:`(L, 2L - 1)`.
    :param string title: Figure title
    :param bool cbar: if :code:`True`, plot the colour bar
    :param string cmap: Name of a `matplotlib colour map <https://matplotlib.org/stable/tutorials/colors/colormaps.html>`_
    :param float vmin: Minimum value.  If :code:`None`, will default to lowest value in :code:`f`.
    :param float vmax: Maximum value.  If :code:`None`, will default to highest value in :code:`f`.
    :param string cbar_label: Label for the colour bar. Requires :code:`cbar=True`
    :param bool oversample: if :code:`True`, oversamples :code:`f` to bandlimit :math:`L=256` so the image is not pixelated
    :param bool centre0: if :code:`True`, forces the colour map to be centred at 0.  Overrides :code:`vmin,vmax`.
    :param bool coasts: if :code:`True`, plots coastlines.

    :return: matplotlib figure
    """
    cmap = copy.copy(cm.get_cmap(cmap))
    cmap.set_bad(alpha=0)

    if oversample:
        L = 256
        f = _oversample(f, L)
    else:
        L = f.shape[0]

    if centre0:
        cbar_end = max([f.max(), abs(f.min())])
        vmax = cbar_end
        vmin = -cbar_end

    f_plt, _ = pyssht.mollweide_projection(f, L)
    fig = plt.figure(figsize=(20, 10))
    if not cbar:
        map_gs = fig.add_gridspec(1, 1)
        map_ax = map_gs.subplots()
        im = map_ax.imshow(f_plt, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    else:
        map_gs = fig.add_gridspec(nrows=1, ncols=2, width_ratios=[40, 1], wspace=0.05)
        map_ax = fig.add_subplot(map_gs[:, :-1])
        cbar_ax = fig.add_subplot(map_gs[:, -1])
        im = map_ax.imshow(f_plt, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label(cbar_label, fontsize=24)
        cbar.ax.tick_params(labelsize="xx-large")
    map_ax.axis("off")
    map_ax.set_title(title, fontsize=24)
    if coasts:
        coast_gs = map_gs[0].subgridspec(1, 1)
        coast_ax = coast_gs.subplots(subplot_kw={"projection": Mollweide()})
        coast_ax.coastlines(linewidth=2)
        coast_ax.patch.set_alpha(0)
    return fig


def plot_wavelet_maps(f, L, B, J_min, dirs=1, spin=0, same_scale=True, **map_args):
    """
    Plots the scaling and wavelet maps of spherical map :code:`f`.  
    
    :param array f: MW sampled image.  Shape :math:`(L, 2L - 1)`.
    :param int L: angular bandlimit
    :param float B: wavelet scale parameter
    :param int J_min: minimum wavelet scale
    :param int dirs: azimuthal bandlimit for directional wavelets
    :param int spin: spin number of spherical signal
    :param bool same_scale: if :code:`True`, wavelet maps are plotted on same colour scale
    :param \**map_args: optional arguments for :meth:`plot_map`

    :return: List of figures
    """
    bls = _multires_bandlimits(L, B, J_min, dirs, spin)
    f_wav, f_scal = pys2let.analysis_px2wav(
        f.flatten().astype(complex), B, L, J_min, dirs, spin, upsample=0
    )
    figs = []
    if "title" in map_args:
        base_title = map_args["title"]
    else:
        base_title = ""
    map_args["title"] = f"{base_title} Scaling function"
    figs.append(plot_map(f_scal.reshape(pyssht.sample_shape(bls[0])), **map_args))

    if same_scale:
        map_args["vmax"] = np.max(f_wav).real

    scale_start = 0
    for i, bl in enumerate(bls[1:], 1):
        scale_length = pyssht.sample_length(bl)
        wav = f_wav[scale_start : scale_start + scale_length]
        map_args["title"] = f"{base_title} Wavelet scale {i}"
        figs.append(plot_map(wav.reshape(pyssht.sample_shape(bl)), **map_args))
        scale_start += scale_length

    return figs


def mollview(image, figsize=(10, 8), **kwargs):
    i = np.random.randint(1000)
    fig = plt.figure(
        num=i, figsize=figsize
    )  # this figure number thing is a bit hacky...
    with suppress_stdout():
        hp.mollview(image, fig=i, **kwargs)
        hp.graticule(30)
    return fig


def plot_evolution(logposteriors, L2s, L1s, figsize=(10, 8)):
    """
    Plot the evolution of the MCMC chain.

    :param logposteriors: array of log posterior probabilities of the saved MCMC samples. Plot shows the negative log posterior.
    :param L2s: array of log gaussian data fidelities (L2 error norms)
    :param L1s: array of log Laplacian priors (L1 norms)
    :param tuple figsize: Figure size

    :return: matplotlib figure
    """
    MAP_idx = np.where(logposteriors == max(logposteriors))
    fig = plt.figure(figsize=figsize)
    plt.subplot(3, 1, 1)
    plt.plot(-logposteriors)
    plt.axvline(MAP_idx, linestyle="--", c="r")
    plt.yscale("log")
    plt.ylabel("-log(posterior)")

    plt.subplot(3, 1, 2)
    plt.plot(L2s)
    plt.axvline(MAP_idx, linestyle="--", c="r")
    plt.yscale("log")
    plt.ylabel("L2")

    plt.subplot(3, 1, 3)
    plt.plot(L1s)
    plt.axvline(MAP_idx, linestyle="--", c="r")
    plt.yscale("log")
    plt.ylabel("L1")
    return fig


def plot_chain_sample(X, figsize=(10, 8)):
    """
    Plots the real and imaginary parts of an MCMC sample

    :param X: MCMC sample
    :param tuple figsize: Figure size

    :return: matplotlib figure
    """
    fig = plt.figure(figsize=figsize)
    plt.subplot(2, 1, 1)
    plt.plot(X.real)
    plt.subplot(2, 1, 2)
    plt.plot(X.imag)
    return fig


def _oversample(f, L):
    flm = pyssht.forward(f, f.shape[0], Reality=True)
    z = np.zeros(L ** 2 - f.shape[0] ** 2)
    flm = np.concatenate((flm, z))
    return pyssht.inverse(flm, L, Reality=True)
