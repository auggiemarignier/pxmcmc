import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from matplotlib import cm
import copy
import pyssht
from cartopy.crs import Mollweide

from pxmcmc.utils import suppress_stdout


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
