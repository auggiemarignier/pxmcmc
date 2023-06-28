"""
Plot the results from main.py.
"""

import argparse
import h5py
import numpy as np
import pys2let
import pyssht
import healpy as hp
import warnings

from pxmcmc import plotting
from pxmcmc import uncertainty
from pxmcmc.transforms import SphericalWaveletTransform
from pxmcmc.measurements import WeakLensing
from pxmcmc.utils import snr, norm, build_mask


parser = argparse.ArgumentParser()
parser.add_argument("datafile", type=str)
parser.add_argument("directory", type=str)
parser.add_argument("--suffix", type=str, default="")
parser.add_argument("--burn", type=int, default=1000)
parser.add_argument("--save_npy", action="store_true")
parser.add_argument("--no-mask", action="store_true")
args = parser.parse_args()


def filename(name, ext="png"):
    return f"{args.directory}/{name}{args.suffix}.{ext}"


# Load results file and extract global parameters
file = h5py.File(args.datafile, "r")
params = {attr: file.attrs[attr] for attr in file.attrs.keys()}
L, B, J_min, setting = params["L"], params["B"], params["J_min"], params["setting"]
nscales = pys2let.pys2let_j_max(B, L, J_min) - J_min + 1
wvlttrans = SphericalWaveletTransform(L, B, J_min,)
mw_shape = pyssht.sample_shape(L, Method="MW")
oversample = L < 256  # get smoother images by padding spectrum with zeros

# Plot evolution of posterior, likelihood and prior probabilities
logpi = file["logposterior"][()]
L2s = file["L2s"][()]
L1s = file["priors"][()]
evo = plotting.plot_evolution(logpi, L2s, L1s)
evo.savefig(filename("evolution"))

lmax = L - 1
nside = 3 * lmax - 1
# Get the ground truth for comparison
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    kappa = hp.read_map("takahasi_4096_000_zs16_kappa.fits")
    kappa_bl = hp.alm2map(hp.map2alm(kappa, lmax=lmax), nside=nside)
    sigma = np.radians(50 / 60)  # 50 arcmin
    kappa_s = hp.smoothing(kappa_bl, sigma=sigma)
truth = pyssht.inverse(pys2let.lm_hp2lm(hp.map2alm(kappa_s, lmax), L), L)


# Need 1 mask with the same bandlimit as the results
# and one at the higher bandlimit that is used for plotting
if args.no_mask:
    mask = np.ones(pyssht.sample_shape(L)).astype(bool)
    highL_mask = (
        np.ones(pyssht.sample_shape(256)).astype(bool) if oversample else np.copy(mask)
    )
else:
    mask = build_mask(L, size=20).astype(bool)
    highL_mask = build_mask(256, size=20).astype(bool) if oversample else np.copy(mask)

# Get and plot the MAP estimate
MAP_idx = np.where(logpi == max(logpi))
MAP_X = file["chain"][MAP_idx][0]
if setting == "synthesis":
    MAP = wvlttrans.inverse(MAP_X)
    MAP_wvlt = np.copy(MAP_X)
else:
    MAP = np.copy(MAP_X)
    MAP_wvlt = wvlttrans.forward(MAP_X)
MAP = np.ascontiguousarray(MAP.reshape(mw_shape).real)
maxapost = plotting.plot_map(
    MAP,
    title="Maximum a posetriori solution",
    cmap="cividis",
    centre0=False,
    mask=~highL_mask,
    oversample=oversample,
)
maxapost.savefig(filename("MAP"))

# Plot the difference with the ground truth
diff = np.ascontiguousarray(truth - MAP).real
diff_perc = 100 * diff / np.max(abs(truth))
cbar_end = min(max([abs(np.min(diff * mask)), np.max(diff * mask)]), 100)
diffp = plotting.plot_map(
    np.abs(diff),
    title="|True - MAP|",
    cmap="binary",
    vmin=0,
    vmax=cbar_end,
    mask=~highL_mask,
    oversample=oversample,
)
diffp.savefig(filename("diff"))

map_wvlt = plotting.plot_chain_sample(MAP_wvlt)
map_wvlt.savefig(filename("MAP_wvlt"))

# Get everythin in image space
chain_pix = np.zeros(
    (file.attrs["nsamples"] - args.burn, pyssht.sample_length(L, Method="MW")),
    dtype=complex,
)
for i, sample in enumerate(file["chain"][args.burn :]):
    if setting == "synthesis":
        chain_pix[i] = wvlttrans.inverse(sample)
    else:
        chain_pix[i] = np.copy(sample)

# Get and plot the uncertainty
ci_range = uncertainty.credible_interval_range(chain_pix).reshape(mw_shape)
ci_map = plotting.plot_map(
    np.ascontiguousarray(ci_range.real),
    title="95% credible interval range",
    cmap="viridis",
    vmin=0,
    # vmax=,
    mask=~highL_mask,
    oversample=oversample,
)
ci_map.savefig(filename("ci_map"))

# Get and plot the mean
mean = np.mean(chain_pix, axis=0).reshape(mw_shape)
mean_map = plotting.plot_map(
    np.ascontiguousarray(mean.real),
    title="Mean solution",
    cmap="cividis",
    centre0=False,
    mask=~highL_mask,
    oversample=oversample,
)
mean_map.savefig(filename("mean"))

# Plot the difference
diff_mean = np.ascontiguousarray(truth - mean).real
diff_perc = 100 * diff_mean / np.max(abs(truth))
cbar_end = min(max([abs(np.min(diff_mean * mask)), np.max(diff_mean * mask)]), 100)
diff_meanp = plotting.plot_map(
    np.abs(diff_mean),
    title="|True - mean|",
    cmap="binary",
    vmin=0,
    vmax=cbar_end,
    mask=~highL_mask,
    oversample=oversample,
)
diff_meanp.savefig(filename("diffmean"))

# Get reconstruction quality
if mask is not None:
    mask = mask.astype(bool)
    print(f"MAP SNR: {snr(truth[mask], diff[mask]):.2f} dB")
    print(f"Mean SNR: {snr(truth[mask], diff_mean[mask]):.2f} dB")
else:
    print(f"MAP SNR: {snr(truth, diff):.2f} dB")
    print(f"Mean SNR: {snr(truth, diff_mean):.2f} dB")

# Get predictive error
wl = WeakLensing(L, mask)
data_obs = wl.forward(truth.flatten())
preds = wl.forward(MAP.flatten())
rel_squared_error = (norm(preds - data_obs) / norm(data_obs)) ** 2
print(f"MAP R2E: {rel_squared_error:.2e}")
preds = wl.forward(mean.flatten())
rel_squared_error = (norm(preds - data_obs) / norm(data_obs)) ** 2
print(f"Mean R2E: {rel_squared_error:.2e}")

# Save numpy arrays
if args.save_npy:
    np.save(filename("mean", "npy"), mean)
    np.save(filename("MAP", "npy"), MAP)
    np.save(filename("CI", "npy"), ci_range)
    np.save(filename("diff", "npy"), diff)
    np.save(filename("diff_mean", "npy"), diff_mean)

# Print globals for reference
print(f"Filename: {args.datafile}")
for attr in file.attrs.keys():
    print(f"{attr}: {file.attrs[attr]}")
