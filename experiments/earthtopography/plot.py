"""
Plots the summary maps from the MCMC in main.py
"""

import argparse
import h5py
import numpy as np
import s2wav
import s2fft
from s2fft import sampling
from s2wav.utils.shapes import j_max
import healpy as hp

from pxmcmc import plotting
from pxmcmc import uncertainty
from pxmcmc.transforms import SphericalWaveletTransform
from pxmcmc.utils import map2alm, snr


parser = argparse.ArgumentParser()
parser.add_argument(
    "datafile", type=str, help="Path to .hdf5 file with pxmcmc results."
)
parser.add_argument("directory", type=str, help="Directory in which to save plots.")
parser.add_argument(
    "--suffix", type=str, default="", help="Optional suffix to output filenames."
)
parser.add_argument(
    "--burn",
    type=int,
    default=0,
    help="Ignore the first <burn> MCMC samples.  Default 100.",
)
parser.add_argument(
    "--save_npy",
    action="store_true",
    help="Also save the output summary maps as .npy files",
)
args = parser.parse_args()


def filename(name):
    return f"{args.directory}/{name}{args.suffix}.png"


# Load results file and extract global parameters
file = h5py.File(args.datafile, "r")
params = {attr: file.attrs[attr] for attr in file.attrs.keys()}
L, B, J_min = params["L"], params["B"], params["J_min"]
try:
    setting = params["setting"]
except KeyError:
    setting = input("Specify setting:\t")
nscales = j_max(B, L, J_min) - J_min + 1

# MCMC samples need to be transformed from wavelet to image space
# so create helper transform instance
wvlttrans = SphericalWaveletTransform(L, B, J_min)
mw_shape = sampling.f_shape(L)

# Plot the evolution of the posterior, likelihood and prior values
# throughout the MCMC chain
logpi = file["logposterior"][()]
L2s = file["L2s"][()]
L1s = file["priors"][()]
evo = plotting.plot_evolution(logpi, L2s, L1s)
evo.savefig(filename("evolution"))

# Load original data, which in this case is also the ground truth
# we are aiming to recover
topo = hp.read_map("ETOPO1_Ice_hpx_256.fits", verbose=False, dtype=float,)
truth = pyssht.inverse(sampling.lm_hp_to_2d(map2alm(topo, L - 1), L).flatten(), L, Reality=True) / 1000
truthp = plotting.plot_map(truth, title="Truth")
truthp.savefig(filename("truth"))

# Get MAP estimate (i.e. highest posterior sample)
MAP_idx = np.where(logpi == max(logpi))
MAP_X = file["chain"][MAP_idx][0]
if setting == "synthesis":
    MAP = wvlttrans.inverse(MAP_X)
    MAP_wvlt = np.copy(MAP_X)
else:
    MAP = np.copy(MAP_X)
    MAP_wvlt = wvlttrans.forward(MAP_X)
MAP = np.ascontiguousarray(MAP.reshape(mw_shape).real)
maxapost = plotting.plot_map(MAP, title="Maximum a posetriori solution")
maxapost.savefig(filename("MAP"))

# Plot the difference with the truth
diff = truth - MAP
cbar_end = max([abs(np.min(diff)), np.max(diff)])
diffp = plotting.plot_map(
    diff,
    title="True - MAP",
    cmap="PuOr",
    vmin=-cbar_end,
    vmax=cbar_end,
)
diffp.savefig(filename("diff"))

# Plot the values of the individual sampled parameters
map_wvlt = plotting.plot_chain_sample(MAP_wvlt)
map_wvlt.savefig(filename("MAP_wvlt"))

# Make sure all the MCMC samples are in image space.
# Drop the first args.burn samples.
chain_pix = np.zeros(
    (file.attrs["nsamples"] - args.burn, mw_sample_length(L))
)
for i, sample in enumerate(file["chain"][args.burn :]):
    if setting == "synthesis":  # sampled wavelet coeffecients so need to transform
        chain_pix[i] = wvlttrans.inverse(sample)
    else:  # sampled image space pixel values so no need to transform
        chain_pix[i] = np.copy(sample)

# Calculate and plot the credible interval range
ci_range = uncertainty.credible_interval_range(chain_pix).reshape(mw_shape)
ci_map = plotting.plot_map(
    ci_range, title="95% credible interval range", cmap="viridis", vmin=0
)
ci_map.savefig(filename("ci_map"))

# Calculate and plot the mean image, and its difference with the truth
mean = np.mean(chain_pix, axis=0).reshape(mw_shape)
mean_map = plotting.plot_map(mean, title="Mean solution")
mean_map.savefig(filename("mean"))

diff_mean = truth - mean
cbar_end = max([abs(np.min(diff_mean)), np.max(diff_mean)])
diffpmean = plotting.plot_map(
    diff_mean,
    title="True - mean",
    cmap="PuOr",
    vmin=-cbar_end,
    vmax=cbar_end,
)
diffpmean.savefig(filename("diff_mean"))

# If noise was added, plot it
if "noise" in params and params["noise"]:
    noise = params["noise"].reshape((L, 2 * L - 1)) / 1000
    noise_map = plotting.plot_map(
        noise, title="Added noise", cmap="binary", oversample=False
    )
    noise_map.savefig(filename("noise"))
    print(f"Input SNR: {snr(truth, noise):.2f} dB")

# Calculate the signal-to-noise ratio to evaluate the
# quality of the recovery
print(f"Mean SNR: {snr(truth, diff_mean):.2f} dB")
print(f"MAP SNR: {snr(truth, diff):.2f} dB")

# Save numpy arrays
if args.save_npy:
    np.save(filename("mean", "npy"), mean)
    np.save(filename("MAP", "npy"), MAP)
    np.save(filename("CI", "npy"), ci_range)
    np.save(filename("diff", "npy"), diff)
    np.save(filename("diff_mean", "npy"), diff_mean)

# Print summary of global parameters used for this inversion
print(f"Filename: {args.datafile}")
for attr in file.attrs.keys():
    print(f"{attr}: {file.attrs[attr]}")
