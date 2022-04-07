import argparse
import h5py
import numpy as np
import pys2let
import pyssht
import random
import healpy as hp
from astropy.coordinates import SkyCoord

from pxmcmc import plotting
from pxmcmc import uncertainty
from pxmcmc.transforms import SphericalWaveletTransform
from pxmcmc.measurements import WeakLensing
from pxmcmc.utils import snr, norm


def build_mask(L):
    """"
    Builds a mask for the galactic plane and ecliptic
    0 at positions to be masked
    i.e. to apply mask do map * mask

    Mask in MW format
    """
    mask = np.ones(pyssht.sample_shape(L))
    thetas, phis = pyssht.sample_positions(L)
    for i, t in enumerate(thetas):
        for j, p in enumerate(phis):
            if np.abs(90 - np.degrees(t)) < 20:
                mask[i, j] = 0

    thetaarray, phiarray = pyssht.sample_positions(L, Grid=True)
    thetaarray = np.degrees(thetaarray) - 90
    phiarray = np.degrees(phiarray) - 180

    c = SkyCoord(phiarray, thetaarray, unit="deg")
    d = c.transform_to("galactic")
    degm = np.abs(d.b.degree)
    for i in range(L):
        for j in range(2 * L - 1):
            if degm[i, j] < 20:
                mask[i, j] = 0

    return mask


parser = argparse.ArgumentParser()
parser.add_argument("datafile", type=str)
parser.add_argument("directory", type=str)
parser.add_argument("--suffix", type=str, default="")
parser.add_argument("--burn", type=int, default=1000)
parser.add_argument("--save_npy", action="store_true")
args = parser.parse_args()


def filename(name, ext="png"):
    return f"{args.directory}/{name}{args.suffix}.{ext}"


file = h5py.File(args.datafile, "r")
params = {attr: file.attrs[attr] for attr in file.attrs.keys()}
L, B, J_min, setting = params["L"], params["B"], params["J_min"], params["setting"]
nscales = pys2let.pys2let_j_max(B, L, J_min) - J_min + 1
wvlttrans = SphericalWaveletTransform(L, B, J_min,)
mw_shape = pyssht.sample_shape(L, Method="MW")

logpi = file["logposterior"][()]
L2s = file["L2s"][()]
L1s = file["priors"][()]
evo = plotting.plot_evolution(logpi, L2s, L1s)
evo.savefig(filename("evolution"))

kappa = hp.read_map("takahasi_4096_000_zs16_kappa.fits")
lmax = L - 1
nside = 3 * lmax - 1
kappa_bl = hp.alm2map(hp.map2alm(kappa, lmax=lmax), nside=nside)
sigma = np.radians(50 / 60)  # 50 arcmin
kappa_s = hp.smoothing(kappa_bl, sigma=sigma)
truth = pyssht.inverse(pys2let.lm_hp2lm(hp.map2alm(kappa_s, lmax), L), L)

mask = build_mask(L).astype(bool)
wl = WeakLensing(L, mask)
highL_mask = ~build_mask(256).astype(bool)

MAP_idx = np.where(logpi == max(logpi))
MAP_X = file["chain"][MAP_idx][0]
if setting == "synthesis":
    MAP = wvlttrans.inverse(MAP_X)
    MAP_wvlt = np.copy(MAP_X)
else:
    MAP = np.copy(MAP_X)
    MAP_wvlt = wvlttrans.forward(MAP_X)
MAP = MAP.reshape(mw_shape)
maxapost = plotting.plot_map(
    MAP,
    title="Maximum a posetriori solution",
    cmap="cividis",
    centre0=False,
    mask=mask,
    oversample=False
)
maxapost.savefig(filename("MAP"))

diff = truth - MAP
diff_perc = 100 * diff / np.max(abs(truth))
cbar_end = min(max([abs(np.min(diff)), np.max(diff)]), 100)
diffp = plotting.plot_map(
    np.abs(diff),
    title="|True - MAP|",
    cmap="binary",
    vmin=0,
    vmax=cbar_end,
    mask=highL_mask,
)
diffp.savefig(filename("diff"))

map_wvlt = plotting.plot_chain_sample(MAP_wvlt)
map_wvlt.savefig(filename("MAP_wvlt"))


chain_pix = np.zeros(
    (file.attrs["nsamples"] - args.burn, pyssht.sample_length(L, Method="MW")),
    dtype=complex,
)
for i, sample in enumerate(file["chain"][args.burn :]):
    if setting == "synthesis":
        chain_pix[i] = wvlttrans.inverse(sample)
    else:
        chain_pix[i] = np.copy(sample)
ci_range = uncertainty.credible_interval_range(chain_pix).reshape(mw_shape)
ci_map = plotting.plot_map(
    ci_range,
    title="95% credible interval range",
    cmap="viridis",
    vmin=0,
    mask=highL_mask,
)
ci_map.savefig(filename("ci_map"))

mean = np.mean(chain_pix, axis=0).reshape(mw_shape)
mean_map = plotting.plot_map(
    mean, title="Mean solution", cmap="cividis", centre0=False, mask=highL_mask,
)
mean_map.savefig(filename("mean"))

diff_mean = truth - mean

mask = mask.astype(bool)
print(f"MAP SNR: {snr(truth[mask], diff[mask]):.2f} dB")
print(f"Mean SNR: {snr(truth[mask], diff_mean[mask]):.2f} dB")

data_obs = wl.forward(truth.flatten())
preds = wl.forward(MAP.flatten())
rel_squared_error = (norm(preds - data_obs) / norm(data_obs)) ** 2
print(f"MAP R2E: {rel_squared_error:.2e}")
preds = wl.forward(mean.flatten())
rel_squared_error = (norm(preds - data_obs) / norm(data_obs)) ** 2
print(f"Mean R2E: {rel_squared_error:.2e}")

if args.save_npy:
    np.save(filename("mean", "npy"), mean)
    np.save(filename("MAP", "npy"), MAP)
    np.save(filename("CI", "npy"), ci_range)
    np.save(filename("diff", "npy"), diff)
    np.save(filename("diff_mean", "npy"), diff_mean)

print(f"Filename: {args.datafile}")
for attr in file.attrs.keys():
    print(f"{attr}: {file.attrs[attr]}")
