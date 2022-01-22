import argparse
import h5py
import numpy as np
import pys2let
import pyssht
from scipy import sparse

from pxmcmc import plotting
from pxmcmc import uncertainty
from pxmcmc.transforms import SphericalWaveletTransform
from pxmcmc.measurements import PathIntegral
from pxmcmc.utils import snr, norm


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


MAP_idx = np.where(logpi == max(logpi))
MAP_X = file["chain"][MAP_idx][0]
if setting == "synthesis":
    MAP = wvlttrans.inverse(MAP_X)
    MAP_wvlt = np.copy(MAP_X)
else:
    MAP = np.copy(MAP_X)
    MAP_wvlt = wvlttrans.forward(MAP_X)
MAP = MAP.reshape(mw_shape).astype(float)
maxapost = plotting.plot_map(
    MAP, title="Maximum a posetriori solution", cmap="seismic_r", centre0=True
)
maxapost.savefig(filename("MAP"))

truth = np.load("ekstrom/ekstrom.npy")
diff = truth - MAP
diff_perc = 100 * diff / np.max(abs(truth))
cbar_end = min(max([abs(np.min(diff)), np.max(diff)]), 100)
diffp = plotting.plot_map(
    np.abs(diff), title="|True - MAP|", cmap="plasma", vmin=0, vmax=cbar_end,
)
diffp.savefig(filename("diff"))

map_wvlt = plotting.plot_chain_sample(MAP_wvlt)
map_wvlt.savefig(filename("MAP_wvlt"))


chain_pix = np.zeros(
    (file.attrs["nsamples"] - args.burn, pyssht.sample_length(L, Method="MW"))
)
for i, sample in enumerate(file["chain"][args.burn :]):
    if setting == "synthesis":
        chain_pix[i] = wvlttrans.inverse(sample)
    else:
        chain_pix[i] = np.copy(sample)
ci_range = uncertainty.credible_interval_range(chain_pix).reshape(mw_shape)
ci_map = plotting.plot_map(
    ci_range, title="95% credible interval range", cmap="viridis", vmin=0
)
ci_map.savefig(filename("ci_map"))

wav_ci_ranges = uncertainty.wavelet_credible_interval_range(
    file["chain"][args.burn :], L, B, J_min
)
vmax = 0
for wav_ci_range in wav_ci_ranges:
    vmax = max([vmax, np.max(plotting._oversample(wav_ci_range, 256))])
for i, wav_ci_range in enumerate(wav_ci_ranges):
    title = "95% credible interval range"
    if i == 0:
        title += " Scaling function"
    else:
        title += f" Wavelet scale {i}"
    wav_ci_map = plotting.plot_map(
        wav_ci_range, title=title, cmap="viridis", vmin=0, vmax=vmax
    )
    wav_ci_map.savefig(filename(f"ci_map_scale{i}"))

mean = np.mean(chain_pix, axis=0).reshape(mw_shape)
mean_map = plotting.plot_map(
    mean, title="Mean solution", cmap="seismic_r", centre0=True
)
mean_map.savefig(filename("mean"))

figs = plotting.plot_wavelet_maps(
    mean, L, B, J_min, title="Mean solution", cmap="seismic_r", centre0=True
)
for i, fig in enumerate(figs):
    fig.savefig(filename(f"mean_scale{i}"))

diff_mean = truth - mean

print(f"MAP SNR: {snr(truth, diff):.2f} dB")
print(f"Mean SNR: {snr(truth, diff_mean):.2f} dB")

path_matrix = sparse.load_npz("/home/auggie/GDM/0S254.npz")
pathint = PathIntegral(path_matrix)
data_obs = np.loadtxt("ekstrom/synthetic_GDM40_0S254_noise.txt")[:, 4]
preds = pathint.forward(MAP.flatten())
rel_squared_error = (norm(preds - data_obs) / norm(data_obs)) ** 2
print(f"MAP R2E: {rel_squared_error:.2e}")
preds = pathint.forward(mean.flatten())
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
