import argparse
import h5py
import numpy as np
import pys2let
import pyssht
import healpy as hp
from math import floor, ceil


from pxmcmc import plotting
from pxmcmc import uncertainty
from pxmcmc.transforms import WaveletTransform
from pxmcmc.utils import map2alm


parser = argparse.ArgumentParser()
parser.add_argument("datafile", type=str)
parser.add_argument("directory", type=str)
parser.add_argument("--suffix", type=str, default="")
parser.add_argument("--burn", type=int, default=3000)
parser.add_argument("--chain_mwlm", action="store_true", help="Convert chain to mwlm")
args = parser.parse_args()


def filename(name):
    return f"{args.directory}/{name}{args.suffix}.png"


file = h5py.File(args.datafile, "r")
params = {attr: file.attrs[attr] for attr in file.attrs.keys()}
L, B, J_min, setting = params["L"], params["B"], params["J_min"], params["setting"]
nscales = pys2let.pys2let_j_max(B, L, J_min) - J_min + 1
wvlttrans = WaveletTransform(
    L,
    B,
    J_min,
    inv_out_type="pixel_mw",
    inv_in_type="pixel_mw",
    fwd_out_type="pixel_mw",
    fwd_in_type="pixel_mw",
)
mw_shape = pyssht.sample_shape(L, Method="MW")

logpi = file["logposterior"][()]
L2s = file["L2s"][()]
L1s = file["L1s"][()]
evo = plotting.plot_evolution(logpi, L2s, L1s)
evo.savefig(filename("evolution"))

topo = hp.read_map("ETOPO1_Ice_hpx_256.fits", verbose=False, dtype=np.float64,)
truth = pyssht.inverse(pys2let.lm_hp2lm(map2alm(topo, L - 1), L), L, Reality=True)

MAP_idx = np.where(logpi == max(logpi))
MAP_X = file["chain"][MAP_idx][0]
if setting == "synthesis":
    MAP = wvlttrans.inverse(MAP_X)
    MAP_wvlt = np.copy(MAP_X)
else:
    MAP = np.copy(MAP_X)
    MAP_wvlt = wvlttrans.forward(MAP_X)
MAP = MAP.reshape(mw_shape).astype(float)
maxapost = plotting.plot_map(MAP, title="Maximum a posetriori solution")
maxapost.savefig(filename("MAP"))

diff = 100 * (truth - MAP) / truth
cbar_end = min(max([abs(floor(np.min(diff))), ceil(np.max(diff))]), 100)
diffp = plotting.plot_map(
    diff,
    title="True - MAP",
    cmap="PuOr",
    vmin=-cbar_end,
    vmax=cbar_end,
    cbar_label="%",
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

if "noise" in params:
    noise = params["noise"].reshape((L, 2 * L - 1))
    noise_map = plotting.plot_map(
        noise, title="Added noise", cmap="binary", oversample=False
    )
    noise_map.savefig(filename("noise"))
