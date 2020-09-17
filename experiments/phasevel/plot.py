import argparse
import h5py
import numpy as np
import pys2let
import pyssht

from pxmcmc import plotting
from pxmcmc import uncertainty
from pxmcmc.transforms import WaveletTransform


parser = argparse.ArgumentParser()
parser.add_argument("datafile", type=str)
parser.add_argument("directory", type=str)
parser.add_argument("setting", type=str)
parser.add_argument("--suffix", type=str, default="")
parser.add_argument("--burn", type=int, default=3000)
args = parser.parse_args()


def filename(name):
    return f"{args.directory}/{name}{args.suffix}.png"


file = h5py.File(args.datafile, "r")
params = {attr: file.attrs[attr] for attr in file.attrs.keys()}
L, B, J_min = params["L"], params["B"], params["J_min"]
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


MAP_idx = np.where(logpi == max(logpi))
MAP_X = file["chain"][MAP_idx][0]
if args.setting == "synthesis":
    MAP = wvlttrans.inverse(MAP_X)
    MAP_wvlt = np.copy(MAP_X)
else:
    MAP = np.copy(MAP_X)
    MAP_wvlt = wvlttrans.forward(MAP_X)
MAP = MAP.reshape(mw_shape).astype(float)
MAP_plt, _ = pyssht.mollweide_projection(MAP, L)
maxapost = plotting.plot_map(MAP_plt, title="Maximum a posetriori solution")
maxapost.savefig(filename("MAP"))

map_wvlt = plotting.plot_chain_sample(MAP_wvlt)
map_wvlt.savefig(filename("MAP_wvlt"))


chain_pix = np.zeros(
    (file.attrs["nsamples"] - args.burn, pyssht.sample_length(L, Method="MW"))
)
for i, sample in enumerate(file["chain"][args.burn :]):
    if args.setting == "synthesis":
        chain_pix[i] = wvlttrans.inverse(sample)
    else:
        chain_pix[i] = np.copy(sample)
ci_range = uncertainty.credible_interval_range(chain_pix).reshape(mw_shape)
ci_range_plt, _ = pyssht.mollweide_projection(ci_range, L)
ci_map = plotting.plot_map(
    ci_range_plt, title="95% credible interval range", cmap="viridis", vmin=0
)
ci_map.savefig(filename("ci_map"))