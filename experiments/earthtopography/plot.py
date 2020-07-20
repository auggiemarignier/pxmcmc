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
from pxmcmc.utils import map2alm, WaveletFormatter, expand_mlm


parser = argparse.ArgumentParser()
parser.add_argument("datafile", type=str)
parser.add_argument("directory", type=str)
parser.add_argument("setting", type=str)
parser.add_argument("--suffix", type=str, default="")
parser.add_argument("--burn", type=int, default=3000)
parser.add_argument("--chain_mwlm", action="store_true", help="Convert chain to mwlm")
args = parser.parse_args()

Nside = 32


def filename(name):
    return f"{args.directory}/{name}{args.suffix}.png"


file = h5py.File(args.datafile, "r")
params = {attr: file.attrs[attr] for attr in file.attrs.keys()}
L, B, J_min = params["L"], params["B"], params["J_min"]
nscales = pys2let.pys2let_j_max(B, L, J_min) - J_min + 1
wvlttrans = WaveletTransform(
    L, B, J_min, fwd_in_type="pixel_mw", fwd_out_type="harmonic_mw"
)
wvltform = WaveletFormatter(L, B, J_min, Nside)

logpi = file["logposterior"][()]
L2s = file["L2s"][()]
L1s = file["L1s"][()]
evo = plotting.plot_evolution(logpi, L2s, L1s)
evo.savefig(filename("evolution"))

topo = hp.read_map(
    "/home/zcfbllm/src_proxmcmc/experiments/earthtopography/ETOPO1_Ice_hpx_256.fits",
    verbose=False,
    dtype=np.float64,
)
truth = pyssht.inverse(pys2let.lm_hp2lm(map2alm(topo, L - 1), L), L, Reality=True)

MAP_idx = np.where(logpi == max(logpi))
MAP = file["predictions"][MAP_idx][0]
MAP = pyssht.inverse(pys2let.map2alm_mw(MAP, L, 0), L, Reality=True)
MAP_plt, _ = pyssht.mollweide_projection(MAP, L)
maxapost = plotting.plot_map(MAP_plt, title="Maximum a posetriori solution")
maxapost.savefig(filename("MAP"))

diff = truth - MAP
cbar_end = max([abs(floor(np.min(diff))), ceil(np.max(diff))])
diff_plt, _ = pyssht.mollweide_projection(diff, L)
diffp = plotting.plot_map(diff_plt, title="True - MAP", cmap="PuOr", vmin=-cbar_end, vmax=cbar_end)
diffp.savefig(filename("diff"))

MAP_X = file["chain"][MAP_idx][0]
if args.chain_mwlm:
    if args.setting == "synthesis":
        wavs, scal = expand_mlm(MAP_X, nscales, flatten_wavs=True)
        scal_lm, wav_lm = wvltform._pixmw2harmmw_wavelets(scal, wavs)
        MAP_X = np.concatenate([scal_lm, wav_lm])
    else:
        MAP_X = wvlttrans.forward(MAP_X)
mapx = plotting.plot_chain_sample(MAP_X)
mapx.savefig(filename("MAP_X"))

ci_range = uncertainty.credible_interval_range(file["predictions"][3000:])
ci_range = pyssht.inverse(pys2let.map2alm_mw(ci_range, L, 0), L, Reality=True)
ci_range_plt, _ = pyssht.mollweide_projection(ci_range, L)
ci_map = plotting.plot_map(
    ci_range_plt, title="95% credible interval range", cmap="viridis", vmin=0
)
ci_map.savefig(filename("ci_map"))
