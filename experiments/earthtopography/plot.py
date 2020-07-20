import argparse
import h5py
import numpy as np
from matplotlib import cm
import pys2let
import healpy as hp
from math import floor, ceil


from pxmcmc import plotting
from pxmcmc import uncertainty
from pxmcmc.transforms import WaveletTransform
from pxmcmc.utils import alm2map, map2alm, WaveletFormatter, expand_mlm


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
truth = alm2map(map2alm(topo, params["L"] - 1), Nside)

MAP_idx = np.where(logpi == max(logpi))
MAP = file["predictions"][MAP_idx][0]
MAP_hp = alm2map(
    pys2let.lm2lm_hp(pys2let.map2alm_mw(MAP, params["L"], 0), params["L"]), Nside
)
maxapost = plotting.mollview(
    MAP_hp, cmap=cm.jet, title="Maximum a posteriori solution", flip="geo"
)
maxapost.savefig(filename("MAP"))

diff = truth - MAP_hp
cbar_end = max([abs(floor(min(diff))), ceil(max(diff))])
diffp = plotting.mollview(
    diff, cmap=cm.PuOr, title="Truth - MAP", flip="geo", min=-cbar_end, max=cbar_end
)
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

# TODO: Sort this out for the analysis setting
if args.chain_mwlm:
    chain = np.zeros(
        (len(file["chain"][args.burn :]), L * L * (nscales + 1)), dtype=np.complex
    )
    if args.setting == "synthesis":
        for i in range(len(chain)):
            print(f"\r{i+1}/{len(chain)}", end="")
            wavs, scal = expand_mlm(
                file["chain"][args.burn + i], nscales, flatten_wavs=True
            )
            scal_lm, wav_lm = wvltform._pixmw2harmmw_wavelets(scal, wavs)
            chain[i] = np.concatenate([scal_lm, wav_lm])
    else:
        for i in range(len(chain)):
            print(f"\r{i+1}/{len(chain)}", end="")
            chain[i] = wvlttrans.forward(file["chain"][args.burn + i])

    basis_els = plotting.plot_basis_els(chain, L, B, J_min, inflate_mads=100)
    basis_els.savefig(filename("basis_els"))

ci_range = uncertainty.credible_interval_range(file["predictions"][3000:])
ci_range_hp = alm2map(
    pys2let.lm2lm_hp(pys2let.map2alm_mw(ci_range, params["L"], 0), params["L"]), Nside
)
ci_map = plotting.mollview(
    ci_range_hp, min=0, title="95% credible interval range", flip="geo"
)
ci_map.savefig(filename("ci_map"))
print()
