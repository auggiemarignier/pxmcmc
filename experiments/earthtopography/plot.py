import argparse
import h5py
import numpy as np
from matplotlib import cm
import pys2let
import healpy as hp

from pxmcmc import plotting
from pxmcmc import uncertainty
from pxmcmc.utils import alm2map


parser = argparse.ArgumentParser()
parser.add_argument("datafile", type=str)
parser.add_argument("directory", type=str)
parser.add_argument("--suffix", type=str, default="")
args = parser.parse_args()

Nside = 32


def filename(name):
    return f"{args.directory}/{name}{args.suffix}.png"


file = h5py.File(args.datafile, "r")
params = {attr: file.attrs[attr] for attr in file.attrs.keys()}

logpi = file["logposterior"][()]
L2s = file["L2s"][()]
L1s = file["L1s"][()]
evo = plotting.plot_evolution(logpi, L2s, L1s)
evo.savefig(filename("evolution"))

topo = hp.read_map(
    "/home/zcfbllm/src_proxmcmc/experiments/earthtopography/ETOPO1_Ice_hpx_256.fits",
    verbose=False,
)
truth = hp.alm2map(hp.map2alm(topo, params["L"] - 1), Nside)

MAP_idx = np.where(logpi == max(logpi))
MAP = file["predictions"][MAP_idx][0]
MAP_hp = alm2map(
    pys2let.lm2lm_hp(pys2let.map2alm_mw(MAP, params["L"], 0), params["L"]), Nside
)
maxapost = plotting.mollview(
    MAP_hp, cmap=cm.jet, title="Maximum a posteriori solution", flip="geo"
)
maxapost.savefig(filename("MAP"))

diff = plotting.mollview(truth - MAP_hp, cmap=cm.jet, title="Truth - MAP", flip="geo")
diff.savefig(filename("diff"))

MAP_X = file["chain"][MAP_idx][0]
mapx = plotting.plot_chain_sample(MAP_X)
mapx.savefig(filename("MAP_X"))


basis_els = plotting.plot_basis_els(
    file["chain"][3000:], params["L"], params["B"], params["J_min"], inflate_mads=100
)  # Careful with L
basis_els.savefig(filename("basis_els"))

ci_range = uncertainty.credible_interval_range(file["predictions"][3000:])
ci_range_hp = alm2map(
    pys2let.lm2lm_hp(pys2let.map2alm_mw(ci_range, params["L"], 0), params["L"]), 32
)
ci_map = plotting.mollview(
    ci_range_hp, min=0, title="95% credible interval range", flip="geo"
)
ci_map.savefig(filename("ci_map"))
