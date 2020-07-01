import argparse
import h5py
import numpy as np
from matplotlib import cm

from pxmcmc import plotting
from pxmcmc import uncertainty


parser = argparse.ArgumentParser()
parser.add_argument("datafile", type=str)
parser.add_argument("directory", type=str)
args = parser.parse_args()


def filename(name):
    return f"{args.directory}/{name}"


file = h5py.File(args.datafile, "r")
params = {attr: file.attrs[attr] for attr in file.attrs.keys()}

logpi = file["logposterior"][()]
L2s = file["L2s"][()]
L1s = file["L1s"][()]
evo = plotting.plot_evolution(logpi, L2s, L1s)
evo.savefig(filename("evolution.png"))


MAP_idx = np.where(logpi == max(logpi))
MAP = file["predictions"][MAP_idx][0]
maxapost = plotting.mollview(MAP, cmap=cm.jet, title="Maximum a posteriori solution")
maxapost.savefig(filename("MAP.png"))

MAP_X = file["chain"][MAP_idx][0]
mapx = plotting.plot_chain_sample(MAP_X)
mapx.savefig(filename("MAP_X.png"))


basis_els = plotting.plot_basis_els(
    file["chain"][750:], params["L"], params["B"], params["J_min"], inflate_mads=100
)  # Careful with L
basis_els.savefig(filename("basis_els.png"))

ci_range = uncertainty.credible_interval_range(file["predictions"][750:])
ci_map = plotting.mollview(ci_range, min=0, title="95% credible interval range")
ci_map.savefig(filename("ci_map.png"))
