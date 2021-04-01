import numpy as np
import argparse
from os import path
from scipy import sparse
from greatcirclepaths import GreatCirclePath
from multiprocessing import Pool
import datetime
from warnings import warn

from pxmcmc.mcmc import MYULA, PxMALA, SKROCK, PxMCMCParams
from pxmcmc.forward import PathIntegralOperator
from pxmcmc.prior import S2_Wavelets_L1
from pxmcmc.saving import save_mcmc


def read_datafile(datafile):
    """
    Expects a file with the following columns for each path:
    start_lat, start_lon, stop_lat, stop_lon, data, error, minor/majorm, n_similar
    Coordinates given in degrees
    """
    start_lat, start_lon, stop_lat, stop_lon, data, sig_d, mima, nsim = np.loadtxt(
        datafile, unpack=True
    )
    start = np.stack([start_lat, start_lon], axis=1)
    stop = np.stack([stop_lat, stop_lon], axis=1)
    if np.any(sig_d < 0):
        warn("Some of the data errors read in are negative. Forcing positivity.")
        sig_d = np.abs(sig_d)
    return start, stop, data, sig_d, mima, nsim


parser = argparse.ArgumentParser()
parser.add_argument("infile", type=str)
parser.add_argument(
    "pathsfile",
    type=str,
    help="path to .npz file with scipy sparse matrix.  If file is not found, sparse matrix will be generated and saved here.",
)
parser.add_argument("--outdir", type=str, default=".")
parser.add_argument("--jobid", type=str, default="0")
parser.add_argument("--algo", type=str, default="myula")
parser.add_argument("--setting", type=str, default="synthesis")
parser.add_argument("--delta", type=float, default=1e-6)
parser.add_argument("--mu", type=float, default=1)
parser.add_argument("--L", type=int, default=20)
parser.add_argument(
    "--nsim", action="store_true", help="Applies wieghting for number of similar paths"
)

args = parser.parse_args()
L = args.L
B = 2
J_min = 2
setting = args.setting


def build_path(start, stop):
    path = GreatCirclePath(start, stop, "MW", L=args.L, weighting="average")
    path.get_points(points_per_rad=160)
    path.fill()
    return path.map


def get_path_matrix(start, stop, processes=16):
    itrbl = [(stt, stp) for (stt, stp) in zip(start, stop)]
    with Pool(processes) as p:
        result = p.starmap_async(build_path, itrbl)
        paths = result.get()
    return sparse.csr_matrix(paths)


start, stop, data, sig_d, _, nsim = read_datafile(args.infile)
if path.exists(args.pathsfile):
    path_matrix = sparse.load_npz(args.pathsfile)
else:
    path_matrix = get_path_matrix(start, stop)
    sparse.save_npz(args.pathsfile, path_matrix)

assert path_matrix.shape[0] == len(data)

if args.nsim:
    sig_d *= np.sqrt(nsim)  # sig_d will get squared later

forwardop = PathIntegralOperator(path_matrix, data, sig_d, setting, L, B, J_min)
params = PxMCMCParams(
    nsamples=int(2e3),
    nburn=0,
    ngap=int(5e2),
    delta=args.delta,
    lmda=args.delta / 2,
    mu=args.mu,
    complex=False,
    verbosity=1e3,
    s=10,
)

regulariser = S2_Wavelets_L1(
    setting,
    forwardop.transform.inverse,
    forwardop.transform.inverse_adjoint,
    params.lmda * params.mu,
    L=L,
    B=B,
    J_min=J_min,
)

print(f"Number of data points: {len(data)}")
print(f"Number of model parameters: {forwardop.nparams}")

NOW = datetime.datetime.now()

if args.algo == "myula":
    mcmc = MYULA(forwardop, regulariser, params)
elif args.algo == "pxmala":
    mcmc = PxMALA(forwardop, regulariser, params, tune_delta=True)
elif args.algo == "skrock":
    mcmc = SKROCK(forwardop, regulariser, params)
else:
    raise ValueError
mcmc.run()

filename = f"{args.algo}_{args.setting}_{NOW.strftime('%d%m%y_%H%M%S')}_{args.jobid}"
save_mcmc(
    mcmc,
    params,
    args.outdir,
    filename=filename,
    L=L,
    B=B,
    J_min=J_min,
    nparams=forwardop.nparams,
    setting=setting,
    time=str(datetime.datetime.now() - NOW),
    nsim=True if args.nsim else False
)
