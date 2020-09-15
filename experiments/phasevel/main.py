import numpy as np
import argparse
from os import path
from scipy import sparse
from greatcirclepaths import GreatCirclePath
from multiprocessing import Pool
import datetime

from pxmcmc.mcmc import MYULA, PxMALA, SKROCK, PxMCMCParams
from pxmcmc.forward import PathIntegralOperator
from pxmcmc.prox import L1
from pxmcmc.saving import save_mcmc


def read_datafile(datafile):
    """
    Expects a file with the following columns for each path:
    start_lat, start_lon, stop_lat, stop_lon, data, error
    Coordinates given in degrees
    TODO: Figure out what to do with minor/major and nsim
    """
    all_data = np.loadtxt(datafile)
    start_lat = all_data[:, 0]
    start_lon = all_data[:, 1]
    start = np.stack([start_lat, start_lon], axis=1)
    stop_lat = all_data[:, 2]
    stop_lon = all_data[:, 3]
    stop = np.stack([stop_lat, stop_lon], axis=1)
    data = all_data[:, 4]
    sig_d = all_data[:, 5]
    return start, stop, data, sig_d


parser = argparse.ArgumentParser()
parser.add_argument("infile", type=str)
parser.add_argument("--outdir", type=str, default=".")
parser.add_argument("--jobid", type=str, default="0")
parser.add_argument(
    "--pathsfile",
    type=str,
    help="path to .npz file with scipy sparse matrix.  If file is not found, sparse matrix will be generated and saved here.",
)
parser.add_argument("--algo", type=str, default="myula")
parser.add_argument("--setting", type=str, default="synthesis")
parser.add_argument("--lmda", type=float, default=2e-10)
parser.add_argument("--delta", type=float, default=1e-10)
parser.add_argument("--mu", type=float, default=1)
parser.add_argument("--L", type=int, default=32)

args = parser.parse_args()
L = args.L
B = 1.5
J_min = 2
setting = args.setting


def build_path(start, stop):
    path = GreatCirclePath(start, stop, "MW", L=args.L, weighting=True)
    path.get_points(100)
    path.fill()
    return path.map


def get_path_matrix(start, stop, processes=4):
    itrbl = [(stt, stp) for (stt, stp) in zip(start, stop)]
    with Pool(processes) as p:
        result = p.starmap_async(build_path, itrbl)
        paths = result.get()
    return sparse.csr_matrix(paths)


start, stop, data, sig_d = read_datafile(args.infile)
if path.exists(args.pathsfile):
    path_matrix = sparse.load_npz(args.pathsfile)
else:
    path_matrix = get_path_matrix(start, stop)
    sparse.save_npz(args.pathsfile)

assert path_matrix.shape[0] == len(data)

forwardop = PathIntegralOperator(path_matrix, data, sig_d, setting, L, B, J_min)
params = PxMCMCParams(
    nsamples=int(5e3),
    nburn=0,
    ngap=int(1),
    delta=args.delta,
    lmda=args.lmda,
    mu=args.mu,
    verbosity=int(1),
    s=10,
)

regulariser = L1(
    setting,
    forwardop.transform.inverse,
    forwardop.transform.inverse_adjoint,
    params.lmda * params.mu,
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
    setting=setting
)
