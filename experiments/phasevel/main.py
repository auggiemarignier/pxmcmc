"""
Inverts global path averaged Rayleigh wave data to obtain
a global map of phase velocity.
This replicates the example shown in the RASTI paper
https://doi.org/10.1093/rasti/rzac010
"""

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
from pxmcmc.prior import S2_Wavelets_L1_Power_Weights
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


def build_path(start, stop, L):
    """
    Find all the MW pixels a great circle passes through.
    """
    path = GreatCirclePath(start, stop, "MW", L=L, weighting="average", latlon=True)
    path.get_points(points_per_rad=160)
    path.fill()
    return path.map


def get_path_matrix(start, stop, L=32, processes=16):
    """
    Build a matrix of all the great cricle paths.
    This is effectively the measurement operator matrix.
    """
    itrbl = [(stt, stp, L) for (stt, stp) in zip(start, stop)]
    with Pool(processes) as p:
        result = p.starmap_async(build_path, itrbl)
        paths = result.get()
    return sparse.csr_matrix(paths)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "infile",
        type=str,
        default="synthetic_GDM40_0S254_L28.txt",
        help="Path to input datafile.",
    )
    parser.add_argument(
        "pathsfile",
        type=str,
        default="0S254L28.npz",
        help="path to .npz file with scipy sparse matrix.  If file is not found, sparse matrix will be generated and saved here.",
    )
    parser.add_argument(
        "--outdir", type=str, default=".", help="Output directory. Default '.'."
    )
    parser.add_argument(
        "--jobid",
        type=str,
        default="0",
        help="Optional ID that will be added to the end of the output filename. Default '0'.",
    )
    parser.add_argument(
        "--algo",
        type=str,
        default="myula",
        help="PxMCMC algorithm to be used. One of ['myula', 'pxmala', 'skrock']. Default 'myula'.",
    )
    parser.add_argument(
        "--setting",
        type=str,
        default="synthesis",
        help="'synthesis' or 'analysis'. Default 'myula'.",
    )
    parser.add_argument(
        "--delta", type=float, default=1e-6, help="PxMCMC step size. Default 1e-6"
    )
    parser.add_argument(
        "--mu",
        type=float,
        default=1,
        help="Regularisation parameter (prior width). Default 1.",
    )
    parser.add_argument(
        "--L", type=int, default=28, help="Angular bandlimit. Default 28."
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=1,
        help="Wavelet power decay factor.  See pxmcmc.prior.S2_Wavelets_L1_Power_Weights. Default 1.",
    )
    parser.add_argument(
        "--nsim",
        action="store_true",
        help="Applies wieghting for number of similar paths",
    )

    # Setup global parameters
    args = parser.parse_args()
    L = args.L
    B = 2
    J_min = 2
    setting = args.setting

    # Read data and path matrix
    start, stop, data, sig_d, _, nsim = read_datafile(args.infile)
    if path.exists(args.pathsfile):
        path_matrix = sparse.load_npz(args.pathsfile)
    else:
        path_matrix = get_path_matrix(start, stop, L)
        sparse.save_npz(args.pathsfile, path_matrix)

    assert path_matrix.shape[0] == len(data)

    if args.nsim:
        sig_d *= np.sqrt(nsim)  # sig_d will get squared later

    # Set up the forward operator, which is a combination of
    # great circle path integration and a spherical wavelet transform
    forwardop = PathIntegralOperator(path_matrix, data, sig_d, setting, L, B, J_min)

    # Set MCMC tuning parameters
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

    # Set up prior, in this case Laplace with weighting to account for
    # pixel size and wavelet scale power
    regulariser = S2_Wavelets_L1_Power_Weights(
        setting,
        forwardop.transform.inverse,
        forwardop.transform.inverse_adjoint,
        params.lmda * params.mu,
        L=L,
        B=B,
        J_min=J_min,
        eta=args.eta,
    )

    print(f"Number of data points: {len(data)}")
    print(f"Number of model parameters: {forwardop.nparams}")

    # Select MCMC sampler
    if args.algo == "myula":
        mcmc = MYULA(forwardop, regulariser, params)
    elif args.algo == "pxmala":
        mcmc = PxMALA(forwardop, regulariser, params, tune_delta=True)
    elif args.algo == "skrock":
        mcmc = SKROCK(forwardop, regulariser, params)
    else:
        raise ValueError

    # RUN!
    NOW = datetime.datetime.now()
    mcmc.run()

    # save
    filename = (
        f"{args.algo}_{args.setting}_{NOW.strftime('%d%m%y_%H%M%S')}_{args.jobid}"
    )
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
        nsim=True if args.nsim else False,
        eta=args.eta,
    )
