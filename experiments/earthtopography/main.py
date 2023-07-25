"""
This example simply recovers an image of global Earth topography.
Effectively, it is a long-winded way of obtaining the spherical wavelet coefficients of the topography image.
Input: Earth topography
Output: PxMCMC samples of Earth topography spherical wavelet coefficients
"""

import healpy as hp
import numpy as np
import datetime
import argparse
import s2wav
import s2fft
from s2fft import sampling

from pxmcmc.mcmc import MYULA, PxMALA, SKROCK, PxMCMCParams
from pxmcmc.forward import SphericalWaveletTransformOperator
from pxmcmc.prior import S2_Wavelets_L1, L1
from pxmcmc.saving import save_mcmc
from pxmcmc.utils import calc_pixel_areas

parser = argparse.ArgumentParser()
parser.add_argument(
    "--infile",
    type=str,
    default="ETOPO1_Ice_hpx_256.fits",
    help="Path to input datafile.",
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
parser.add_argument("--L", type=int, default=32, help="Angular bandlimit. Default 32.")
parser.add_argument("--makenoise", action="store_true", help="Add noise to data.")
parser.add_argument(
    "--sigma", type=float, default=1, help="Noise level to be added to data."
)
parser.add_argument(
    "--scaleafrica",
    type=int,
    default=0,
    help="Factor by which to increase the noise level in Africa.",
)
args = parser.parse_args()

# Set up wavelet parameters
L = args.L
B = 1.5
J_min = 2
sigma = args.sigma
setting = args.setting

# Read input data
if "_hpx_" in args.infile:
    topo = hp.read_map(args.infile, verbose=False)
    topo_d_lm = hp.map2alm(topo, L - 1)
    topo_d = pys2let.alm2map_mw(sampling.lm_hp_to_2d(topo_d_lm, L).flatten(), L, 0)
elif "_mw_" in args.infile:
    topo = np.load(args.infile)
    topo_d = topo.reshape((L, 2 * L - 1))
else:
    raise ValueError("Check filename")

if args.makenoise:  # Adding noise to data, as noise would be present in real data
    np.random.seed(2)
    areas = calc_pixel_areas(L)
    sig_d = np.sqrt(sigma**2 / areas)
    if args.scaleafrica:  # Extra noisy in Africa
        thetas = np.deg2rad(np.linspace(60, 120, 100))
        phis = np.deg2rad(np.linspace(-30, 30, 100))
        block = np.zeros((L, 2 * L - 1))
        for theta in thetas:
            theta_ind = int((theta * (2 * L - 1) / np.pi - 1) // 2)
            for phi in phis:
                phi_ind = int(phi * (2 * L - 1) / (2 * np.pi))
                block[theta_ind, phi_ind] = 1
        sig_d[block == 1] *= args.scaleafrica
    sig_d = sig_d.flatten()  # flatten() by default goes to C ordering like in s2let
    noise = np.random.normal(0, sig_d)
    topo_d += noise
    np.random.seed(None)
else:
    sig_d = sigma
    noise = 0

# Set up forward operator (measurement and transform).
# This makes data predictions for a given MCMC sample, and compares
# with the observed data.
# In this case, we're using a simple Identity measurement operator
# and a spherical wavelet transform.
# See the definition of SphericalWaveletTransformOperator and the
# docs for ForwardOperator.
forwardop = SphericalWaveletTransformOperator(
    topo_d / 1000, sig_d, setting, L, B, J_min
)

# Set MCMC tuning parameters
params = PxMCMCParams(
    nsamples=int(1e2),  # In practice you would want much more than this
    nburn=int(0),
    ngap=int(5e2),
    delta=args.delta,
    lmda=1e-6,
    mu=args.mu,
    complex=False,
    verbosity=5e3,
    s=10,
)

# Set up the regulariser/prior.
# This calculates the prior probability of the MCMC sample,
# and importantly calculates the proximal mapping of the prior
# to more efficiently navigate the non-smooth parameter space.
# See the docs of L1.
regulariser = S2_Wavelets_L1(
    setting,
    forwardop.transform.inverse,
    forwardop.transform.inverse_adjoint,
    params.lmda * params.mu,
    L=L,
    B=B,
    J_min=J_min,
)

print(f"Number of data points: {len(topo_d)}")
print(f"Number of model parameters: {forwardop.nparams}")

# Choose PxMCMC sampler.
# Three different samplers have been implemented in this pacakge.
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

# Save the results!
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
    noise=noise,
    setting=setting,
    sigma=sigma,
    scaleafrica=args.scaleafrica,
    time=str(datetime.datetime.now() - NOW),
)
