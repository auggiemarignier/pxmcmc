"""
Solves the weak lensing mass-mapping problem with PxMCMC.
This replicates the example in the RASTI paper
https://doi.org/10.1093/rasti/rzac010
"""

import numpy as np
import argparse
import datetime
import s2fft
from s2wav import sampling
import healpy as hp

from pxmcmc.measurements import WeakLensing
from pxmcmc.transforms import SphericalWaveletTransform
from pxmcmc.forward import ForwardOperator
from pxmcmc.mcmc import PxMCMCParams, MYULA, PxMALA, SKROCK
from pxmcmc.prior import S2_Wavelets_L1
from pxmcmc.saving import save_mcmc
from pxmcmc.utils import build_mask


def load_gammas(kappa_fits_file: str, L: int, wl: WeakLensing):
    """
    Loads a fits file containing the kappa ground truth
    Expects kappa in healpix format
    Needs a WL operator with initialised mask

    Returns gamma predictions in MW format
    """
    kappa = hp.read_map(kappa_fits_file)
    lmax = L - 1
    nside = 3 * lmax - 1
    kappa_bl = hp.alm2map(hp.map2alm(kappa, lmax=lmax), nside=nside)
    sigma = np.radians(50 / 60)  # 50 arcmin
    kappa_s = hp.smoothing(kappa_bl, sigma=sigma)
    kappa_mw = pyssht.inverse(sampling.flm_hp_to_2d(hp.map2alm(kappa_s, lmax).flatten(), L), L)

    return wl.forward(kappa_mw)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "infile",
        type=str,
        help="A fits file containing the kappa ground truth in healpix format",
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
        "--L", type=int, default=512, help="Angular bandlimit. Default 512."
    )

    # Set global parameters
    args = parser.parse_args()
    L = args.L
    B = 2
    J_min = 2
    setting = args.setting

    # Build a Euclid-like mask and create synthetic shear data
    mask = build_mask(L, size=10)
    measurement = WeakLensing(L, mask, ngal=np.full_like(mask, 30))
    gammas_truth = load_gammas(args.infile, L, measurement)

    # Choose transform
    transform = SphericalWaveletTransform(L, B, J_min)

    # Combine measurement and transform operators into a single
    # forward operator
    forward_operator = ForwardOperator(
        gammas_truth,
        1 / measurement.inv_cov,
        setting,
        transform=transform,
        measurement=measurement,
        nparams=transform.ncoefs,
    )

    # Set mcmc parameters
    params = PxMCMCParams(
        nsamples=int(5e3),
        nburn=10e6,
        ngap=int(500),
        delta=args.delta,
        lmda=args.delta / 2,
        mu=args.mu,
        complex=False,
        verbosity=1e3,
    )

    # Set prior (L1 norm with pixel area weighting)
    prior = S2_Wavelets_L1(
        setting,
        transform.inverse,
        transform.inverse_adjoint,
        params.lmda * params.mu,
        L=L,
        B=B,
        J_min=J_min,
    )

print(f"Number of data points: {gammas_truth.size}")
print(f"Number of model parameters: {forward_operator.nparams}")

# Choose PxMCMC algorithm
if args.algo == "myula":
    mcmc = MYULA(forward_operator, prior, params)
elif args.algo == "pxmala":
    mcmc = PxMALA(forward_operator, prior, params, tune_delta=True)
elif args.algo == "skrock":
    mcmc = SKROCK(forward_operator, prior, params)
else:
    raise ValueError

# RUN
NOW = datetime.datetime.now()
mcmc.run()

# Save
filename = f"{args.algo}_{args.setting}_{NOW.strftime('%d%m%y_%H%M%S')}_{args.jobid}"
save_mcmc(
    mcmc,
    params,
    args.outdir,
    filename=filename,
    L=L,
    B=B,
    J_min=J_min,
    nparams=forward_operator.nparams,
    setting=setting,
    time=str(datetime.datetime.now() - NOW),
)
