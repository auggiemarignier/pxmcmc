import numpy as np
import argparse
from os import path
import datetime
import random
import pyssht
from pys2let import lm_hp2lm
from astropy.coordinates import SkyCoord
import healpy as hp

from pxmcmc.measurements import WeakLensing
from pxmcmc.transforms import SphericalWaveletTransform
from pxmcmc.forward import ForwardOperator
from pxmcmc.mcmc import PxMCMCParams, MYULA
from pxmcmc.prior import S2_Wavelets_L1
from pxmcmc.saving import save_mcmc


def build_mask(L):
    """"
    Builds a mask for the galactic plane and ecliptic
    0 at positions to be masked
    i.e. to apply mask do map * mask

    Mask in MW format
    """
    mask = np.ones(pyssht.sample_shape(L))
    thetas, phis = pyssht.sample_positions(L)
    for i, t in enumerate(thetas):
        for j, p in enumerate(phis):
            if np.abs(90 - np.degrees(t)) < 20:
                mask[i, j] = 0

    thetaarray, phiarray = pyssht.sample_positions(L, Grid=True)
    thetaarray = np.degrees(thetaarray) - 90
    phiarray = np.degrees(phiarray) - 180

    c = SkyCoord(phiarray, thetaarray, unit="deg")
    d = c.transform_to("galactic")
    degm = np.abs(d.b.degree)
    for i in range(L):
        for j in range(2 * L - 1):
            rand = random.random()
            if rand < 0.1:
                mask[i, j] = 0
            if degm[i, j] < 20:
                mask[i, j] = 0

    return mask


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
    kappa_mw = pyssht.inverse(lm_hp2lm(hp.map2alm(kappa_s, lmax), L), L)

    return wl.forward(kappa_mw)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "infile",
        type=str,
        help="A fits file containing the kappa ground truth in healpix format",
    )
    parser.add_argument("--outdir", type=str, default=".")
    parser.add_argument("--jobid", type=str, default="0")
    parser.add_argument("--algo", type=str, default="myula")
    parser.add_argument("--setting", type=str, default="synthesis")
    parser.add_argument("--delta", type=float, default=1e-6)
    parser.add_argument("--mu", type=float, default=1)
    parser.add_argument("--L", type=int, default=512)

    args = parser.parse_args()
    L = args.L
    B = 2
    J_min = 2
    setting = args.setting

    mask = build_mask(L)
    measurement = WeakLensing(L, mask, ngal=np.full_like(mask, 30))
    gammas_truth = load_gammas(args.infile, L, measurement)

    transform = SphericalWaveletTransform(L, B, J_min)
    forward_operator = ForwardOperator(
        gammas_truth,
        1 / measurement.inv_cov,
        setting,
        transform=transform,
        measurement=measurement,
        nparams=transform.ncoefs,
    )

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

NOW = datetime.datetime.now()
mcmc = MYULA(forward_operator, prior, params)
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
    nparams=forward_operator.nparams,
    setting=setting,
    time=str(datetime.datetime.now() - NOW),
)
