import healpy as hp
import numpy as np
import datetime
import argparse
import pys2let
import pyssht

from pxmcmc.mcmc import MYULA, PxMALA, SKROCK, PxMCMCParams
from pxmcmc.forward import WaveletTransformOperator
from pxmcmc.prox import L1
from pxmcmc.saving import save_mcmc
from pxmcmc.utils import calc_pixel_areas

parser = argparse.ArgumentParser()
parser.add_argument("--infile", type=str, default="ETOPO1_Ice_hpx_256.fits")
parser.add_argument("--outdir", type=str, default=".")
parser.add_argument("--jobid", type=str, default="0")

parser.add_argument("--algo", type=str, default="myula")
parser.add_argument("--setting", type=str, default="synthesis")
parser.add_argument("--delta", type=float, default=1e-10)
parser.add_argument("--mu", type=float, default=1)

parser.add_argument("--L", type=int, default=16)
parser.add_argument("--sigma", type=float, default=1)
parser.add_argument("--makenoise", action="store_true")
parser.add_argument("--scaleafrica", type=int, default=0)
args = parser.parse_args()


L = args.L
B = 1.5
J_min = 2
sigma = args.sigma
setting = args.setting
if "_hpx_" in args.infile:
    topo = hp.read_map(args.infile, verbose=False)
    topo_d_lm = hp.map2alm(topo, L - 1)
    topo_d = pys2let.alm2map_mw(pys2let.lm_hp2lm(topo_d_lm, L), L, 0)
elif "_mw_" in args.infile:
    topo = np.load(args.infile)
    topo_d = topo.reshape((L, 2 * L - 1))
else:
    raise ValueError("Check filename")

if args.makenoise:
    areas = calc_pixel_areas(L)
    sig_d = np.sqrt(sigma ** 2 / areas)
    if args.scaleafrica:
        thetas = np.deg2rad(np.linspace(60, 120, 100))
        phis = np.deg2rad(np.linspace(-30, 30, 100))
        block = np.zeros((L, 2 * L - 1))
        for theta in thetas:
            theta_ind = pyssht.theta_to_index(theta, L)
            for phi in phis:
                phi_ind = pyssht.phi_to_index(phi, L)
                block[theta_ind, phi_ind] = 1
        sig_d[block == 1] *= args.scaleafrica
    sig_d = sig_d.flatten()  # flatten() by default goes to C ordering like in s2let
    noise = np.random.normal(0, sig_d)
    topo_d += noise
else:
    sig_d = sigma
    noise = None

forwardop = WaveletTransformOperator(topo_d, sig_d, setting, L, B, J_min)
params = PxMCMCParams(
    nsamples=int(5e3),
    nburn=0,
    ngap=int(1),
    # ngap=int(1e2),
    complex=True,
    delta=args.delta,
    lmda=1e-7,
    mu=args.mu,
    verbosity=int(1),
    # verbosity=int(1e2),
    s=10,
)

regulariser = L1(
    setting,
    forwardop.transform.inverse,
    forwardop.transform.inverse_adjoint,
    params.lmda * params.mu,
)

print(f"Number of data points: {len(topo_d)}")
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
    sig_d=sig_d,
    nparams=forwardop.nparams,
    noise=noise,
)
