import numpy as np
import datetime
import argparse
import pys2let
import pyssht

from pxmcmc.mcmc import MYULA, PxMALA, SKROCK, PxMCMCParams
from pxmcmc.forward import WaveletTransformOperator
from pxmcmc.prior import S2_Wavelets_L1, L1
from pxmcmc.saving import save_mcmc
from pxmcmc.utils import calc_pixel_areas

parser = argparse.ArgumentParser()
parser.add_argument("--infile", type=str, default="ETOPO1_Ice_hpx_256.fits")
parser.add_argument("--outdir", type=str, default=".")
parser.add_argument("--jobid", type=str, default="0")

parser.add_argument("--algo", type=str, default="myula")
parser.add_argument("--setting", type=str, default="synthesis")
parser.add_argument("--delta", type=float, default=5e-8)
parser.add_argument("--mu", type=float, default=1)

parser.add_argument("--L", type=int, default=32)
parser.add_argument("--sigma", type=float, default=1)
parser.add_argument("--makenoise", action="store_true")
parser.add_argument("--scaleafrica", type=int, default=0)
args = parser.parse_args()


L = args.L
B = 1.5
J_min = 2
sigma = args.sigma
setting = args.setting
topo = np.load(args.infile)
topo_d = topo.reshape((L, 2 * L - 1))

if args.makenoise:
    np.random.seed(2)
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
    np.random.seed(None)
else:
    sig_d = sigma
    noise = None

forwardop = WaveletTransformOperator(topo_d / 1000, sig_d, setting, L, B, J_min)
params = PxMCMCParams(
    nsamples=int(1e4),
    nburn=int(0),
    ngap=int(5e2),
    delta=args.delta,
    lmda=1e-7,
    mu=args.mu,
    complex=False,
    verbosity=5e3,
    s=10,
)

regulariser = S2_Wavelets_L1(
    setting,
    forwardop.transform.inverse,
    forwardop.transform.inverse_adjoint,
    params.lmda * params.mu,
    L=L,
    B=B,
    J_min=J_min
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
    nparams=forwardop.nparams,
    noise=noise,
    setting=setting,
    sigma=sigma,
    scaleafrica=args.scaleafrica,
    time=datetime.datetime.now() - NOW
)
