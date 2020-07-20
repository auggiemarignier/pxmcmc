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

parser = argparse.ArgumentParser()
parser.add_argument("--infile", type=str, default="ETOPO1_Ice_hpx_256.fits")
parser.add_argument("--outdir", type=str, default=".")
parser.add_argument("--jobid", type=str, default="0")

parser.add_argument("--algo", type=str, default="myula")
parser.add_argument("--setting", type=str, default="synthesis")
parser.add_argument("--delta", type=float, default=1e-10)
parser.add_argument("--mu", type=float, default=1)
args = parser.parse_args()

sig_d = 200

L = 16
B = 1.5
J_min = 2
Nside = 32
setting = args.setting
if "_hpx_" in args.infile:
    topo = hp.read_map(args.infile, verbose=False)
    topo_d_lm = hp.map2alm(topo, L - 1)
    topo_d = pys2let.alm2map_mw(pys2let.lm_hp2lm(topo_d_lm, L), L, 0)
elif "_mw_" in args.infile:
    topo = np.load(args.infile)
    topo_d_lm = pyssht.forward(topo, L, Reality=True)
    topo_d = pys2let.alm2map_mw(topo_d_lm, L, 0)  # just to get shapes right
else:
    raise ValueError("Check filename")

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
    s=10
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
)
