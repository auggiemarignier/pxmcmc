import healpy as hp
import datetime
import argparse
import pys2let

from pxmcmc.mcmc import MYULA, PxMALA, PxMCMCParams
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

topo = hp.read_map(args.infile, verbose=False)
sig_d = 0.03

L = 16
B = 1.5
J_min = 2
Nside = 32
setting = args.setting
# topo_d = hp.ud_grade(topo, Nside)
topo_d_lm = hp.map2alm(topo, L - 1)
topo_d = pys2let.alm2map_mw(pys2let.lm_hp2lm(topo_d_lm, L), L, 0)

forwardop = WaveletTransformOperator(topo_d, sig_d, setting, L, B, J_min)
params = PxMCMCParams(
    nsamples=int(5e3),
    nburn=0,
    ngap=int(1e2),
    complex=True,
    delta=args.delta,
    lmda=1e-7,
    mu=args.mu,
    verbosity=int(1e2),
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
else:
    raise ValueError
mcmc.run()

save_mcmc(
    mcmc,
    params,
    args.outdir,
    filename=f"{args.algo}_{args.setting}_{NOW.strftime('%d%m%y_%H%M%S')}_{args.jobid}",
    L=L,
    B=B,
    J_min=J_min,
    sig_d=sig_d,
    nparams=forwardop.nparams,
)
