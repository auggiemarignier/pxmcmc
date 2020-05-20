import healpy as hp
import numpy as np
import pys2let

from pxmcmc.mcmc import PxMCMC, PxMCMCParams
from pxmcmc.forward import ISWTOperator
from pxmcmc.saving import save_mcmc

root_dir = "/Users/auggiemarignier/Documents/PhD/PxMCMC/experiments/checkerboard"

L = 15
B = 1.5
J_min = 2
sig_d = 0.01
true_model = hp.read_map(f"{root_dir}/chkrbrd30.fits", verbose=False)
alms = pys2let.lm_hp2lm(hp.map2alm(true_model, lmax=L), L + 1)
noise = np.random.normal(scale=sig_d, size=alms.shape)
# data = alms + noise
data = alms

forwardop = ISWTOperator(data, sig_d, L, B, J_min)
params = PxMCMCParams(
    nsamples=int(1e6),
    nburn=int(1e4),
    ngap=0,
    complex=True,
    delta=1e-7,
    lmda=3e-7,
    mu=1e-7,
    verbosity=1000,
)
mcmc = PxMCMC(forwardop, params, X_func=None)
mcmc.mcmc()

save_mcmc(
    mcmc, params, "/Volumes/Elements", L=L, B=B, J_min=J_min, sig_d=sig_d, nparams=forwardop.nparams
)
