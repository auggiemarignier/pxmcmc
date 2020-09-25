import healpy as hp
import numpy as np
import pys2let

from pxmcmc.mcmc import PxMALA, PxMCMCParams
from pxmcmc.forward import ISWTOperator
from pxmcmc.prior import L1
from pxmcmc.saving import save_mcmc

L = 16
lmax = L - 1
B = 1.5
J_min = 2
sig_d = 0.03
true_model = hp.read_map(f"chkrbrd30.fits", verbose=False)
alms = pys2let.lm_hp2lm(hp.map2alm(true_model, lmax), L)
noise = np.random.normal(scale=sig_d, size=alms.shape)
data = alms + noise
# data = alms

setting = "synthesis"
forwardop = ISWTOperator(data, sig_d, L, B, J_min, setting)
params = PxMCMCParams(
    nsamples=int(5e5),
    nburn=0,
    ngap=0,
    complex=True,
    delta=5e-8,
    lmda=3e-7,
    mu=1e2,
    verbosity=1000,
)
regulariser = L1(setting, None, None, params.lmda * params.mu)
mcmc = PxMALA(forwardop, regulariser, params)
mcmc.run()

save_mcmc(
    mcmc,
    params,
    "/Volumes/Elements/PxMCMCoutputs/checkerboard",
    filename="pxmala_synth",
    L=L,
    B=B,
    J_min=J_min,
    sig_d=sig_d,
    nparams=forwardop.nparams,
)
