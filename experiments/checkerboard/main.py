import healpy as hp
import numpy as np
import pys2let

from pxmcmc.mcmc import PxMCMC, PxMCMCParams
from pxmcmc.forward import ISWTOperator
from pxmcmc.saving import Outfile

L = 15
B = 1.5
J_min = 2
sig_d = 0.01
true_model = hp.read_map("chkrbrd30.fits", verbose=False)
alms = pys2let.lm_hp2lm(hp.map2alm(true_model, lmax=L), L + 1)
noise = np.random.normal(scale=sig_d, size=alms.shape)
data = alms + noise

forwardop = ISWTOperator(data, sig_d, L, B, J_min)
params = PxMCMCParams(nsamples=int(5e5), nburn=0, ngap=0, verbosity=1000, complex=True)
mcmc = PxMCMC(forwardop, params)
mcmc.mcmc()

writer = Outfile(mcmc.logPi, mcmc.preds, mcmc.chain, "ISWT")
writer.write_outfiles()
