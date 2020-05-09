import numpy as np
import healpy as hp

from pxmcmc.mcmc import PxMCMC, PxMCMCParams
from pxmcmc.forward import ForwardOperator
from pxmcmc.saving import Outfile


def simpledata(Nside, sig_d):
    simple = np.full(hp.nside2npix(Nside), 10)
    noise = np.random.normal(scale=sig_d, size=simple.shape)
    return simple + noise


data = simpledata(4, 0.01)
forwardop = ForwardOperator(data, 0.01)
params = PxMCMCParams(nsamples=int(5e5), nburn=0, ngap=0, nparams=forwardop.nparams,)
mcmc = PxMCMC(forwardop, params)
mcmc.mcmc()

writer = Outfile(mcmc.logPi, mcmc.preds, mcmc.chain, ".")
writer.write_outfiles()
