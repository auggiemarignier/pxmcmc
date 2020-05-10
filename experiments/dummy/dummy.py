import numpy as np
import healpy as hp

from pxmcmc.mcmc import PxMCMC, PxMCMCParams
from pxmcmc.forward import ForwardOperator, ISWTOperator
from pxmcmc.saving import Outfile


def simpledata(Nside, sig_d):
    simple = np.full(hp.nside2npix(Nside), 10)
    noise = np.random.normal(scale=sig_d, size=simple.shape)
    return simple + noise


def simpledata_lm(Nside, sig_d, L):
    realspace = simpledata(Nside, sig_d)
    alm = hp.map2alm(realspace, lmax=L)
    return alm


# data = simpledata(4, 0.01)
# forwardop = ForwardOperator(data, 0.01)
# params = PxMCMCParams(nsamples=int(5e5), nburn=0, ngap=0, nparams=forwardop.nparams,)
# mcmc = PxMCMC(forwardop, params)
# mcmc.mcmc()

# writer = Outfile(mcmc.logPi, mcmc.preds, mcmc.chain, "identity")
# writer.write_outfiles()


data = simpledata_lm(4, 0.01, 10)
forwardop = ISWTOperator(data, 0.01, 10, 1.5, 2)
params = PxMCMCParams(
    nsamples=int(1e6), nburn=0, ngap=0, nparams=forwardop.nparams, complex=True, delta=1e-7, lmda=3e-7, mu=1e-7
)
mcmc = PxMCMC(forwardop, params)
mcmc.mcmc()

writer = Outfile(mcmc.logPi, mcmc.preds, mcmc.chain, "ISWT")
writer.write_outfiles()
