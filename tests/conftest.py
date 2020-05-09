import pytest
import numpy as np
import healpy as hp

from pxmcmc.mcmc import PxMCMC
from pxmcmc.forward import ForwardOperator


@pytest.fixture
def Nside():
    return 32


@pytest.fixture
def sig_d():
    return 0.01


@pytest.fixture
def simpledata(Nside, sig_d):
    simple = np.ones(hp.nside2npix(Nside))
    noise = np.random.normal(scale=sig_d, size=simple.shape)
    return simple + noise


@pytest.fixture
def forwardop(simpledata, sig_d):
    return ForwardOperator(simpledata, sig_d)


@pytest.fixture
def mcmc(forwardop):
    return PxMCMC(forwardop)

