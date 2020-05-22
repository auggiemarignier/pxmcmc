import pytest
import numpy as np
import healpy as hp
import pys2let

from pxmcmc.mcmc import PxMCMC
from pxmcmc.forward import ForwardOperator, ISWTOperator


@pytest.fixture
def Nside():
    return 32


@pytest.fixture
def sig_d():
    return 0.01


@pytest.fixture
def L():
    return 10


@pytest.fixture
def B():
    return 1.5


@pytest.fixture
def J_min():
    return 2


@pytest.fixture
def simpledata(Nside, sig_d):
    return np.ones(hp.nside2npix(Nside))


@pytest.fixture
def simpledata_lm(simpledata, L, B, J_min):
    return pys2let.lm_hp2lm(hp.map2alm(simpledata, L), L + 1)


@pytest.fixture
def forwardop(simpledata, sig_d):
    return ForwardOperator(simpledata, sig_d)


@pytest.fixture
def iswtoperator(simpledata_lm, sig_d):
    return ISWTOperator(simpledata_lm, sig_d, 10, 1.5, 2)


@pytest.fixture
def mcmc(forwardop):
    return PxMCMC(forwardop)

