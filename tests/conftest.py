import pytest
import numpy as np
import healpy as hp
import pys2let

from pxmcmc.mcmc import PxMCMC
from pxmcmc.forward import ForwardOperator, ISWTOperator, SWC2PixOperator
from pxmcmc.prox import L1


@pytest.fixture
def Nside():
    return 32


@pytest.fixture
def sig_d():
    return 1


@pytest.fixture
def L():
    return 10


@pytest.fixture
def B():
    return 1.5


@pytest.fixture
def J_min():
    return 2


@pytest.fixture(params=["analysis", "synthesis"])
def setting(request):
    return request.param


@pytest.fixture
def simpledata(Nside, sig_d):
    return np.ones(hp.nside2npix(Nside))


@pytest.fixture
def simpledata_lm(simpledata, L, B, J_min):
    return pys2let.lm_hp2lm(hp.map2alm(simpledata, L), L)


@pytest.fixture
def forwardop(simpledata, sig_d, setting):
    return ForwardOperator(simpledata, sig_d, setting)


@pytest.fixture
def iswtoperator(simpledata_lm, sig_d, L, B, J_min, setting):
    return ISWTOperator(simpledata_lm, sig_d, L, B, J_min, setting)


@pytest.fixture
def mcmc(forwardop):
    return PxMCMC(forwardop)


@pytest.fixture
def swc2pixoperator(simpledata, sig_d, Nside, L, B, J_min, setting):
    return SWC2PixOperator(simpledata, sig_d, Nside, L, B, J_min, setting)


@pytest.fixture
def L1regulariser(setting):
    T = 50

    def identity(X):
        return np.matmul(np.eye(100), X)

    return L1(setting, identity, identity, T)
