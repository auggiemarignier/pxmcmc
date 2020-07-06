import pytest
import numpy as np
import pys2let

from pxmcmc.mcmc import PxMCMC
from pxmcmc.forward import ForwardOperator, ISWTOperator, SWC2PixOperator
from pxmcmc.prox import L1
from pxmcmc.transforms import WaveletTransform
from pxmcmc.utils import alm2map, WaveletFormatter


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
def simpledata_lm(L):
    data = np.zeros(L * L, dtype=np.complex)
    for el in range(L):
        em = 0
        while em <= el:
            rand = np.asarray(np.random.rand(), dtype=np.complex)
            data[el * el + el - em] = pow(-1.0, -em) * rand.conjugate()
            data[el * el + el + em] = rand
            em += 1
    return data


@pytest.fixture
def simpledata(simpledata_lm, L):
    return pys2let.alm2map_mw(simpledata_lm, L, 0)


@pytest.fixture
def simpledata_hp_lm(L):
    return np.random.rand(L * (L + 1) // 2).astype(np.complex)


@pytest.fixture
def simpledata_hp(simpledata_hp_lm, Nside):
    return alm2map(simpledata_hp_lm, Nside)


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


@pytest.fixture
def waveletformatter(L, B, J_min, Nside):
    return WaveletFormatter(L, B, J_min, Nside)


@pytest.fixture
def wvlttransform(L, B, J_min, Nside):
    return WaveletTransform(L, B, J_min, Nside)
