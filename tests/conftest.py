import pytest
from pytest_cases import parametrize_with_cases, fixture
import numpy as np
import pys2let

from pxmcmc.forward import WaveletTransformOperator
from pxmcmc.prox import L1
from pxmcmc.transforms import WaveletTransform
from pxmcmc.utils import alm2map, WaveletFormatter


@pytest.fixture
def Nside():
    return 32

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
def all_data(
    simpledata_lm, simpledata_hp_lm, simpledata, simpledata_hp,
):
    data = {
        "harmonic_mw": simpledata_lm,
        "harmonic_hp": simpledata_hp_lm,
        "pixel_mw": simpledata,
        "pixel_hp": simpledata_hp,
    }
    return data


def case_sig_d_int():
    return 1


def case_sig_d_array():
    return np.ones(pys2let.mw_size(10))


@fixture
@parametrize_with_cases("format", cases=[case_sig_d_int, case_sig_d_array])
def sig_d(format):
    return format


@pytest.fixture
def swtoperator(simpledata, sig_d, L, B, J_min, setting):
    return WaveletTransformOperator(simpledata, sig_d, setting, L, B, J_min)


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
