import pytest
from pytest_cases import parametrize_with_cases, fixture
import numpy as np
import s2wav

from pxmcmc.utils import mw_sample_length


@pytest.fixture
def Nside():
    return 32


@pytest.fixture
def L():
    return 10


@pytest.fixture
def B():
    return 2


@pytest.fixture
def J_min():
    return 2


@pytest.fixture(params=["analysis", "synthesis"])
def setting(request):
    return request.param


@pytest.fixture
def simpledata_lm(L):
    data = np.zeros(L * L, dtype=complex)
    for el in range(L):
        em = 0
        while em <= el:
            rand = np.asarray(np.random.rand(), dtype=complex)
            data[el * el + el - em] = pow(-1.0, -em) * rand.conjugate()
            data[el * el + el + em] = rand
            em += 1
    return data


@pytest.fixture
def simpledata(simpledata_lm, L):
    return pys2let.alm2map_mw(simpledata_lm, L, 0).real


def case_sig_d_int():
    return 0.1


def case_sig_d_array():
    return np.full(mw_sample_length(10), 0.1)


@fixture
@parametrize_with_cases("format", cases=[case_sig_d_int, case_sig_d_array])
def sig_d(format):
    return format
