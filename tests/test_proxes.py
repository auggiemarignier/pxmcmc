import numpy as np
import pytest

from pxmcmc.prior import L1, S2_Wavelets_L1


@pytest.fixture
def L1regulariser(setting):
    T = 50

    def identity(X):
        return np.matmul(np.eye(100), X)

    return L1(setting, identity, identity, T)


def test_L1(L1regulariser):
    from pxmcmc.utils import soft

    X = np.arange(100)
    assert np.alltrue(L1regulariser.proxf(X) == soft(X, L1regulariser.T))


@pytest.fixture
def S2_Wavs_L1reg(setting, L, B, J_min):
    reg = S2_Wavelets_L1(setting, None, None, 0.5, L, B, J_min)

    def identity(X):
        if setting == "analysis":
            return np.matmul(np.eye(L * (2 * L - 1)), X)
        else:
            return np.matmul(np.eye(reg.nscales * L * (2 * L - 1)), X)

    reg.fwd = identity
    reg.adj = identity
    return reg


def test_S2_Wavlets_L1(S2_Wavs_L1reg, simpledata):
    """
    Since the soft thresholding and weighting have been tested individually
    just make sure it runs
    """
    if S2_Wavs_L1reg.setting == "analysis":
        S2_Wavs_L1reg.proxf(simpledata)
    else:
        S2_Wavs_L1reg.proxf(
            np.concatenate([simpledata for _ in range(S2_Wavs_L1reg.nscales)])
        )
