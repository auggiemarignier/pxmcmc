import numpy as np
import pytest

from pxmcmc.prior import L1, S2_Wavelets_L1, S2_Wavelets_L1_Power_Weights


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


@pytest.mark.parametrize(
    "setting",
    [
        "synthesis",
        pytest.param(
            "analysis",
            marks=pytest.mark.xfail(reason="Analysis prox not yet implemented"),
        ),
    ],
)
def test_S2_Wavlets_L1(setting, L, B, J_min):
    """
    Since the soft thresholding and weighting have been tested individually
    just make sure it runs
    """
    reg = S2_Wavelets_L1(setting, None, None, 1, L, B, J_min)

    def identity(X):
        return np.matmul(np.eye(reg.map_weights.size), X)

    reg.fwd = identity
    reg.adj = identity

    data = np.ones(reg.map_weights.size)
    reg.proxf(data)


@pytest.mark.parametrize(
    "setting",
    [
        "synthesis",
        pytest.param(
            "analysis",
            marks=pytest.mark.xfail(reason="Analysis prox not yet implemented"),
        ),
    ],
)
def test_S2_Wavelets_L1_Power_Weights(setting, L, B, J_min):
    """
    Since the soft thresholding and weighting have been tested individually
    just make sure it runs
    """
    reg = S2_Wavelets_L1_Power_Weights(setting, None, None, 1, L, B, J_min, eta=1)

    def identity(X):
        return np.matmul(np.eye(reg.map_weights.size), X)

    reg.fwd = identity
    reg.adj = identity

    data = np.ones(reg.map_weights.size)
    reg.proxf(data)
