import numpy as np
import pytest
from pyssht import sample_length

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
        if setting == "analysis":
            return np.matmul(np.eye(L * (2 * L - 1)), X)
        else:
            return np.matmul(np.eye((reg.nscales + 1) * L * (2 * L - 1)), X)

    reg.fwd = identity
    reg.adj = identity

    data = np.ones(sample_length(reg.L))
    if reg.setting == "analysis":
        reg.proxf(data)
    else:
        reg.proxf(np.concatenate([data for _ in range(reg.nscales + 1)]))
