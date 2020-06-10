import numpy as np


def test_L1(L1regulariser):
    from pxmcmc.utils import soft

    X = np.arange(100)
    assert np.alltrue(L1regulariser.proxf(X) == soft(X, L1regulariser.T))
