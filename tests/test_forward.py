import pys2let
import pytest
import numpy as np
import healpy as hp
from pxmcmc.forward import ForwardOperator

Nside = 32
sig_d = 0.01


@pytest.fixture
def simpledata():
    simple = np.ones(hp.nside2npix(Nside))
    noise = np.random.normal(scale=sig_d, size=simple.shape)
    return simple + noise


def test_BaseForward(simpledata):
    fwd = ForwardOperator(simpledata, sig_d)
    mcmc_sample = np.ones(simpledata.shape)
    assert np.allclose(fwd.forward(mcmc_sample), simpledata, atol=1e-1)


def test_BaseGradg(simpledata):
    fwd = ForwardOperator(simpledata, sig_d=1)
    mcmc_sample = np.ones(simpledata.shape)
    preds = fwd.forward(mcmc_sample)
    gradg = fwd.calc_gradg(preds)
    assert np.allclose(gradg, np.zeros(simpledata.shape), atol=1e-1)