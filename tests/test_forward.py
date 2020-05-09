import numpy as np


def test_BaseForward(forwardop):
    mcmc_sample = np.ones(forwardop.data.shape)
    assert np.allclose(forwardop.forward(mcmc_sample), forwardop.data, atol=1e-1)


def test_BaseGradg(forwardop):
    mcmc_sample = np.ones(forwardop.data.shape)
    forwardop.sig_d = 1  # hack
    preds = forwardop.forward(mcmc_sample)
    gradg = forwardop.calc_gradg(preds)
    assert np.allclose(gradg, np.zeros(forwardop.data.shape), atol=1e-1)
