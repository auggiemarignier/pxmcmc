import numpy as np
import healpy as hp
import pys2let


def test_BaseForward(forwardop):
    mcmc_sample = np.ones(forwardop.data.shape)
    assert np.allclose(forwardop.forward(mcmc_sample), forwardop.data, atol=1e-1)


def test_BaseGradg(forwardop):
    mcmc_sample = np.ones(forwardop.data.shape)
    forwardop.sig_d = 1  # hack
    preds = forwardop.forward(mcmc_sample)
    gradg = forwardop.calc_gradg(preds)
    assert np.allclose(gradg, np.zeros(forwardop.data.shape), atol=1e-1)


def test_ISWTForceTiling(simpledata):
    from pxmcmc.forward import ISWTOperator

    L = 10
    B = 1.5
    J_min = 2
    data = pys2let.lm_hp2lm(hp.map2alm(simpledata, L), L + 1)
    forwardop = ISWTOperator(data, 0.01, L, B, J_min)
    X = np.ones(forwardop.nparams) + 1j * np.ones(forwardop.nparams)
    phi_l, psi_lm = pys2let.wavelet_tiling(
        B, L + 1, 1, 0, J_min
    )  # phi_l = 0, bug in pys2let?
    psi_lm = psi_lm[:, J_min:]
    phi_lm = np.zeros(((L + 1) ** 2, 1), dtype=np.complex)
    for ell in range(L + 1):
        phi_lm[ell * ell + ell] = phi_l[ell]
    basis = np.concatenate((phi_lm, psi_lm), axis=1)
    expected = X * basis.flatten()
    
    print(expected)
    assert np.array_equal(forwardop.force_tiling(X), expected)
