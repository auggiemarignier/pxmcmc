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


def test_ISWTForceTiling(iswtoperator):

    L = iswtoperator.L
    B = iswtoperator.B
    J_min = iswtoperator.J_min
    X = np.ones(iswtoperator.nparams) + 1j * np.ones(iswtoperator.nparams)
    phi_l, psi_lm = pys2let.wavelet_tiling(
        B, L + 1, 1, 0, J_min
    )  # phi_l = 0, bug in pys2let?
    psi_lm = psi_lm[:, J_min:]
    phi_lm = np.zeros(((L + 1) ** 2, 1), dtype=np.complex)
    for ell in range(L + 1):
        phi_lm[ell * ell + ell] = phi_l[ell]
    basis = np.concatenate((phi_lm, psi_lm), axis=1)
    expected = X * basis.flatten()

    assert np.array_equal(iswtoperator.force_tiling(X), expected)


def test_ISWTForward(iswtoperator, Nside):
    from pxmcmc.utils import flatten_mlm

    L = iswtoperator.L
    B = iswtoperator.B
    J_min = iswtoperator.J_min
    f = np.ones(hp.nside2npix(Nside))
    flm_hp = hp.map2alm(f, L)
    f_wav_lm_hp, f_scal_lm_hp = pys2let.analysis_axisym_lm_wav(flm_hp, B, L + 1, J_min)
    f_wav_lm = np.zeros(((L + 1) ** 2, f_wav_lm_hp.shape[1]))
    for j in range(iswtoperator.nscales):
        f_wav_lm[:, j] = pys2let.lm_hp2lm(
            np.ascontiguousarray(f_wav_lm_hp[:, j]), L + 1
        )
    f_scal_lm = pys2let.lm_hp2lm(f_scal_lm_hp, L + 1)
    X = flatten_mlm(f_wav_lm, f_scal_lm)

    flm = pys2let.lm_hp2lm(flm_hp, L + 1)
    assert np.allclose(iswtoperator.forward(X), flm)


def test_ISWTGradg(iswtoperator):
    iswtoperator.sig_d = 1
    iswtoperator.pf = np.ones(iswtoperator.nparams)
    preds = np.ones(len(iswtoperator.data))
    expected = np.concatenate([1 - iswtoperator.data] * iswtoperator.basis.shape[1])
    assert np.allclose(iswtoperator.calc_gradg(preds), expected)
