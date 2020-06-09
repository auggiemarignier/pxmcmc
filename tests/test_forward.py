import numpy as np
import healpy as hp
import pys2let
from scipy.special import sph_harm
import pytest


def test_BaseForward(forwardop):
    mcmc_sample = np.ones(forwardop.data.shape)
    with pytest.raises(NotImplementedError):
        forwardop.forward(mcmc_sample)


def test_BaseGradg(forwardop):
    preds = np.ones(forwardop.data.shape)
    with pytest.raises(NotImplementedError):
        forwardop.calc_gradg(preds)


@pytest.mark.parametrzie("setting", ["analysis", "synthesis"], indirect=["setting"])
def test_ISWTForward(iswtoperator, Nside):
    from pxmcmc.utils import flatten_mlm

    L = iswtoperator.L
    B = iswtoperator.B
    J_min = iswtoperator.J_min
    f = np.ones(hp.nside2npix(Nside))
    flm_hp = hp.map2alm(f, L)
    flm = pys2let.lm_hp2lm(flm_hp, L + 1)

    if iswtoperator.setting == "synthesis":
        f_wav_lm_hp, f_scal_lm_hp = pys2let.analysis_axisym_lm_wav(
            flm_hp, B, L + 1, J_min
        )
        f_wav_lm = np.zeros(((L + 1) ** 2, f_wav_lm_hp.shape[1]), dtype=np.complex)
        for j in range(iswtoperator.nscales):
            f_wav_lm[:, j] = pys2let.lm_hp2lm(
                np.ascontiguousarray(f_wav_lm_hp[:, j]), L + 1
            )
        f_scal_lm = pys2let.lm_hp2lm(f_scal_lm_hp, L + 1)
        X = flatten_mlm(f_wav_lm, f_scal_lm)
    else:
        X = np.copy(f)

    assert np.allclose(iswtoperator.forward(X), flm)


@pytest.mark.parametrzie("setting", ["analysis", "synthesis"], indirect=["setting"])
def test_ISWTGradg(iswtoperator):
    # TODO: Come up with a better test; this just tests implementation
    from pxmcmc.utils import flatten_mlm

    iswtoperator.sig_d = 1
    preds = np.ones(len(iswtoperator.data))
    diff = preds - iswtoperator.data

    if iswtoperator.setting == "synthesis":
        f = pys2let.alm2map_mw(diff, iswtoperator.L + 1, 0)
        f_wav, f_scal = pys2let.synthesis_adjoint_axisym_wav_mw(
            f, iswtoperator.B, iswtoperator.L + 1, iswtoperator.J_min
        )
        f_scal_lm = pys2let.map2alm_mw(f_scal, iswtoperator.L + 1, 0)
        f_wav_lm = np.zeros(
            [(iswtoperator.L + 1) ** 2, iswtoperator.nscales], dtype=np.complex
        )
        vlen = (iswtoperator.L + 1) * (2 * (iswtoperator.L + 1) - 1)
        for j in range(iswtoperator.nscales):
            f_wav_lm[:, j] = pys2let.map2alm_mw(
                f_wav[j * vlen : (j + 1) * vlen + 1], iswtoperator.L + 1, 0
            )
        expected = flatten_mlm(f_wav_lm, f_scal_lm)
    else:
        expected = np.copy(diff)
    result = iswtoperator.calc_gradg(preds)
    assert np.allclose(result, expected)


def test_SWC2PixForward(swc2pixoperator):
    from pxmcmc.utils import flatten_mlm

    L = swc2pixoperator.L
    B = swc2pixoperator.B
    J_min = swc2pixoperator.J_min
    Nside = swc2pixoperator.Nside
    f = np.ones(hp.nside2npix(Nside))
    flm_hp = hp.map2alm(f, L)
    f_wav_lm_hp, f_scal_lm_hp = pys2let.analysis_axisym_lm_wav(flm_hp, B, L + 1, J_min)
    f_wav_lm = np.zeros(((L + 1) ** 2, f_wav_lm_hp.shape[1]), dtype=np.complex)
    for j in range(swc2pixoperator.nscales):
        f_wav_lm[:, j] = pys2let.lm_hp2lm(
            np.ascontiguousarray(f_wav_lm_hp[:, j]), L + 1
        )
    f_scal_lm = pys2let.lm_hp2lm(f_scal_lm_hp, L + 1)
    if swc2pixoperator.setting == "synthesis":
        X = flatten_mlm(f_wav_lm, f_scal_lm)
    else:
        X = np.copy(f)

    assert np.allclose(swc2pixoperator.forward(X), f)


# @pytest.mark.parametrize("l,m", [(0, 0), (1, -1), (1, 0), (1, 1)])
# def test_SWC2PixGradg(swc2pixoperator, l, m):
#     swc2pixoperator.sid_d = 1
#     swc2pixoperator.pf = np.ones(swc2pixoperator.pf.shape)
#     preds = np.ones(len(swc2pixoperator.data))
#     theta, phi = hp.pix2ang(
#         swc2pixoperator.Nside, np.arange(hp.nside2npix(swc2pixoperator.Nside))
#     )
#     expected = np.sum(sph_harm(m, l, phi, theta) * (preds - swc2pixoperator.data))
#     L = swc2pixoperator.L
#     nb = swc2pixoperator.basis.shape[1]
#     lm_idxs = [n * ((L + 1) ** 2) + l ** 2 + l + m for n in range(nb)]
#     gradg = swc2pixoperator.calc_gradg(preds)
#     assert np.allclose(np.take(gradg, lm_idxs), expected)
