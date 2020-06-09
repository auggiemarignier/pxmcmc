import numpy as np
import healpy as hp
import pys2let
import pytest


def test_BaseForward(forwardop):
    mcmc_sample = np.ones(forwardop.data.shape)
    with pytest.raises(NotImplementedError):
        forwardop.forward(mcmc_sample)


def test_BaseGradg(forwardop):
    preds = np.ones(forwardop.data.shape)
    with pytest.raises(NotImplementedError):
        forwardop.calc_gradg(preds)


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


def test_ISWTGradg(iswtoperator):
    # TODO: Come up with a better test; this just tests implementation
    from pxmcmc.utils import flatten_mlm

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
        vlen = pys2let.mw_size(iswtoperator.L + 1)
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


def test_SWC2PixGradg(swc2pixoperator):
    from pxmcmc.utils import flatten_mlm

    L = swc2pixoperator.L
    B = swc2pixoperator.B
    J_min = swc2pixoperator.J_min
    Nside = swc2pixoperator.Nside

    preds = 1 + swc2pixoperator.data
    if swc2pixoperator.setting == "analysis":
        expected = np.ones(hp.nside2npix(Nside), dtype=np.complex)
    else:
        diff = np.ones(pys2let.mw_size(L + 1), dtype=np.complex)
        f_wav, f_scal = pys2let.synthesis_adjoint_axisym_wav_mw(diff, B, L + 1, J_min)
        f_scal_lm = pys2let.map2alm_mw(f_scal, L + 1, 0)
        f_wav_lm = np.zeros(((L + 1) ** 2, swc2pixoperator.nscales), dtype=np.complex)
        vlen = pys2let.mw_size(L + 1)
        for j in range(swc2pixoperator.nscales):
            f_wav_lm[:, j] = pys2let.map2alm_mw(
                f_wav[j * vlen : (j + 1) * vlen + 1], L + 1, 0
            )
        expected = flatten_mlm(f_wav_lm, f_scal_lm)

    result = swc2pixoperator.calc_gradg(preds)
    assert np.allclose(result, expected)
