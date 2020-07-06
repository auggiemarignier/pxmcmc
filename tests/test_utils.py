from pxmcmc import utils
import numpy as np
import pytest


def test_flattenmlm():
    f_wav_lm = np.ones((861, 9))
    for i in range(f_wav_lm.shape[1]):
        f_wav_lm[:, i] += i
    f_scal_lm = np.zeros((861,))
    expected = np.concatenate([[i] * 861 for i in range(10)])
    assert all(utils.flatten_mlm(f_wav_lm, f_scal_lm) == expected)


def test_expandmlm():
    mlm = np.ones((8610,))
    f_wav_lm, f_scal_lm = utils.expand_mlm(mlm, 9)
    assert f_wav_lm.shape == (861, 9)
    assert f_scal_lm.shape == (861,)


@pytest.mark.parametrize(
    "ins,thresh,outs",
    [
        ([1, 2, 3], 2, [0, 0, 1]),
        ([-1, -2, -3], 2, [0, 0, -1]),
        ([1 + 1j, 0.5 - 0.5j, 0], 1, [(1 + 1j) * (np.sqrt(2) - 1) / np.sqrt(2), 0, 0]),
    ],
)
def test_soft(ins, thresh, outs):
    assert all(utils.soft(ins, T=thresh) == outs)


@pytest.mark.parametrize(
    "ins,thresh,outs", [(np.arange(1, 11), 0.3, [0, 0, 0, 0, 0, 0, 0, 8, 9, 10])]
)
def test_hard(ins, thresh, outs):
    assert all(utils.hard(ins, T=thresh) == outs)


@pytest.mark.parametrize("order,X,expected", [(0, 5, 1), (1, 2, 2), (5, 3, 3363)])
def test_chebyshev1(order, X, expected):
    assert utils.chebyshev1(X, order=order) == expected


@pytest.mark.parametrize("order,X,expected", [(0, 5, 1), (1, 2, 4), (5, 3, 6930)])
def test_chebyshev2(order, X, expected):
    assert utils.chebyshev2(X, order=order) == expected


@pytest.mark.parametrize("order,X,expected", [(0, 5, 0), (1, 2, 1), (5, 3, 5945)])
def test_cheb1der(order, X, expected):
    assert utils.cheb1der(X, order=order) == expected


def test_formatter_pix2pix(waveletformatter, simpledata_hp):
    mw = waveletformatter._pixhp2pixmw(simpledata_hp)
    hp_rec = waveletformatter._pixmw2pixhp(mw)
    assert np.allclose(simpledata_hp, hp_rec)
    mw_rec = waveletformatter._pixhp2pixmw(hp_rec)
    assert np.allclose(mw, mw_rec)


def test_formatter_harmmw2pixmw_wavelets(waveletformatter, simpledata_lm, simpledata):
    scal_lm = np.copy(simpledata_lm)
    wav_lm = np.column_stack([simpledata_lm for _ in range(waveletformatter.nscales)])
    scal_mw, wav_mw = waveletformatter._harmmw2pixmw_wavelets(scal_lm, wav_lm)
    assert np.allclose(scal_mw, simpledata)
    for j in range(waveletformatter.nscales):
        assert np.allclose(wav_mw[:, j], simpledata)


def test_formatter_harmhp2pixhp_wavelets(
    waveletformatter, simpledata_hp_lm, simpledata_hp
):
    scal_lm = np.copy(simpledata_hp_lm)
    wav_lm = np.column_stack(
        [simpledata_hp_lm for _ in range(waveletformatter.nscales)]
    )
    scal_mw, wav_mw = waveletformatter._harmhp2pixhp_wavelets(scal_lm, wav_lm)
    assert np.allclose(scal_mw, simpledata_hp)
    for j in range(waveletformatter.nscales):
        assert np.allclose(wav_mw[:, j], simpledata_hp)


def test_harmhp2harmmw_wavelets(waveletformatter, simpledata_hp_lm):
    scal_hp_lm = np.copy(simpledata_hp_lm)
    wav_hp_lm = np.column_stack(
        [simpledata_hp_lm for _ in range(waveletformatter.nscales)]
    )
    scal_lm, wav_lm = waveletformatter._harmhp2harmmw_wavelets(scal_hp_lm, wav_hp_lm)
    assert scal_lm.shape == (waveletformatter.L ** 2,)
    assert scal_lm.dtype == np.complex
    assert wav_lm.shape == (waveletformatter.L ** 2, waveletformatter.nscales)
    assert wav_lm.dtype == np.complex
