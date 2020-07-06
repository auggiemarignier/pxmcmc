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


def test_formatter_pix2pix(waveletformatter, simpledata):
    mw = waveletformatter._pixhp2pixmw(simpledata)
    hp_rec = waveletformatter._pixmw2pixhp(mw)
    assert np.allclose(simpledata, hp_rec)
    mw_rec = waveletformatter._pixhp2pixmw(hp_rec)
    assert np.allclose(mw, mw_rec)
