from pxmcmc import utils
import numpy as np
import pytest
import pys2let
import pyssht
from s2fft import sampling


def test_flattenmlm():
    f_wav_lm = np.ones((861, 9))
    for i in range(f_wav_lm.shape[1]):
        f_wav_lm[:, i] += i
    f_scal_lm = np.zeros((861,))
    expected = np.concatenate([[i] * 861 for i in range(10)])
    assert all(utils.flatten_mlm(f_wav_lm, f_scal_lm) == expected)


def test_expandmlm():
    mlm = np.ones((8610,))
    f_wav_lm, f_scal_lm = utils.expand_mlm(mlm, nscales=9)
    assert f_wav_lm.shape == (861, 9)
    assert f_scal_lm.shape == (861,)


def test_flatten_expand_multires(simpledata, L, B, J_min):
    f_mw = simpledata.astype(complex)
    f_wav, f_scal = pys2let.analysis_px2wav(f_mw, B, L, J_min, 1, 0, upsample=0)
    f_scalwav = utils.flatten_mlm(f_wav, f_scal)
    f_wav_expanded, f_scal_expanded = utils.expand_mlm(
        f_scalwav, nscalcoefs=len(f_scal)
    )
    assert np.array_equal(f_scal, f_scal_expanded)
    assert np.array_equal(f_wav, f_wav_expanded)


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


def test_pixel_area():
    area = utils.pixel_area(1, 0, np.pi, 0, 2 * np.pi)
    assert np.isclose(area, 4 * np.pi)


@pytest.mark.parametrize("alpha, area", [(np.pi / 2, 2 * np.pi), (np.pi, 4 * np.pi)])
def test_polar_cap_area(alpha, area):
    cap = utils.polar_cap_area(1, alpha)
    assert np.isclose(cap, area)


def test_calc_pixel_areas(L):
    areas = utils.calc_pixel_areas(L)
    assert np.isclose(np.sum(areas), 4 * np.pi)


def test_s2_integrate(L):

    flm = np.zeros((L * L), dtype=complex)
    for el in range(L):
        m = 0
        ind = sampling.elm2ind(el, m)
        flm[ind] = np.random.randn()
        for m in range(1, el + 1):
            ind_pm = sampling.elm2ind(el, m)
            ind_nm = sampling.elm2ind(el, -m)
            flm[ind_pm] = np.random.randn() + 1j * np.random.randn()
            flm[ind_nm] = (-1) ** m * np.conj(flm[ind_pm])
    I0 = flm[0] * np.sqrt(4 * np.pi)
    f = pyssht.inverse(flm, L, Reality=True).flatten()

    assert np.isclose(I0, utils.s2_integrate(f, L))
