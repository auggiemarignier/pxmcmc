from pxmcmc import utils
import numpy as np
import pytest
import pys2let
import pyssht


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


def test_flatten_expand_multires(L, B, J_min):
    f_mw = np.empty(pyssht.sample_length(L), dtype=complex)
    f_wav, f_scal = pys2let.analysis_px2wav(f_mw, B, L, J_min, 1, 0, upsample=0)
    f_scalwav = utils.flatten_mlm(f_wav, f_scal)
    f_scal_expanded, f_wav_expanded = utils.expand_mlm(
        f_scalwav, nscal=len(f_scal), nwav=len(f_wav)
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


@pytest.fixture
def waveletformatter(L, B, J_min, Nside):
    return utils.WaveletFormatter(L, B, J_min, Nside)


def test_formatter_pix2pix(waveletformatter, simpledata_hp):
    mw = waveletformatter._pixhp2pixmw(simpledata_hp)
    hp_rec = waveletformatter._pixmw2pixhp(mw)
    assert np.allclose(simpledata_hp, hp_rec)
    mw_rec = waveletformatter._pixhp2pixmw(hp_rec)
    assert np.allclose(mw, mw_rec)


def test_formatter_harmmw2pixmw_wavelets(waveletformatter, simpledata_lm, simpledata):
    scal_lm = np.copy(simpledata_lm)
    wav_lm = np.concatenate([simpledata_lm for _ in range(waveletformatter.nscales)])
    scal_mw, wav_mw = waveletformatter._harmmw2pixmw_wavelets(scal_lm, wav_lm)
    assert np.allclose(scal_mw, simpledata)
    assert np.allclose(
        wav_mw, np.concatenate([simpledata for _ in range(waveletformatter.nscales)])
    )


def test_formatter_harmhp2pixhp_wavelets(
    waveletformatter, simpledata_hp_lm, simpledata_hp
):
    scal_lm = np.copy(simpledata_hp_lm)
    wav_lm = np.concatenate([simpledata_hp_lm for _ in range(waveletformatter.nscales)])
    scal_hp, wav_hp = waveletformatter._harmhp2pixhp_wavelets(scal_lm, wav_lm)
    assert np.allclose(scal_hp, simpledata_hp)
    assert np.allclose(
        wav_hp, np.concatenate([simpledata_hp for _ in range(waveletformatter.nscales)])
    )


def test_harmhp2harmmw_wavelets(waveletformatter, simpledata_hp_lm):
    scal_hp_lm = np.copy(simpledata_hp_lm)
    wav_hp_lm = np.concatenate(
        [simpledata_hp_lm for _ in range(waveletformatter.nscales)]
    )
    scal_lm, wav_lm = waveletformatter._harmhp2harmmw_wavelets(scal_hp_lm, wav_hp_lm)
    assert scal_lm.shape == (waveletformatter.L ** 2,)
    assert scal_lm.dtype == np.complex
    assert wav_lm.shape == (waveletformatter.L ** 2 * waveletformatter.nscales,)
    assert wav_lm.dtype == np.complex


def test_harmmw2harmhp_wavelets(waveletformatter, simpledata_lm):
    scal_lm = np.copy(simpledata_lm)
    wav_lm = np.concatenate([simpledata_lm for _ in range(waveletformatter.nscales)])
    scal_hp_lm, wav_hp_lm = waveletformatter._harmmw2harmhp_wavelets(scal_lm, wav_lm)
    assert scal_hp_lm.shape == (waveletformatter.L * (waveletformatter.L + 1) // 2,)
    assert scal_hp_lm.dtype == np.complex
    assert wav_hp_lm.shape == (
        waveletformatter.L * (waveletformatter.L + 1) * waveletformatter.nscales // 2,
    )
    assert wav_hp_lm.dtype == np.complex


def test_formatter_harm2harm_wavelets(waveletformatter, simpledata_lm):
    scal_lm = np.copy(simpledata_lm)
    wav_lm = np.concatenate([simpledata_lm for _ in range(waveletformatter.nscales)])
    scal_hp_lm, wav_hp_lm = waveletformatter._harmmw2harmhp_wavelets(scal_lm, wav_lm)
    scal_lm_rec, wav_lm_rec = waveletformatter._harmhp2harmmw_wavelets(
        scal_hp_lm, wav_hp_lm
    )
    assert np.allclose(scal_lm_rec, scal_lm)
    assert np.allclose(wav_lm_rec, wav_lm)
    scal_hp_lm_rec, wav_hp_lm_rec = waveletformatter._harmmw2harmhp_wavelets(
        scal_lm_rec, wav_lm_rec
    )
    assert np.allclose(scal_hp_lm_rec, scal_hp_lm)
    assert np.allclose(wav_hp_lm_rec, wav_hp_lm)


def test_harmhp2pixmw_wavelets(waveletformatter, simpledata_hp_lm):
    scal_hp_lm = np.copy(simpledata_hp_lm)
    wav_hp_lm = np.concatenate(
        [simpledata_hp_lm for _ in range(waveletformatter.nscales)]
    )
    scal_lm, wav_lm = waveletformatter._harmhp2pixmw_wavelets(scal_hp_lm, wav_hp_lm)
    assert scal_lm.shape == (pys2let.mw_size(waveletformatter.L),)
    assert scal_lm.dtype == np.complex
    assert wav_lm.shape == (
        pys2let.mw_size(waveletformatter.L) * waveletformatter.nscales,
    )
    assert wav_lm.dtype == np.complex


def test_pixmw2harmhp_wavelets(waveletformatter, simpledata):
    scal = np.copy(simpledata)
    wav = np.concatenate([simpledata for _ in range(waveletformatter.nscales)])
    scal_hp_lm, wav_hp_lm = waveletformatter._pixmw2harmhp_wavelets(scal, wav)
    assert scal_hp_lm.shape == (waveletformatter.L * (waveletformatter.L + 1) // 2,)
    assert scal_hp_lm.dtype == np.complex
    assert wav_hp_lm.shape == (
        waveletformatter.L * (waveletformatter.L + 1) * waveletformatter.nscales // 2,
    )
    assert wav_hp_lm.dtype == np.complex


def test_pixhp2harmhp_wavelets(waveletformatter, simpledata_hp):
    scal = np.copy(simpledata_hp)
    wav = np.concatenate([simpledata_hp for _ in range(waveletformatter.nscales)])
    scal_hp_lm, wav_hp_lm = waveletformatter._pixhp2harmhp_wavelets(scal, wav)
    assert scal_hp_lm.shape == (waveletformatter.L * (waveletformatter.L + 1) // 2,)
    assert scal_hp_lm.dtype == np.complex
    assert wav_hp_lm.shape == (
        waveletformatter.L * (waveletformatter.L + 1) * waveletformatter.nscales // 2,
    )
    assert wav_hp_lm.dtype == np.complex


def test_formatter_pixmw2harmhp2pixmw_wavelets(waveletformatter, simpledata):
    scal = np.copy(simpledata)
    wav = np.concatenate([simpledata for _ in range(waveletformatter.nscales)])
    scal_hp_lm, wav_hp_lm = waveletformatter._pixmw2harmhp_wavelets(scal, wav)
    scal_rec, wav_rec = waveletformatter._harmhp2pixmw_wavelets(scal_hp_lm, wav_hp_lm)
    assert np.allclose(scal_rec, scal)
    assert np.allclose(wav_rec, wav)
    scal_hp_lm_rec, wav_hp_lm_rec = waveletformatter._pixmw2harmhp_wavelets(
        scal_rec, wav_rec
    )
    assert np.allclose(scal_hp_lm_rec, scal_hp_lm)
    assert np.allclose(wav_hp_lm_rec, wav_hp_lm)


def test_formatter_pixmw2harmhp2pixmw(waveletformatter, simpledata):
    hp_lm = waveletformatter._pixmw2harmhp(simpledata)
    mw_rec = waveletformatter._harmhp2pixmw(hp_lm)
    assert np.allclose(mw_rec, simpledata)


@pytest.mark.parametrize(
    "datatype", ["harmonic_mw", "harmonic_hp", "pixel_mw", "pixel_hp"]
)
def test_formatter_splitflatten_wavelets(
    datatype,
    waveletformatter,
    simpledata_lm,
    simpledata_hp_lm,
    simpledata,
    simpledata_hp,
):
    data = {
        "harmonic_mw": simpledata_lm,
        "harmonic_hp": simpledata_hp_lm,
        "pixel_mw": simpledata,
        "pixel_hp": simpledata_hp,
    }
    wav = np.concatenate([data[datatype] for _ in range(waveletformatter.nscales)])
    wav_s = waveletformatter._split_wavelets(wav)
    wav_rec = waveletformatter._flatten_wavelets(wav_s)
    assert np.array_equal(wav_rec, wav)


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
        ind = pyssht.elm2ind(el, m)
        flm[ind] = np.random.randn()
        for m in range(1, el + 1):
            ind_pm = pyssht.elm2ind(el, m)
            ind_nm = pyssht.elm2ind(el, -m)
            flm[ind_pm] = np.random.randn() + 1j * np.random.randn()
            flm[ind_nm] = (-1) ** m * np.conj(flm[ind_pm])
    I0 = flm[0] * np.sqrt(4 * np.pi)
    f = pyssht.inverse(flm, L, Method="MW", Reality=True).flatten()

    assert np.isclose(I0, utils.s2_integrate(f, L))
