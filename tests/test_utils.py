from pxmcmc import utils
import numpy as np
import healpy as hp
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


@pytest.mark.parametrize("start,stop,point", [((0, 0), (90, 0), (45, 0))])
def test_point_on_gcp(start, stop, point, Nside):
    path = utils.GreatCirclePath(start, stop, Nside)
    path._get_points()
    assert point in path.points


@pytest.mark.parametrize("start,stop", [((60, 60), (0, 0))])
def test_pixels_on_gcp(start, stop, Nside):
    path = utils.GreatCirclePath(start, stop, Nside)
    path.fill()
    assert all(pix == 0 or pix == 1 for pix in path.map)


@pytest.mark.parametrize("start,stop,course", [((-33, -71.6), (31.4, 121.8), -94.41)])
def test_gcp_course(start, stop, course, Nside):
    path = utils.GreatCirclePath(start, stop, Nside)
    assert np.round(np.rad2deg(path._course()), 2) == course
