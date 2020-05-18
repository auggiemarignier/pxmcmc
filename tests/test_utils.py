from pxmcmc import utils
import numpy as np
import pytest


def test_flattenmlm():
    f_wav_lm = np.ones((861, 9))
    f_scal_lm = np.ones((861,))
    assert all(utils.flatten_mlm(f_wav_lm, f_scal_lm) == np.ones((8610,)))


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
