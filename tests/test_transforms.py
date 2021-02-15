import numpy as np
import pytest

from pxmcmc.utils import flatten_mlm
from pxmcmc.transforms import WaveletTransform


@pytest.fixture
def wvlttransform(L, B, J_min, Nside):
    return WaveletTransform(L, B, J_min, Nside)


def test_wavelet_fwdback(wvlttransform, simpledata):
    X_wav = wvlttransform.forward(simpledata)
    data_rec = wvlttransform.inverse(X_wav)
    assert np.allclose(simpledata, data_rec)


def test_wavelet_fwd_adjoint_dot(wvlttransform, simpledata):
    # if y = Ax and g = A'f, show that f'y = g'x
    x = np.copy(simpledata)
    y = wvlttransform.forward(x)

    scal_lm = np.copy(simpledata)
    wav_lm = np.column_stack([simpledata for _ in range(wvlttransform.nscales)])
    f = flatten_mlm(wav_lm, scal_lm)
    g = wvlttransform.forward_adjoint(f)

    dot_diff = f.conj().T.dot(y) - g.conj().T.dot(x)
    assert np.isclose(dot_diff, 0)


def test_wavelet_inv_adjoint_dot(wvlttransform, simpledata):
    scal_lm = np.copy(simpledata)
    wav_lm = np.column_stack([simpledata for _ in range(wvlttransform.nscales)])
    x = flatten_mlm(wav_lm, scal_lm)
    y = wvlttransform.inverse(x)

    f = np.copy(simpledata)
    g = wvlttransform.inverse_adjoint(f)

    dot_diff = f.conj().T.dot(y) - g.conj().T.dot(x)
    assert np.isclose(dot_diff, 0)
