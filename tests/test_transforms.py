import numpy as np
from pytest_cases import parametrize_with_cases, fixture

from pxmcmc.utils import flatten_mlm
from pxmcmc.transforms import SphericalWaveletTransform


@fixture
@parametrize_with_cases(
    "harmonic", cases=[lambda: False]
)  # add true case when functions are added to pys2let
def wvlttransform(L, B, J_min, harmonic):
    return SphericalWaveletTransform(L, B, J_min, harmonic=harmonic)


def test_wavelet_fwdback(wvlttransform, simpledata):
    X_wav = wvlttransform.forward(simpledata)
    data_rec = wvlttransform.inverse(X_wav)
    assert np.allclose(simpledata, data_rec)


def test_wavelet_fwd_adjoint_dot(wvlttransform, simpledata):
    # if y = Ax and g = A'f, show that f'y = g'x
    x = np.copy(simpledata)
    y = wvlttransform.forward(x)

    scal = np.random.rand(wvlttransform.nscal)
    wav = np.random.rand(wvlttransform.nwav)
    f = flatten_mlm(wav, scal)
    g = wvlttransform.forward_adjoint(f.astype(complex))

    dot_diff = f.conj().T.dot(y) - g.conj().T.dot(x)
    assert np.isclose(dot_diff, 0)


def test_wavelet_inv_adjoint_dot(wvlttransform, simpledata):
    scal = np.random.rand(wvlttransform.nscal)
    wav = np.random.rand(wvlttransform.nwav)
    x = flatten_mlm(wav, scal)
    y = wvlttransform.inverse(x.astype(complex))

    f = np.copy(simpledata)
    g = wvlttransform.inverse_adjoint(f)

    dot_diff = f.conj().T.dot(y) - g.conj().T.dot(x)
    assert np.isclose(dot_diff, 0)
