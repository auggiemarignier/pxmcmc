import numpy as np
import pytest

from pxmcmc.utils import flatten_mlm


@pytest.mark.parametrize(
    "inout_type", ["harmonic_mw", "harmonic_hp", "pixel_mw", "pixel_hp"]
)
def test_wavelet_fwdback(
    inout_type,
    wvlttransform,
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
    X_wav = wvlttransform.forward(
        data[inout_type], in_type=inout_type, out_type=inout_type
    )
    data_rec = wvlttransform.inverse(X_wav, in_type=inout_type, out_type=inout_type)
    assert np.allclose(data[inout_type], data_rec)


def test_wavelet_fwd_adjoint_dot(wvlttransform, simpledata, simpledata_lm):
    # if y = Ax and g = A'f, show that f'y = g'x
    x = np.copy(simpledata)
    y = wvlttransform.forward(x, in_type="pixel_mw", out_type="pixel_mw")

    scal_lm = np.copy(simpledata_lm)
    wav_lm = np.column_stack([simpledata_lm for _ in range(wvlttransform.nscales)])
    f = flatten_mlm(wav_lm, scal_lm)
    g = wvlttransform.forward_adjoint(f, in_type="harmonic_mw", out_type="pixel_mw")

    dot_diff = f.T.dot(y) - g.T.dot(x)
    assert np.isclose(dot_diff, 0)


def test_wavelet_inv_adjoint_dot(wvlttransform, simpledata_hp, simpledata_hp_lm):
    scal_lm = np.copy(simpledata_hp)
    wav_lm = np.column_stack([simpledata_hp for _ in range(wvlttransform.nscales)])
    x = flatten_mlm(wav_lm, scal_lm)
    y = wvlttransform.inverse(x, in_type="pixel_hp", out_type="pixel_mw")

    f = np.copy(simpledata_hp_lm)
    g = wvlttransform.inverse_adjoint(f, in_type="harmonic_hp", out_type="pixel_mw")

    dot_diff = f.T.dot(y) - g.T.dot(x)
    assert np.isclose(dot_diff, 0)
