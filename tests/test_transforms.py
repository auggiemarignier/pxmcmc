import numpy as np
import pytest


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
    X_wav = wvlttransform.forward(data[inout_type], in_type=inout_type, out_type=inout_type)
    data_rec = wvlttransform.inverse(X_wav, in_type=inout_type, out_type=inout_type)
    assert np.allclose(data[inout_type], data_rec)
