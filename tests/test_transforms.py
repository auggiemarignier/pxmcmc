from pxmcmc.transforms import WaveletTransform
import pytest
import numpy as np


@pytest.fixture
def wvlttransform(L, B, J_min, Nside):
    return WaveletTransform(L, B, J_min, Nside)


@pytest.mark.parametrize("in_type, out_type", [("harmonic_mw", "harmonic_mw"), ("harmonic_hp", "harmonic_hp")])
def test_wavelet_lm_fwdback(simpledata_lm, wvlttransform, in_type, out_type):
    X_wav = wvlttransform.forward(simpledata_lm, in_type=in_type, out_type=out_type)
    data_rec = wvlttransform.inverse(X_wav, in_type=in_type, out_type=out_type)
    assert np.allclose(simpledata_lm, data_rec)
