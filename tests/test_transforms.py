from pxmcmc.transforms import WaveletTransform
import pytest
import numpy as np


@pytest.fixture
def wvlttransform(L, B, J_min, Nside):
    return WaveletTransform(L, B, J_min, Nside)


def test_wavelet_lm_fwdback(simpledata_lm, wvlttransform):
    data_rec = wvlttransform.inverse(wvlttransform.forward(simpledata_lm))
    assert np.allclose(simpledata_lm, data_rec)
