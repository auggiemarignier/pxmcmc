from pxmcmc.transforms import WaveletTransform
import pytest
import numpy as np
import pys2let

from pxmcmc.utils import map2alm





@pytest.mark.parametrize(
    "in_type, out_type",
    [("harmonic_mw", "harmonic_mw"), ("harmonic_hp", "harmonic_hp")],
)
def test_wavelet_lm_fwdback(simpledata_lm, wvlttransform, in_type, out_type, L):
    if in_type == "harmonic_hp":
        simpledata_lm = pys2let.lm2lm_hp(simpledata_lm, L)
    X_wav = wvlttransform.forward(simpledata_lm, in_type=in_type, out_type=out_type)
    data_rec = wvlttransform.inverse(X_wav, in_type=in_type, out_type=out_type)
    assert np.allclose(simpledata_lm, data_rec)


@pytest.mark.parametrize(
    "in_type, out_type",
    [("pixel_mw", "pixel_mw"), ("pixel_hp", "pixel_hp")],
)
def test_wavelet_pix_fwdback(simpledata, wvlttransform, in_type, out_type, L, Nside):
    if in_type == "pixel_mw":
        simpledata = pys2let.alm2map_mw(pys2let.lm_hp2lm(map2alm(simpledata, Nside), L), L, 0)
    X_wav = wvlttransform.forward(simpledata, in_type=in_type, out_type=out_type)
    data_rec = wvlttransform.inverse(X_wav, in_type=in_type, out_type=out_type)
    assert np.allclose(simpledata, data_rec)
