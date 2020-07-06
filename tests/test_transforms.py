import numpy as np


def test_wavelet_lm_fwdback(simpledata_lm, wvlttransform):
    X_wav = wvlttransform.forward(simpledata_lm, in_type="harmonic_mw", out_type="harmonic_mw")
    data_rec = wvlttransform.inverse(X_wav, in_type="harmonic_mw", out_type="harmonic_mw")
    assert np.allclose(simpledata_lm, data_rec)


def test_wavelet_lm_fwdback_hp(simpledata_hp_lm, wvlttransform):
    X_wav = wvlttransform.forward(simpledata_hp_lm, in_type="harmonic_hp", out_type="harmonic_hp")
    data_rec = wvlttransform.inverse(X_wav, in_type="harmonic_hp", out_type="harmonic_hp")
    assert np.allclose(simpledata_hp_lm, data_rec)


def test_wavelet_pix_fwdback(simpledata, wvlttransform):
    X_wav = wvlttransform.forward(simpledata, in_type="pixel_mw", out_type="pixel_mw")
    data_rec = wvlttransform.inverse(X_wav, in_type="pixel_mw", out_type="pixel_mw")
    assert np.allclose(simpledata, data_rec)


def test_wavelet_pix_fwdback_hp(simpledata_hp, wvlttransform):
    X_wav = wvlttransform.forward(simpledata_hp, in_type="pixel_hp", out_type="pixel_hp")
    data_rec = wvlttransform.inverse(X_wav, in_type="pixel_hp", out_type="pixel_hp")
    assert np.allclose(simpledata_hp, data_rec)
