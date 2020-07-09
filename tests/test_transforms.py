import numpy as np
import pytest

from pxmcmc.utils import flatten_mlm


@pytest.mark.parametrize(
    "inout_type", ["harmonic_mw", "harmonic_hp", "pixel_mw", "pixel_hp"]
)
def test_wavelet_fwdback(inout_type, wvlttransform, all_data):
    data = all_data[inout_type]
    X_wav = wvlttransform.forward(data, in_type=inout_type, out_type=inout_type)
    data_rec = wvlttransform.inverse(X_wav, in_type=inout_type, out_type=inout_type)
    assert np.allclose(data, data_rec)


@pytest.mark.parametrize(
    "inout_type",
    [
        pytest.param(
            "harmonic_mw",
            marks=pytest.mark.xfail(reason="Adjoints only work in MW pixel space"),
        ),
        pytest.param(
            "harmonic_hp",
            marks=pytest.mark.xfail(reason="Adjoints only work in MW pixel space"),
        ),
        "pixel_mw",
        pytest.param(
            "pixel_hp",
            marks=pytest.mark.xfail(reason="Adjoints only work in MW pixel space"),
        ),
    ],
)
def test_wavelet_fwd_adjoint_dot(
    inout_type, wvlttransform, all_data,
):
    data = all_data[inout_type]
    # if y = Ax and g = A'f, show that f'y = g'x
    x = np.copy(data)
    y = wvlttransform.forward(x, in_type=inout_type, out_type=inout_type)

    scal_lm = np.copy(data)
    wav_lm = np.column_stack([data for _ in range(wvlttransform.nscales)])
    f = flatten_mlm(wav_lm, scal_lm)
    g = wvlttransform.forward_adjoint(f, in_type=inout_type, out_type=inout_type)

    dot_diff = f.conj().T.dot(y) - g.conj().T.dot(x)
    assert np.isclose(dot_diff, 0)


@pytest.mark.parametrize(
    "inout_type",
    [
        pytest.param(
            "harmonic_mw",
            marks=pytest.mark.xfail(reason="Adjoints only work in MW pixel space"),
        ),
        pytest.param(
            "harmonic_hp",
            marks=pytest.mark.xfail(reason="Adjoints only work in MW pixel space"),
        ),
        "pixel_mw",
        pytest.param(
            "pixel_hp",
            marks=pytest.mark.xfail(reason="Adjoints only work in MW pixel space"),
        ),
    ],
)
def test_wavelet_inv_adjoint_dot(
    inout_type, wvlttransform, all_data,
):
    data = all_data[inout_type]
    scal_lm = np.copy(data)
    wav_lm = np.column_stack([data for _ in range(wvlttransform.nscales)])
    x = flatten_mlm(wav_lm, scal_lm)
    y = wvlttransform.inverse(x, in_type=inout_type, out_type=inout_type)

    f = np.copy(data)
    g = wvlttransform.inverse_adjoint(f, in_type=inout_type, out_type=inout_type)

    dot_diff = f.conj().T.dot(y) - g.conj().T.dot(x)
    assert np.isclose(dot_diff, 0)
