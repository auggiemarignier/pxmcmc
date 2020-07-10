import pys2let
import numpy as np

from pxmcmc.utils import expand_mlm, flatten_mlm

# These tests mostly test implementation
# Tests on individal transform and measurement operators are more valuable


def test_WaveletTransformOperator_forward(swtoperator):
    sample = np.random.rand(swtoperator.nparams).astype(np.complex)
    if swtoperator.setting == "analysis":
        expected = np.copy(sample)
    else:
        wav, scal = expand_mlm(sample, swtoperator.transform.nscales, flatten_wavs=True)
        B = swtoperator.transform.B
        L = swtoperator.transform.L
        J_min = swtoperator.transform.J_min
        expected = pys2let.synthesis_axisym_wav_mw(wav, scal, B, L, J_min)

    assert np.allclose(swtoperator.forward(sample), expected)


def test_WaveletTransformOperator_gradg(swtoperator):
    preds = np.random.rand(len(swtoperator.data)).astype(np.complex)
    if swtoperator.setting == "analysis":
        expected = preds - swtoperator.data
    else:
        B = swtoperator.transform.B
        L = swtoperator.transform.L
        J_min = swtoperator.transform.J_min
        diff = preds - swtoperator.data
        expected = flatten_mlm(
            *pys2let.synthesis_adjoint_axisym_wav_mw(diff, B, L, J_min)
        )

    assert np.allclose(swtoperator.calc_gradg(preds), expected)
