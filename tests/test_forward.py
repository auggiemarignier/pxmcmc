import pys2let
import numpy as np

from pxmcmc.utils import expand_mlm


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