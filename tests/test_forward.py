import pyssht
from scipy import sparse
import numpy as np
from pytest_cases import parametrize_with_cases

from pxmcmc.forward import SphericalWaveletTransformOperator, PathIntegralOperator

# These tests only test output size, to ensure something is returned
# Tests on individal transform and measurement operators are more valuable


def case_swtoperator(simpledata, sig_d, L, B, J_min, setting):
    return SphericalWaveletTransformOperator(simpledata, sig_d, setting, L, B, J_min)


def case_pathintoperator(simpledata, sig_d, setting, L, B, J_min):
    pathmatrix = sparse.random(len(simpledata), mw_sample_length(L))
    return PathIntegralOperator(pathmatrix, simpledata, sig_d, setting, L, B, J_min)


@parametrize_with_cases("operator", cases=".")
def test_operator_forward(operator):
    sample = np.random.rand(operator.nparams).astype(complex)
    preds = operator.forward(sample)
    assert len(preds) == len(operator.data)


@parametrize_with_cases("operator", cases=".")
def test_operator_gradg(operator):
    preds = np.random.rand(len(operator.data))
    gradg = operator.calc_gradg(preds)
    assert len(gradg) == operator.nparams
