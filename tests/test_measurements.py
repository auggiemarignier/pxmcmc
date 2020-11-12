import numpy as np
import pyssht
from scipy import sparse

from pxmcmc.measurements import PathIntegral


def test_pathintegral_dot(L):
    pathmatrix = sparse.random(100, pyssht.sample_length(L, Method="MW"))
    pathint = PathIntegral(pathmatrix)

    xlm = np.zeros(L * L, complex)
    for el in range(L):
        m = 0
        ind = pyssht.elm2ind(el, m)
        xlm[ind] = np.random.randn()
        for m in range(1, el + 1):
            ind_pm = pyssht.elm2ind(el, m)
            ind_nm = pyssht.elm2ind(el, -m)
            xlm[ind_pm] = np.random.randn() + 1j * np.random.randn()
            xlm[ind_nm] = (-1)**m * np.conj(xlm[ind_pm])
    x = pyssht.inverse(xlm, L, Method="MW", Reality=True).flatten()
    yt = pathint.forward(x)

    y = np.random.rand(100)
    xt = pathint.adjoint(y)

    dot_product_error = y.conj().dot(yt) - xt.conj().dot(x)
    assert np.isclose(dot_product_error, 0)


def test_pathintegral_fwd_weights(L):
    pathmatrix = np.zeros(pyssht.sample_shape(L, Method="MW"))
    piby2_index = pyssht.theta_to_index(np.pi / 2, L)
    pathmatrix[piby2_index, :] = 1
    pathmatrix = np.expand_dims(pathmatrix.flatten(), axis=0)
    spacing_in_phi = pyssht.sample_positions(L)[1][1]
    weights = np.full(pathmatrix.shape[1], spacing_in_phi)
    pathmatrix = sparse.csr_matrix(pathmatrix * weights)
    pathint = PathIntegral(pathmatrix)

    X = np.ones(pyssht.sample_length(L, Method="MW"))
    pred = pathint.forward(X)

    assert np.isclose(pred, 2 * np.pi)
