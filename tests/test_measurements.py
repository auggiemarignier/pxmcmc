import numpy as np
import pyssht
from scipy import sparse

from pxmcmc.measurements import PathIntegral, WeakLensingHarmonic, WeakLensing


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
            xlm[ind_nm] = (-1) ** m * np.conj(xlm[ind_pm])
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


def test_weaklensingharmonic_dot(L):
    operator = WeakLensingHarmonic(L)

    # Generate random convergence
    klm = np.random.rand(L * L) + 1j * np.random.rand(L * L)
    klm[:4] = 0  # remove mono/dipole

    # Generate random shear
    glm = np.random.rand(L * L) + 1j * np.random.rand(L * L)
    glm[:4] = 0  # remove mono/dipole

    # Apply operator
    k_to_g = operator.forward(klm)
    g_to_k = operator.adjoint(glm)

    # Perform adjoint operator dot test.
    a = abs(np.vdot(klm, g_to_k))
    b = abs(np.vdot(glm, k_to_g))
    assert np.count_nonzero(klm) > 0
    assert np.count_nonzero(glm) > 0
    assert np.count_nonzero(k_to_g) > 0
    assert np.count_nonzero(g_to_k) > 0
    assert np.isclose(a, b)


def test_weaklensing_dot(L):
    operator = WeakLensing(L)

    # Generate random convergence
    klm = np.random.rand(L * L) + 1j * np.random.rand(L * L)
    klm[:4] = 0  # remove mono/dipole
    kappa = pyssht.inverse(klm, L)

    # Generate random shear
    glm = np.random.rand(L * L) + 1j * np.random.rand(L * L)
    glm[:4] = 0  # remove mono/dipole
    gamma = operator.mask_forward(pyssht.inverse(glm, L))

    # Apply operator
    k_to_g = operator.forward(kappa)
    g_to_k = operator.adjoint(gamma)

    # Perform adjoint operator dot test.
    a = abs(np.vdot(kappa, g_to_k))
    b = abs(np.vdot(gamma, k_to_g))
    assert np.count_nonzero(kappa) > 0
    assert np.count_nonzero(gamma) > 0
    assert np.count_nonzero(k_to_g) > 0
    assert np.count_nonzero(g_to_k) > 0
    assert np.isclose(a, b)


def test_weaklensing_mask_dot(L):
    # Generate random mask
    mask = np.zeros(L * (2 * L - 1), dtype=int)
    mask[:int(0.5 * mask.size)] = 1
    np.random.shuffle(mask)
    mask = mask.reshape((L, 2 * L - 1))

    operator = WeakLensing(L, mask=mask)

    # Generate random convergence
    klm = np.random.rand(L * L) + 1j * np.random.rand(L * L)
    klm[:4] = 0  # remove mono/dipole
    kappa = pyssht.inverse(klm, L)

    # Generate random shear
    glm = np.random.rand(L * L) + 1j * np.random.rand(L * L)
    glm[:4] = 0  # remove mono/dipole
    gamma = operator.mask_forward(pyssht.inverse(glm, L))

    # Apply operator
    k_to_g = operator.forward(kappa)
    g_to_k = operator.adjoint(gamma)

    # Perform adjoint operator dot test.
    a = abs(np.vdot(kappa, g_to_k))
    b = abs(np.vdot(gamma, k_to_g))
    assert np.count_nonzero(kappa) > 0
    assert np.count_nonzero(gamma) > 0
    assert np.count_nonzero(k_to_g) > 0
    assert np.count_nonzero(g_to_k) > 0
    assert np.isclose(a, b)
