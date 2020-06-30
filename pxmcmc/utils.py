import numpy as np
import healpy as hp
from contextlib import contextmanager
import os
import sys
import pys2let


def flatten_mlm(wav_lm, scal_lm):
    """
    Takes a set of wavelet and scaling coefficients and flattens them into a single vector
    """
    buff = wav_lm.ravel(order="F")
    mlm = np.concatenate((scal_lm, buff))
    return mlm


def expand_mlm(mlm, nscales):
    """
    Sepatates scaling and wavelet coefficients from a single vector to separate arrays.
    """
    v_len = mlm.size // (nscales + 1)
    assert v_len > 0
    scal_lm = mlm[:v_len]
    wav_lm = np.zeros((v_len, nscales), dtype=np.complex)
    for i in range(nscales):
        wav_lm[:, i] = mlm[(i + 1) * v_len : (i + 2) * v_len]
    return wav_lm, scal_lm


def soft(X, T=0.1):
    """
    Soft thresholding of a vector X with threshold T.  If Xi is less than T, then soft(Xi) = 0, otherwise soft(Xi) = Xi-T.
    """
    X = np.array(X)
    t = _sign(X) * (np.abs(X) - T)
    t[np.abs(X) <= T] = 0
    return t


def hard(X, T=0.1):
    """
    Hard thresholding of a vector X with fraction threshold T. T is the fraction kept, i.e. the largest 100T% absolute values are kept, the others are thresholded to 0.
    TODO: What happens when all elements of X are equal?
    """
    X_srt = np.sort(abs(X))
    thresh_ind = int(T * len(X))
    thresh_val = X_srt[-thresh_ind]
    X[abs(X) < thresh_val] = 0
    return X


def _sign(z):
    abs = np.abs(z)
    z[abs == 0] = 0
    abs[abs == 0] = 1
    return z / abs


@contextmanager
def suppress_stdout():
    """
    Suppresses stdout from some healpy functions
    """
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


def map2alm(image, lmax, **kwargs):
    with suppress_stdout():
        return hp.map2alm(image, lmax, **kwargs)


def alm2map(alm, nside, **kwargs):
    with suppress_stdout():
        return hp.alm2map(alm, nside, **kwargs)


def get_parameter_from_chain(chain, L, base, el, em):
    assert np.abs(em) <= el
    base_start = base * (L) ** 2
    index_in_base = el * el + el + em
    return chain[:, base_start + index_in_base]


def wavelet_basis(L, B, J_min, dirs=1, spin=0):
    theta_l, psi_lm = pys2let.wavelet_tiling(B, L, dirs, spin, J_min)
    psi_lm = psi_lm[:, J_min:]
    theta_lm = _fix_theta(L, B, J_min)
    basis = np.concatenate((theta_lm, psi_lm), axis=1)
    return basis


def _fix_theta(L, B, J_min):
    J_max = pys2let.pys2let_j_max(B, L, J_min)
    nscales = J_max - J_min + 1
    dummy_psilm = np.zeros(((L) ** 2, nscales), dtype=np.complex)
    dummy_thetalm = np.zeros(((L) ** 2), dtype=np.complex)
    for ell in range(L):
        for em in range(-ell, ell + 1):
            dummy_thetalm[ell * ell + ell + em] = np.sqrt((2 * ell + 1) / (4 * np.pi))

    dummy_psilm_hp = np.zeros(
        (L * (L + 1) // 2, dummy_psilm.shape[1]), dtype=np.complex
    )
    dummy_thetalm_hp = pys2let.lm2lm_hp(dummy_thetalm.flatten(), L)
    dummy_lm_hp = pys2let.synthesis_axisym_lm_wav(
        dummy_psilm_hp, dummy_thetalm_hp, B, L, J_min
    )
    theta_lm = pys2let.lm_hp2lm(dummy_lm_hp, L)
    return np.expand_dims(theta_lm, axis=1)


def chebyshev1(X, order):
    """
    Calculates the Chebyshev polynomial of the first kind of the given order at point X.
    Uses the recurrence relation
            T_{k+1}(X) = 2XT_{k}(X) - T_{k-1}(X)
            T_{1}(X) = X
            T_{0}(X) = 1
    """
    if order < 0:
        raise ValueError("order must be >= 0")
    elif order == 0:
        return 1
    elif order == 1:
        return X
    else:
        return 2 * X * chebyshev1(X, order - 1) - chebyshev1(X, order - 2)


def chebyshev2(X, order):
    """
    Calculates the Chebyshev polynomial of the second kind of the given order at point X.
    Uses the recurrence relation
            U_{k+1}(X) = 2XU_{k}(X) - U_{k-1}(X)
            U_{1}(X) = 2X
            U_{0}(X) = 1
    """
    if order < 0:
        raise ValueError("order must be >= 0")
    elif order == 0:
        return 1
    elif order == 1:
        return 2 * X
    else:
        return 2 * X * chebyshev2(X, order - 1) - chebyshev2(X, order - 2)


def cheb1der(X, order):
    """
    Evaluates the derivative of the Chebyshev polynomial of the first kind of the given order at point X.
    Uses the relation
            dT_{n}/dx = nU_{n-1}
    """
    if order < 0:
        raise ValueError("order must be > 0")
    elif order == 0:
        return 0
    else:
        return order * chebyshev2(X, order - 1)
