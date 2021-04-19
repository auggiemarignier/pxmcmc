import numpy as np
import healpy as hp
from contextlib import contextmanager
import os
import sys
import pys2let
import pyssht


def flatten_mlm(wav_lm, scal_lm):
    """
    Takes a set of wavelet and scaling coefficients and flattens them into a single vector
    """
    buff = wav_lm.ravel(order="F")
    mlm = np.concatenate((scal_lm, buff))
    return mlm


def expand_mlm(mlm, nscales=None, nscalcoefs=None, flatten_wavs=False):
    """
    Sepatates scaling and wavelet coefficients from a single vector to separate arrays.
    Must be given one of 'nscales' or 'nscalcoefs'.
    'nscales' is number of wavlet scales + 1 scaling function (not multires)
    'nscalcoefs' is number of scaling coefficients in multiresolution
    """
    if nscales is None and nscalcoefs is None:
        raise ValueError("Set either 'nscales', or 'nscalcoefs'")
    elif nscales is not None and nscalcoefs is not None:
        raise ValueError("Give only one of 'nscales' or 'nscalcoefs'")
    elif nscales is not None:
        v_len = mlm.size // (nscales + 1)
        assert v_len > 0
        scal_lm = mlm[:v_len]
        wav_lm = np.zeros((v_len, nscales), dtype=np.complex)
        for i in range(nscales):
            wav_lm[:, i] = mlm[(i + 1) * v_len : (i + 2) * v_len]
        if flatten_wavs:
            wav_lm = np.concatenate([wav_lm[:, i] for i in range(nscales)])
    elif nscalcoefs is not None :
        scal_lm = mlm[:nscalcoefs]
        wav_lm = mlm[nscalcoefs:]
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


def pixel_area(r, theta1, theta2, phi1, phi2):
    return r ** 2 * (np.cos(theta1) - np.cos(theta2)) * (phi2 - phi1)


def polar_cap_area(r, alpha):
    return 2 * np.pi * r ** 2 * (1 - np.cos(alpha))


def calc_pixel_areas(L, r=1):
    thetas, phis = pyssht.sample_positions(L)
    nthetas, nphis = thetas.shape[0], phis.shape[0]
    areas = np.zeros((nthetas, nphis), dtype=np.float64)
    phis = np.append(phis, [2 * np.pi])
    areas[0] = polar_cap_area(r, thetas[0]) / nphis
    for t, theta1 in enumerate(thetas[:-1]):
        theta2 = thetas[t + 1]
        for p, phi1 in enumerate(phis[:-1]):
            phi2 = phis[p + 1]
            areas[t + 1][p] = pixel_area(r, theta1, theta2, phi1, phi2)
    return areas


def mw_weights(m):
    if m == 1:
        w = 1j * np.pi / 2
    elif m == -1:
        w = -1j * np.pi / 2
    elif m % 2 == 0:
        w = 2.0 / (1.0 - m * m)
    else:
        w = 0

    return w


def weights_theta(L):
    wr = np.zeros(2 * L - 1, dtype=complex)
    for i, m in enumerate(range(-(L - 1), L)):
        wr[i] = mw_weights(m) * np.exp(-1j * m * np.pi / (2 * L - 1))
    wr = (np.fft.fft(np.fft.ifftshift(wr)) * 2 * np.pi / (2 * L - 1) ** 2).real
    return wr


def mw_map_weights(L):
    wr = weights_theta(L)
    q = np.copy(wr[0:L])
    for i, j in enumerate(range(2 * L - 2, L - 1, -1)):
        q[i] = q[i] + wr[j]
    Q = np.outer(q, np.ones(2 * L - 1)).flatten()
    return Q


def s2_integrate(f, L):
    f_weighted = mw_map_weights(L) * f
    return f_weighted.sum()


def norm(x):
    return np.linalg.norm(x)


def snr(signal, noise):
    return 20 * np.log10(norm(signal) / norm(noise))
