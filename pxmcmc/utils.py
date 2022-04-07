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

    :param wav_lm: array of wavelet coefficients. Gets flattened if has multiple dimensions
    :param scal_lm: array of wavelet coefficients

    :return: 1D array of coefficients, with the scaling coefficients first
    """
    buff = wav_lm.ravel(order="F")
    mlm = np.concatenate((scal_lm, buff))
    return mlm


def expand_mlm(mlm, nscales=None, nscalcoefs=None, flatten_wavs=False):
    """
    Sepatates scaling and wavelet coefficients from a single vector to separate arrays.

    :param mlm: 1D array containing scaling and wavelet coefficients.  Assumes the scaling coefficients come first followed by the wavelet coefficients at increasingly smaller scales.
    :param int nscales: number of wavlet scales + 1 scaling function. Use only if coefficients do not follow the multiresolution algorithm, otherwise leave as :code:`None`
    :param int nscalcoefs: number of scaling function coefficients.  Use only if coefficients follow the multiresolution algorithm, otherwise leave as :code:`None`
    :param bool flatten_wavs: flattens the wavelet coefficients into a 1D array, otherwise 2D.  Ignored if :code:`nscalcoefs` is not :code:`None`.

    :return: tuple (wavelet coefficients, scaling coefficients)
    """
    if nscales is None and nscalcoefs is None:
        raise ValueError("Set either 'nscales', or 'nscalcoefs'")
    elif nscales is not None and nscalcoefs is not None:
        raise ValueError("Give only one of 'nscales' or 'nscalcoefs'")
    elif nscales is not None:
        v_len = mlm.size // (nscales + 1)
        assert v_len > 0
        scal_lm = mlm[:v_len]
        wav_lm = np.zeros((v_len, nscales), dtype=complex)
        for i in range(nscales):
            wav_lm[:, i] = mlm[(i + 1) * v_len : (i + 2) * v_len]
        if flatten_wavs:
            wav_lm = np.concatenate([wav_lm[:, i] for i in range(nscales)])
    elif nscalcoefs is not None:
        scal_lm = mlm[:nscalcoefs]
        wav_lm = mlm[nscalcoefs:]
    return wav_lm, scal_lm


def soft(X, T=0.1):
    """
    Soft thresholding of a vector X with threshold T.  If :math:`X_i < T`, then :math:`\mathrm{soft}(X_i) = 0`, otherwise :math:`\mathrm{soft}(X_i) = X_i-T`.

    :param X: vector of values to threshold
    :param float T: threshold.  Can be a vector of same size as :code:`X`
    
    :return: thresholded vector
    """
    X = np.array(X)
    t = _sign(X) * (np.abs(X) - T)
    t[np.abs(X) <= T] = 0
    return t


def hard(X, T=0.1):
    """
    Hard thresholding of a vector X with fraction threshold T. T is the fraction kept, i.e. the largest 100T% absolute values are kept, the others are thresholded to 0.
    TODO: What happens when all elements of X are equal?

    :meta private:
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
    :meta private:
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


def _multires_bandlimits(L, B, J_min, dirs=1, spin=0):
    phi_l, psi_lm = pys2let.wavelet_tiling(B, L, dirs, J_min, spin)
    psi_l = np.zeros((psi_lm.shape[1], L), dtype=complex)
    for j, psi in enumerate(psi_lm.T):
        psi_l[j, :] = np.array([psi[el ** 2 + el] for el in range(L)])
    gamma_l = np.vstack([phi_l, psi_l])
    bandlimits = np.zeros(gamma_l.shape[0], dtype=int)
    for j, gamma in enumerate(gamma_l):
        bandlimits[j] = np.nonzero(gamma)[0].max() + 1
    return bandlimits


def chebyshev1(X, order):
    """
    Calculates the Chebyshev polynomial of the first kind of the given order at point X.
    Uses the recurrence relation

    :math:`T_{k+1}(X) = 2XT_{k}(X) - T_{k-1}(X)`

    :math:`T_{1}(X) = X`

    :math:`T_{0}(X) = 1`

    :param X: point at which to calulate :math:`T_{k+1}`
    :param int order: polynomial order

    :return: value of Chebyshev polynomial of the first kind of the given order
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

    :math:`U_{k+1}(X) = 2XU_{k}(X) - U_{k-1}(X)`

    :math:`U_{1}(X) = 2X`

    :math:`U_{0}(X) = 1`

    :param X: point at which to calulate :math:`U_{k+1}`
    :param int order: polynomial order

    :return: value of Chebyshev polynomial of the second kind of the given order
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

    :math:`dT_{n}/dx = nU_{n-1}`

    :param X: point at which to calulate :math:`dT_{n}/dx`
    :param int order: polynomial order

    :return: derivative of Chebyshev polynomial of the second kind of the given order
    """
    if order < 0:
        raise ValueError("order must be > 0")
    elif order == 0:
        return 0
    else:
        return order * chebyshev2(X, order - 1)


def pixel_area(r, theta1, theta2, phi1, phi2):
    """
    Calculates area of a spherical rectangle.  Angles must be given in radians.

    :params float r: radius
    :params float theta1: starting colatitude
    :params float theta2: ending colatitude
    :params float phi1: starting longitude
    :params float phi2: ending longitude

    :return: area of rectangle in squared radians
    """
    return r ** 2 * (np.cos(theta1) - np.cos(theta2)) * (phi2 - phi1)


def polar_cap_area(r, theta):
    """
    Calculates the area of a polar cap.

    :params float r: radius
    :params float theta: colatitude in radians of bottom of cap

    :return: area of polar cap in squared radians
    """
    return 2 * np.pi * r ** 2 * (1 - np.cos(theta))


def calc_pixel_areas(L, r=1):
    """
    Calculates the areas of all the pixels in `MW <https://arxiv.org/abs/1110.6298>` sampling.

    :param int L: bandlimit
    :param float r: radius

    :return: array of pixel areas with shape :math:`(L, 2L-1)`
    """
    thetas, phis = pyssht.sample_positions(L)
    nthetas, nphis = thetas.shape[0], phis.shape[0]
    areas = np.zeros((nthetas, nphis), dtype=float)
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
    """
    Calculates exact quadrature weights for `MW <https://arxiv.org/abs/1110.6298>` sampling.

    :param int L: bandlimit

    :return: 1D array of quadrature weights with shape :math:`(L(2L-1),)`
    """
    wr = weights_theta(L)
    q = np.copy(wr[0:L])
    for i, j in enumerate(range(2 * L - 2, L - 1, -1)):
        q[i] = q[i] + wr[j]
    Q = np.outer(q, np.ones(2 * L - 1)).flatten()
    return Q


def s2_integrate(f, L):
    """
    Integrates a spherical image over the whole sphere

    :param array f: spherical image as 1D array of shape :math:`(L(2L-1),)`
    :param int L: bandlimit

    :return: integral of :code:`f` over the sphere

    .. todo::
       Infer :code:`L` from shape of :code:`f`
    """
    f_weighted = mw_map_weights(L) * f
    return f_weighted.sum()


def norm(x):
    return np.linalg.norm(x)


def snr(signal, noise):
    """
    Calculates the signal to noise ratio

    :math:`\mathrm{SNR} = 20\log_{10}\left(\\frac{\|\mathrm{signal}\|_2}{\|\mathrm{noise}\|_2}\\right)`

    :param signal: signal array
    :param noise: noise array

    :return: signal-to-noise ratio in decibels
    """
    return 20 * np.log10(norm(signal) / norm(noise))
