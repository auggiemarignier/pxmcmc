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


class WaveletFormatter:
    """
    Helper class for transforming wavelet and scaling functions between hp and mw
    both in pixel and harmonic space
    Wavelet maps are expected as 2D arrays, with the second dimension as the number of wavelet maps
    """

    def __init__(self, L, B, J_min, Nside, spin=0):
        self.L = L
        self.B = B
        self.J_min = J_min
        self.Nside = Nside
        self.spin = 0
        self.J_max = pys2let.pys2let_j_max(self.B, self.L, self.J_min)
        self.nscales = self.J_max - self.J_min + 1

    def _split_wavelets(self, wav):
        vlen = wav.size // self.nscales
        wavs = np.column_stack(
            [wav[i * vlen : (i + 1) * vlen] for i in range(self.nscales)]
        )
        return wavs

    def _flatten_wavelets(self, wav):
        return np.concatenate([wav[:, j] for j in range(self.nscales)])

    def _pixmw2harmhp(self, f_mw):
        if not isinstance(f_mw, complex):
            f_mw = f_mw.astype(complex)
        f_mw_lm = pys2let.map2alm_mw(f_mw, self.L, self.spin)
        return pys2let.lm2lm_hp(f_mw_lm, self.L)

    def _pixmw2pixhp(self, f_mw):
        f_hp_lm = self._pixmw2harmhp(f_mw)
        return alm2map(f_hp_lm, self.Nside)

    def _pixhp2harmmw(self, f_hp):
        f_hp_lm = map2alm(f_hp, self.L - 1)
        return pys2let.lm_hp2lm(f_hp_lm, self.L)

    def _pixhp2pixmw(self, f_hp):
        f_mw_lm = self._pixhp2harmmw(f_hp)
        return pys2let.alm2map_mw(f_mw_lm, self.L, self.spin)

    def _harmhp2pixmw(self, f_lm_hp):
        f_mw_lm = pys2let.lm_hp2lm(f_lm_hp, self.L)
        return pys2let.alm2map_mw(f_mw_lm, self.L, self.spin)

    def _harmmw2pixmw_wavelets(self, scal_lm, wav_lm):
        scal_mw = pys2let.alm2map_mw(scal_lm, self.L, self.spin)
        wav_mw = np.zeros(pys2let.mw_size(self.L) * self.nscales, dtype=np.complex)
        offset = pys2let.mw_size(self.L)
        offset_lm = self.L ** 2
        for j in range(self.nscales):
            wav_mw[j * offset : (j + 1) * offset] = pys2let.alm2map_mw(
                wav_lm[j * offset_lm : (j + 1) * offset_lm], self.L, self.spin,
            )
        return scal_mw, wav_mw

    def _pixhp2pixmw_wavelets(self, scal, wav):
        scal_mw = self._pixhp2pixmw(scal)
        wav_mw = np.zeros(pys2let.mw_size(self.L) * self.nscales, dtype=np.complex)
        offset = pys2let.mw_size(self.L)
        offset_hp = hp.nside2npix(self.Nside)
        for j in range(self.nscales):
            wav_mw[j * offset : (j + 1) * offset] = self._pixhp2pixmw(
                wav[j * offset_hp : (j + 1) * offset_hp]
            )
        return scal_mw, wav_mw

    def _harmhp2pixhp_wavelets(self, scal_hp_lm, wav_hp_lm):
        scal_hp = alm2map(scal_hp_lm, self.Nside)
        wav_hp = np.zeros(hp.nside2npix(self.Nside) * self.nscales, dtype=np.complex)
        offset = hp.nside2npix(self.Nside)
        offset_lm = self.L * (self.L + 1) // 2
        for j in range(self.nscales):
            wav_hp[j * offset : (j + 1) * offset] = alm2map(
                wav_hp_lm[j * offset_lm : (j + 1) * offset_lm], self.Nside,
            )
        return scal_hp, wav_hp

    def _harmhp2harmmw_wavelets(self, scal_lm_hp, wav_lm_hp):
        scal_lm = pys2let.lm_hp2lm(scal_lm_hp, self.L)
        wav_lm = np.zeros(self.L * self.L * self.nscales, dtype=np.complex)
        offset_lm = self.L ** 2
        offset_lm_hp = self.L * (self.L + 1) // 2
        for j in range(self.nscales):
            wav_lm[j * offset_lm : (j + 1) * offset_lm] = pys2let.lm_hp2lm(
                wav_lm_hp[j * offset_lm_hp : (j + 1) * offset_lm_hp], self.L,
            )
        return scal_lm, wav_lm

    def _harmhp2pixmw_wavelets(self, scal_lm_hp, wav_lm_hp):
        scal_lm = pys2let.lm_hp2lm(scal_lm_hp, self.L)
        scal_mw = pys2let.alm2map_mw(scal_lm, self.L, self.spin)
        wav_mw = np.zeros(pys2let.mw_size(self.L) * self.nscales, dtype=np.complex)
        offset = pys2let.mw_size(self.L)
        offset_lm_hp = self.L * (self.L + 1) // 2
        for j in range(self.nscales):
            buff = pys2let.lm_hp2lm(
                wav_lm_hp[j * offset_lm_hp : (j + 1) * offset_lm_hp], self.L,
            )
            wav_mw[j * offset : (j + 1) * offset] = pys2let.alm2map_mw(
                buff, self.L, self.spin
            )
        return scal_mw, wav_mw

    def _harmmw2harmhp_wavelets(self, scal_lm, wav_lm):
        scal_lm_hp = pys2let.lm2lm_hp(scal_lm, self.L)
        wav_lm_hp = np.zeros(
            self.L * (self.L + 1) * self.nscales // 2, dtype=np.complex
        )
        offset_lm = self.L ** 2
        offset_lm_hp = self.L * (self.L + 1) // 2
        for j in range(self.nscales):
            wav_lm_hp[j * offset_lm_hp : (j + 1) * offset_lm_hp] = pys2let.lm2lm_hp(
                wav_lm[j * offset_lm : (j + 1) * offset_lm], self.L
            )
        return scal_lm_hp, wav_lm_hp

    def _pixmw2harmhp_wavelets(self, scal, wav):
        scal_lm_hp = self._pixmw2harmhp(scal)
        wav_lm_hp = np.zeros(
            self.L * (self.L + 1) * self.nscales // 2, dtype=np.complex
        )
        offset = pys2let.mw_size(self.L)
        offset_lm_hp = self.L * (self.L + 1) // 2
        for j in range(self.nscales):
            wav_lm_hp[j * offset_lm_hp : (j + 1) * offset_lm_hp] = self._pixmw2harmhp(
                wav[j * offset : (j + 1) * offset]
            )
        return scal_lm_hp, wav_lm_hp

    def _pixmw2harmmw_wavelets(self, scal, wav):
        scal_lm = pys2let.map2alm_mw(scal, self.L, self.spin)
        wav_lm = np.zeros(self.L ** 2 * self.nscales, dtype=np.complex)
        offset = pys2let.mw_size(self.L)
        offset_lm = self.L ** 2
        for j in range(self.nscales):
            wav_lm[j * offset_lm : (j + 1) * offset_lm] = pys2let.map2alm_mw(
                wav[j * offset : (j + 1) * offset], self.L, self.spin
            )
        return scal_lm, wav_lm

    def _pixmw2pixhp_wavelets(self, scal, wav):
        scal_lm_hp, wav_lm_hp = self._pixmw2harmhp_wavelets(scal, wav)
        scal_hp = alm2map(scal_lm_hp, self.Nside)
        wav_hp = np.zeros(hp.nside2npix(self.Nside) * self.nscales, dtype=np.complex)
        offset = self.L * (self.L + 1) // 2
        offset_hp = hp.nside2npix(self.Nside)
        for j in range(self.nscales):
            wav_hp[j * offset_hp : (j + 1) * offset_hp] = alm2map(
                wav_lm_hp[j * offset : (j + 1) * offset], self.Nside
            )
        return scal_hp, wav_hp

    def _pixhp2harmhp_wavelets(self, scal, wav):
        scal_lm_hp = map2alm(scal, self.L - 1)
        wav_lm_hp = np.zeros(
            self.L * (self.L + 1) // 2 * self.nscales, dtype=np.complex
        )
        offset_hp = hp.nside2npix(self.Nside)
        offset_hp_lm = self.L * (self.L + 1) // 2
        for j in range(self.nscales):
            wav_lm_hp[j * offset_hp_lm : (j + 1) * offset_hp_lm] = map2alm(
                wav[j * offset_hp : (j + 1) * offset_hp], self.L - 1
            )
        return scal_lm_hp, wav_lm_hp


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
