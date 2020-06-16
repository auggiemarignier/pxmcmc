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
    phi_l, psi_lm = pys2let.wavelet_tiling(B, L, dirs, spin, J_min)
    psi_lm = psi_lm[:, J_min:]
    phi_lm = _fix_phi(L, B, J_min)
    basis = np.concatenate((phi_lm, psi_lm), axis=1)
    return basis


def _fix_phi(L, B, J_min):
    J_max = pys2let.pys2let_j_max(B, L, J_min)
    nscales = J_max - J_min + 1
    dummy_psilm = np.zeros(((L) ** 2, nscales), dtype=np.complex)
    dummy_philm = np.zeros(((L) ** 2), dtype=np.complex)
    for ell in range(L):
        for em in range(-ell, ell + 1):
            dummy_philm[ell * ell + ell + em] = np.sqrt((2 * ell + 1) / (4 * np.pi))

    dummy_psilm_hp = np.zeros(
        (L * (L + 1) // 2, dummy_psilm.shape[1]), dtype=np.complex
    )
    dummy_philm_hp = pys2let.lm2lm_hp(dummy_philm.flatten(), L)
    dummy_lm_hp = pys2let.synthesis_axisym_lm_wav(
        dummy_psilm_hp, dummy_philm_hp, B, L, J_min
    )
    phi_lm = pys2let.lm_hp2lm(dummy_lm_hp, L)
    return np.expand_dims(phi_lm, axis=1)


class GreatCirclePath:
    def __init__(self, start, stop, Nside):
        """
        Finds all the points along a great circle path and the corresponding healpix pixels
        start/stop are tuples (lat, lon) in degrees
        """
        self.start = start
        self.stop = stop
        self.Nside = Nside
        self.map = np.zeros(hp.nside2npix(Nside))

        self._endpoints2rad()

    def fill(self):
        pixels = [hp.ang2pix(self.Nside, *point) for point in self.points]
        self.map[pixels] = 1

    def _get_points(self):
        pass

    def _endpoints2rad(self):
        """
        Converts endpoints to radians
        Latitudes become colatitudes
        """
        # start_lt, start_ln = self.start
        # stop_lt, stop_ln = self.stop
        # start_clt = 90 - start_lt
        # stop_clt = 90 - stop_lt
        # self.start = tuple(np.deg2rad((start_clt, start_ln)))
        # self.stop = tuple(np.deg2rad((stop_clt, stop_ln)))
        self.start = tuple(np.deg2rad(self.start))
        self.stop = tuple(np.deg2rad(self.stop))

    def _course(self):
        """
        Calculates course from start point to stop point
        """
        phi1 = self.start[0]
        phi2 = self.stop[0]
        lmda12 = self.stop[1] - self.start[1]
        if lmda12 > 180:
            lmda12 -= 360
        elif lmda12 < -180:
            lmda12 += 360
        else:
            pass
        numerator = np.cos(phi2) * np.sin(lmda12)
        denominator = np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(
            phi2
        ) * np.cos(lmda12)
        return np.arctan2(numerator, denominator)