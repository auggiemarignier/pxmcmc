import pys2let
import numpy as np
import healpy as hp

from pxmcmc.utils import expand_mlm, flatten_mlm, alm2map, map2alm


class Transform:
    def forward(self):
        """
        e.g. spherical image to spherical harmonics
        """
        raise NotImplementedError

    def inverse(self):
        """
        e.g. spherical harmonics to spherical image
        """
        raise NotImplementedError

    def forward_adjoint(self):
        """
        e.g. spherical harmonics to spherical image
        """
        raise NotImplementedError

    def inverse_adjoint(self):
        """
        e.g. spherical image to spherical harmonics
        """
        raise NotImplementedError


class IdentityTransform(Transform):
    def __init__(self):
        pass

    def forward(self, X):
        return X

    def forward_adjoint(self, X):
        return X

    def inverse(self, X):
        return X

    def inverse_adjoint(self, X):
        return X


class WaveletTransform(Transform):
    def __init__(
        self, L, B, J_min, Nside=None, dirs=1, spin=0,
    ):
        self.L = L
        self.B = B
        self.J_min = J_min
        self.Nside = Nside
        self.J_max = pys2let.pys2let_j_max(self.B, self.L, self.J_min)
        self.nscales = self.J_max - self.J_min + 1
        self.dirs = dirs
        self.spin = spin

    def inverse(self, X, in_type="harmonic_mw", out_type="harmonic_mw"):
        """
        X is a set of wavlet and scaling harmonic coefficients in MW format
        If out_type is 'harmonic_mw', returns spherical harmonic coefficients in MW format
        If out_type is 'pixel_hp', returns a Healpix map
        """
        clm_hp = self._mw_wav_lm2hp_lm(X)
        if out_type == "harmonic_hp":
            return clm_hp
        elif out_type == "harmonic_mw":
            return pys2let.lm_hp2lm(clm_hp, self.L)
        elif out_type == "pixel_hp":
            return alm2map(clm_hp, self.Nside)
        else:
            raise NotImplementedError

    def inverse_adjoint(self, f, in_type="harmonic_mw", out_type="harmonic_mw"):
        if in_type == "harmonic_mw":
            f = pys2let.alm2map_mw(f, self.L, self.spin)
        elif in_type == "pixel_hp":
            f = map2alm(f, self.L)
            f = pys2let.lm_hp2lm(f, self.L)
            f = pys2let.alm2map_mw(f, self.L, self.spin)
        elif in_type == "harmonic_hp":
            f = pys2let.lm_hp2lm(f, self.L)
            f = pys2let.alm2map_mw(f, self.L, self.spin)
        else:
            pass
        return flatten_mlm(*self._mw2mw_lm_adjoint(f))

    def forward(self, X, in_type="harmonic_mw", out_type="harmonic_mw"):
        """
        X is an input map in either pixel or harmonic space
        Returns a vector of scaling and wavelet coefficients in either pixel or harmonic space
        """
        self._check_inout_types(in_type, out_type)
        if in_type == "harmonic_mw":
            X_hp_lm = pys2let.lm2lm_hp(X, self.L)
        elif in_type == "harmonic_hp":
            X_hp_lm = X
        elif in_type == "pixel_hp":
            X_hp_lm = map2alm(X, self.L - 1)
        else:
            X_hp_lm = self._pixmw2pixhp(X)

        X_wav_hp_lm, X_scal_hp_lm = pys2let.analysis_axisym_lm_wav(
            X_hp_lm, self.B, self.L, self.J_min
        )

        if out_type == "harmonic_hp":
            return flatten_mlm(X_wav_hp_lm, X_scal_hp_lm)
        elif out_type == "harmonic_mw":
            X_scal_lm, X_wav_lm = self._harmonic_hp2harmonic_mw_wavelets(
                X_scal_hp_lm, X_wav_hp_lm
            )
            return flatten_mlm(X_wav_lm, X_scal_lm)
        elif out_type == "pixel_hp":
            X_scal, X_wav = self._harmonic_hp2pix_hp_wavelets(X_scal_hp_lm, X_wav_hp_lm)
            return flatten_mlm(X_wav, X_scal)
        else:
            X_scal_lm, X_wav_lm = self._harmonic_hp2harmonic_mw_wavelets(
                X_scal_hp_lm, X_wav_hp_lm
            )
            X_scal, X_wav = self._harmonic_mw2pix_mw_wavelets(X_scal_lm, X_wav_lm)
            return flatten_mlm(X_wav, X_scal)

    def _mw2mw_lm_adjoint(self, f):
        """
        f is a MW map
        Returns harmonic wavelet coefficients in MW format
        """
        f_wav, f_scal = pys2let.synthesis_adjoint_axisym_wav_mw(
            f, self.B, self.L, self.J_min
        )
        f_scal_lm = pys2let.map2alm_mw(f_scal, self.L, self.spin)
        f_wav_lm = np.zeros([self.L ** 2, self.nscales], dtype=np.complex)
        vlen = self.L * (2 * self.L - 1)
        for j in range(self.nscales):
            f_wav_lm[:, j] = pys2let.map2alm_mw(
                f_wav[j * vlen : (j + 1) * vlen + 1], self.L, self.spin
            )
        return f_wav_lm, f_scal_lm

    def _mw_wav_lm2hp_lm(self, X):
        wav_lm, scal_lm = expand_mlm(X, self.nscales)
        scal_lm_hp = pys2let.lm2lm_hp(scal_lm, self.L)
        wav_lm_hp = np.zeros(
            [(self.L) * (self.L + 1) // 2, self.nscales], dtype=np.complex,
        )
        for j in range(self.nscales):
            wav_lm_hp[:, j] = pys2let.lm2lm_hp(
                np.ascontiguousarray(wav_lm[:, j]), self.L
            )
        clm_hp = pys2let.synthesis_axisym_lm_wav(
            wav_lm_hp, scal_lm_hp, self.B, self.L, self.J_min
        )
        return clm_hp

    def _check_inout_types(self, in_type, out_type):
        if in_type not in ["harmonic_mw", "harmonic_hp", "pixel_mw", "pixel_hp"]:
            raise ValueError("Wrong input format")
        if out_type not in ["harmonic_mw", "harmonic_hp", "pixel_mw", "pixel_hp"]:
            raise ValueError("Wrong output format")
