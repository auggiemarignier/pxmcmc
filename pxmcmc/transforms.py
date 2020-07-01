import pys2let
import numpy as np

from pxmcmc.utils import expand_mlm, alm2map


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


class WaveletTransform(Transform):
    def __init__(self, L, B, J_min, Nside=None, dirs=1, spin=0, out_type="harmonic"):
        self.L = L
        self.B = B
        self.J_min = J_min
        self.Nside = Nside
        self.J_max = pys2let.pys2let_j_max(self.B, self.L, self.J_min)
        self.nscales = self.J_max - self.J_min + 1
        self.dirs = dirs
        self.spin = spin
        assert out_type in ["harmonic", "pixel"]
        self.out_type = out_type

    def inverse(self, X):
        """
        X is a of wavlet and scaling harmonic coefficients in MW format
        If out_type is 'harmonic', returns spherical harmonic coefficients in MW format
        If out_type is 'pixel', returns a Healpix map
        """
        clm_hp = self._mw_wav_lm2hp_lm(X)
        if self.out_type == "harmonic":
            return pys2let.lm_hp2lm(clm_hp, self.L)
        else:
            return alm2map(clm_hp, self.Nside)

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
