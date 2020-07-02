import pys2let
import numpy as np

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
        self,
        L,
        B,
        J_min,
        Nside=None,
        dirs=1,
        spin=0,
        pred_out_type="harmonic_mw",
        param_in_type="harmonic_mw",
    ):
        self.L = L
        self.B = B
        self.J_min = J_min
        self.Nside = Nside
        self.J_max = pys2let.pys2let_j_max(self.B, self.L, self.J_min)
        self.nscales = self.J_max - self.J_min + 1
        self.dirs = dirs
        self.spin = spin
        assert pred_out_type in ["harmonic_mw", "harmonic_hp", "pixel_mw", "pixel_hp"]
        self.pred_out_type = (
            pred_out_type  # prediction outputs in harmonic/pixel space hp/mw format
        )
        assert param_in_type in ["harmonic_mw", "harmonic_hp", "pixel_mw", "pixel_hp"]
        self.param_in_type = (
            param_in_type  # wavelet coefficients in harmonic/pixel space hp/mw format
        )

    def inverse(self, X):
        """
        X is a of wavlet and scaling harmonic coefficients in MW format
        If pred_out_type is 'harmonic_mw', returns spherical harmonic coefficients in MW format
        If pred_out_type is 'pixel_hp', returns a Healpix map
        """
        clm_hp = self._mw_wav_lm2hp_lm(X)
        if self.pred_out_type == "harmonic_hp":
            return clm_hp
        elif self.pred_out_type == "harmonic_mw":
            return pys2let.lm_hp2lm(clm_hp, self.L)
        elif self.pred_out_type == "pixel_hp":
            return alm2map(clm_hp, self.Nside)
        else:
            raise NotImplementedError

    def inverse_adjoint(self, f):
        if self.param_in_type == "harmonic_mw":
            f = pys2let.alm2map_mw(f, self.L, self.spin)
        elif self.param_in_type == "pixel_hp":
            f = map2alm(f, self.L)
            f = pys2let.lm_hp2lm(f, self.L)
            f = pys2let.alm2map_mw(f, self.L, self.spin)
        elif self.param_in_type == "harmonic_hp":
            f = pys2let.lm_hp2lm(f, self.L)
            f = pys2let.alm2map_mw(f, self.L, self.spin)
        else:
            pass
        return flatten_mlm(*self._mw2mw_lm_adjoint(f))

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
