import pys2let
import pyssht
import numpy as np

from pxmcmc.utils import expand_mlm, flatten_mlm, WaveletFormatter


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
        fwd_in_type=None,
        fwd_out_type=None,
        inv_in_type=None,
        inv_out_type=None,
        fwd_adj_in_type=None,
        fwd_adj_out_type=None,
        inv_adj_in_type=None,
        inv_adj_out_type=None,
    ):
        self.L = L
        self.B = B
        self.J_min = J_min
        self.Nside = Nside
        self.J_max = pys2let.pys2let_j_max(self.B, self.L, self.J_min)
        self.nscales = self.J_max - self.J_min + 1
        self.dirs = dirs
        self.spin = spin

        self._formatter = WaveletFormatter(L, B, J_min, Nside, spin=spin)
        self._get_nparams()

        self.fwd_in_type = fwd_in_type
        self.fwd_out_type = fwd_out_type
        self.inv_in_type = inv_in_type
        self.inv_out_type = inv_out_type
        self.fwd_adj_in_type = fwd_adj_in_type
        self.fwd_adj_out_type = fwd_adj_out_type
        self.inv_adj_in_type = inv_adj_in_type
        self.inv_adj_out_type = inv_adj_out_type

    def forward(self, X, in_type=None, out_type=None):
        """
        X is an input map in either pixel or harmonic space
        Returns a vector of scaling and wavelet coefficients in either pixel or harmonic space
        """
        if self.fwd_in_type is not None:
            in_type = self.fwd_in_type
        if self.fwd_out_type is not None:
            out_type = self.fwd_out_type
        self._check_inout_types(in_type, out_type)

        X = self._intype2mwpx(X, in_type)
        if not isinstance(X, complex):
            X = X.astype(complex)
        X_wav, X_scal = pys2let.analysis_px2wav(
            X, self.B, self.L, self.J_min, N=1, spin=0, upsample=0
        )
        X_scal_out, X_wav_out = self._wavelets_mwpx2outtype(X_scal, X_wav, out_type)
        return flatten_mlm(X_wav_out, X_scal_out)

    def inverse(self, X, in_type=None, out_type=None):
        """
        X is a vector of wavlet and scaling
        """
        if self.inv_in_type is not None:
            in_type = self.inv_in_type
        if self.inv_out_type is not None:
            out_type = self.inv_out_type
        self._check_inout_types(in_type, out_type)

        wav, scal = expand_mlm(X, nscalcoefs=self.nscal)
        scal, wav = self._wavelets_intype2mwpx(scal, wav, in_type)
        if not isinstance(scal, complex):
            scal = scal.astype(complex)
        if not isinstance(scal, complex):
            scal = scal.astype(complex)
        X = pys2let.synthesis_wav2px(
            wav, scal, self.B, self.L, self.J_min, self.dirs, self.spin, upsample=0
        )
        return self._mwpx2outtype(X, out_type)

    def inverse_adjoint(self, X, in_type=None, out_type=None):
        """
        X is a input map
        """
        if self.inv_adj_in_type is not None:
            in_type = self.inv_adj_in_type
        if self.inv_adj_out_type is not None:
            out_type = self.inv_adj_out_type
        self._check_inout_types(in_type, out_type)

        X = self._intype2mwpx(X, in_type)
        if not isinstance(X, complex):
            X = X.astype(complex)
        X_wav, X_scal = pys2let.synthesis_adjoint_axisym_wav_mw(X, self.B, self.L, self.J_min)
        X_scal_out, X_wav_out = self._wavelets_mwpx2outtype(
            X_scal, X_wav, out_type
        )
        return flatten_mlm(X_wav_out, X_scal_out)

    def forward_adjoint(self, X, in_type=None, out_type=None):
        """
        X is a vector of wavlet and scaling functions
        """
        if self.fwd_adj_in_type is not None:
            in_type = self.fwd_adj_in_type
        if self.fwd_adj_out_type is not None:
            out_type = self.fwd_adj_out_type
        self._check_inout_types(in_type, out_type)

        wav, scal = expand_mlm(X, self.nscales, flatten_wavs=True)
        scal, wav = self._wavelets_intype2mwpx(scal, wav, in_type)
        if not isinstance(scal, complex):
            scal = scal.astype(complex)
        if not isinstance(scal, complex):
            scal = scal.astype(complex)
        X = pys2let.analysis_adjoint_axisym_wav_mw(wav, scal, self.B, self.L, self.J_min)
        return self._mwpx2outtype(X, out_type)

    def _intype2mwpx(self, X, in_type):
        if in_type == "pixel_mw":
            return X
        elif in_type == "harmonic_mw":
            return pys2let.alm2map_mw(X, self.L, self.spin)
        elif in_type == "pixel_hp":
            return self._formatter._pixhp2pixmw(X)
        else:
            return self._formatter._harmhp2pixmw(X)

    def _wavelets_intype2mwpx(self, scal, wav, in_type):
        if in_type == "pixel_mw":
            return scal, wav
        elif in_type == "harmonic_mw":
            return self._formatter._harmmw2pixmw_wavelets(scal, wav)
        elif in_type == "pixel_hp":
            return self._formatter._pixhp2pixmw_wavelets(scal, wav)
        else:
            return self._formatter._harmhp2pixmw_wavelets(scal, wav)

    def _mwpx2outtype(self, X, out_type):
        if out_type == "pixel_mw":
            return X
        elif out_type == "harmonic_mw":
            return pys2let.map2alm_mw(X, self.L, self.spin)
        elif out_type == "pixel_hp":
            return self._formatter._pixmw2pixhp(X)
        else:
            return self._formatter._pixmw2harmhp(X)

    def _wavelets_mwpx2outtype(self, X_scal, X_wav, out_type):
        if out_type == "pixel_mw":
            return X_scal, X_wav
        elif out_type == "harmonic_mw":
            return self._formatter._pixmw2harmmw_wavelets(X_scal, X_wav)
        elif out_type == "pixel_hp":
            return self._formatter._pixmw2pixhp_wavelets(X_scal, X_wav)
        else:
            return self._formatter._pixmw2harmhp_wavelets(X_scal, X_wav)

    def _check_inout_types(self, in_type, out_type):
        if in_type not in ["harmonic_mw", "harmonic_hp", "pixel_mw", "pixel_hp"]:
            raise ValueError(f"Wrong input format: {in_type}")
        if out_type not in ["harmonic_mw", "harmonic_hp", "pixel_mw", "pixel_hp"]:
            raise ValueError(f"Wrong output format: {out_type}")

    def _get_nparams(self):
        """
        Counts the number of wavelet and scaling coefs for
        the multiresolution algorithm
        """
        f_mw = np.empty(pyssht.sample_length(self.L), dtype=complex)
        f_wav, f_scal = pys2let.analysis_px2wav(
            f_mw, self.B, self.L, self.J_min, self.dirs, self.spin, upsample=0
        )
        self.nwav, self.nscal = (f_wav.shape[0], f_scal.shape[0])
        self.nparams = self.nwav + self.nscal
