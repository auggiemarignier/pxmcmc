import pys2let
import pyssht
import numpy as np

from pxmcmc.utils import expand_mlm, flatten_mlm


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
    ):
        self.L = L
        self.B = B
        self.J_min = J_min
        self.Nside = Nside
        self.J_max = pys2let.pys2let_j_max(self.B, self.L, self.J_min)
        self.nscales = self.J_max - self.J_min + 1
        self.dirs = dirs
        self.spin = spin

        self._get_nparams()

    def forward(self, X):
        """
        X is an input map in either pixel or harmonic space
        Returns a vector of scaling and wavelet coefficients in either pixel or harmonic space
        """
        if not isinstance(X, complex):
            X = X.astype(complex)
        X_wav, X_scal = pys2let.analysis_px2wav(
            X, self.B, self.L, self.J_min, N=1, spin=0, upsample=0
        )
        return flatten_mlm(X_wav, X_scal)

    def inverse(self, X):
        """
        X is a vector of wavlet and scaling
        """
        wav, scal = expand_mlm(X, nscalcoefs=self.nscal)
        if not isinstance(scal, complex):
            scal = scal.astype(complex)
        if not isinstance(scal, complex):
            scal = scal.astype(complex)
        X = pys2let.synthesis_wav2px(
            wav, scal, self.B, self.L, self.J_min, self.dirs, self.spin, upsample=0
        )
        return X

    def inverse_adjoint(self, X):
        """
        X is a input map
        """
        if not isinstance(X, complex):
            X = X.astype(complex)
        X_wav, X_scal = pys2let.synthesis_adjoint_axisym_wav_mw(X, self.B, self.L, self.J_min)
        return flatten_mlm(X_wav, X_scal)

    def forward_adjoint(self, X):
        """
        X is a vector of wavlet and scaling functions
        """
        wav, scal = expand_mlm(X, self.nscales, flatten_wavs=True)
        if not isinstance(scal, complex):
            scal = scal.astype(complex)
        if not isinstance(scal, complex):
            scal = scal.astype(complex)
        X = pys2let.analysis_adjoint_axisym_wav_mw(wav, scal, self.B, self.L, self.J_min)
        return X

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
