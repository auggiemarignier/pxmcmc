import pys2let
import pyssht
import numpy as np

from pxmcmc.utils import expand_mlm, flatten_mlm


class Transform:
    """Base class to wrap transformations."""
    def forward(self):
        """
        e.g. spherical image to spherical harmonics.  Implemented by user in custom child class.
        """
        raise NotImplementedError

    def inverse(self):
        """
        e.g. spherical harmonics to spherical image.  Implemented by user in custom child class.
        """
        raise NotImplementedError

    def forward_adjoint(self):
        """
        e.g. spherical harmonics to spherical image.  Implemented by user in custom child class.
        """
        raise NotImplementedError

    def inverse_adjoint(self):
        """
        e.g. spherical image to spherical harmonics.  Implemented by user in custom child class.
        """
        raise NotImplementedError


class IdentityTransform(Transform):
    """Identity transform i.e. does nothing"""
    def __init__(self):
        pass

    def forward(self, X):
        """:meta private:"""
        return X

    def forward_adjoint(self, X):
        """:meta private:"""
        return X

    def inverse(self, X):
        """:meta private:"""
        return X

    def inverse_adjoint(self, X):
        """:meta private:"""
        return X


class WaveletTransform(Transform):
    """
    Spherical wavelet transforms

    :param int L: angular bandlimit
    :param float B: wavelet scale parameter
    :param int J_min: minimum wavelet scale
    :param int dirs: azimuthal bandlimit for directional wavelets
    :param int spin: spin number of spherical signal
    """
    def __init__(
        self, L, B, J_min, dirs=1, spin=0,
    ):
        self.L = L
        self.B = B
        self.J_min = J_min
        self.J_max = pys2let.pys2let_j_max(self.B, self.L, self.J_min)
        self.nscales = self.J_max - self.J_min + 1
        self.dirs = dirs
        self.spin = spin

        self._get_ncoefs()

    def forward(self, X):
        """
        Transform image to spherical wavelet space.

        :param X: spherical image as a 1D array
        :return: 1D array of spherical wavelet coefficients
        """
        if not isinstance(X, complex):
            X = X.astype(complex)
        X_wav, X_scal = pys2let.analysis_px2wav(
            X, self.B, self.L, self.J_min, N=1, spin=0, upsample=0
        )
        return flatten_mlm(X_wav, X_scal)

    def inverse(self, X):
        """
        Transform spherical wavelet to image space.

        :param X: 1D array of spherical wavelet coefficients
        :return: spherical image as a 1D array
        """
        wav, scal = expand_mlm(X, nscalcoefs=self.nscal)
        if not isinstance(scal, complex):
            scal = scal.astype(complex)
        if not isinstance(scal, complex):
            wav = wav.astype(complex)
        X = pys2let.synthesis_wav2px(
            wav, scal, self.B, self.L, self.J_min, self.dirs, self.spin, upsample=0
        )
        return X

    def inverse_adjoint(self, X):
        """
        Adjoint transform image to spherical wavelet space.

        :param X: spherical image as a 1D array
        :return: 1D array of spherical wavelet coefficients
        """
        if not isinstance(X, complex):
            X = X.astype(complex)
        X_wav, X_scal = pys2let.synthesis_adjoint_px2wav(
            X, self.B, self.L, self.J_min, self.dirs, self.spin, upsample=0
        )
        return flatten_mlm(X_wav, X_scal)

    def forward_adjoint(self, X):
        """
        Transform spherical wavelet to image space.

        :param X: 1D array of spherical wavelet coefficients
        :return: spherical image as a 1D array
        """
        wav, scal = expand_mlm(X, nscalcoefs=self.nscal)
        if not isinstance(scal, complex):
            scal = scal.astype(complex)
        if not isinstance(scal, complex):
            wav = wav.astype(complex)
        X = pys2let.analysis_adjoint_wav2px(
            wav, scal, self.B, self.L, self.J_min, self.dirs, self.spin, upsample=0
        )
        return X

    def _get_ncoefs(self):
        """
        Counts the number of wavelet and scaling coefs for
        the multiresolution algorithm
        """
        f_mw = np.empty(pyssht.sample_length(self.L), dtype=complex)
        f_wav, f_scal = pys2let.analysis_px2wav(
            f_mw, self.B, self.L, self.J_min, self.dirs, self.spin, upsample=0
        )
        self.nwav, self.nscal = (f_wav.shape[0], f_scal.shape[0])
        self.ncoefs = self.nwav + self.nscal
