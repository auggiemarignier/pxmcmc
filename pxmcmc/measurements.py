from scipy import sparse
import numpy as np
from warnings import warn


class Measurement:
    """
    Base class.

    :param int ndata: number of observed data points
    :param int npix: number of pixels in image
    """
    def __init__(self, ndata, npix):
        self.ndata = ndata
        self.npix = npix

    def forward(self, X):
        """
        Forward modelling from image to observations.  Implemented by user in custom child class.

        :param X: image
        :return: observations
        """
        raise NotImplementedError

    def adjoint(self, Y):
        """
        Adjoint modelling from observations to image.  Not necessarily the inverse of :meth:`self.forward`.  Implemented by user in custom child class.

        :param Y: observations
        :return: image
        """
        raise NotImplementedError


class Identity(Measurement):
    """
    Identity measurement operator i.e. what goes in comes out.
    """
    def __init__(self, ndata, npix):
        super().__init__(ndata, npix)
        self.eye = sparse.eye(self.ndata, self.npix)
        self.eye_adj = self.eye.getH()

    def forward(self, X):
        """:meta private:"""
        assert len(X) == self.npix
        return self.eye.dot(X)

    def adjoint(self, Y):
        """:meta private:"""
        assert len(Y) == self.ndata
        return self.eye_adj.dot(Y)


class PathIntegral(Measurement):
    """
    Path integration using a matrix that describes a set of paths.

    .. todo::
       Since this is just a matrix multiplication, can be renamed to something more generic.

    :param path_matrix: :math:`N_{\mathrm{paths}}\\times N_{\mathrm{pix}}` matrix describing a set of paths.
    """
    def __init__(self, path_matrix):
        self.path_matrix = path_matrix
        self.path_matrix_adj = self.path_matrix.getH()

        self.ndata, self.npix = path_matrix.shape

    def forward(self, X):
        """:meta private:"""
        assert len(X) == self.npix
        return self.path_matrix.dot(X)

    def adjoint(self, Y):
        """:meta private:"""
        assert len(Y) == self.ndata
        return self.path_matrix_adj.dot(Y)


class SphericalForwardModel:
    """
    Weak Gravitational Lensing spherical Forward model

    Supports additional complexity which should simply
    be appended to the dir_op and adj_op objects
    appropriately (e.g. psf degridding step, psf
    deconvolution etc.)
    """

    def __init__(self, L, mask=None, ngal=None):
        """Construct class to hold the spherical forward and
        forward adjoint operators.

        Args:

                L (int): Spherical harmonic bandlimit
                mask (int array): Map of realspace masking.
                ngal (int array): Map of galaxy observation count.

        Raises:

                ValueError: Raised if L is not positive
                ValueError: Raised if mask is the wrong shape.
                WarningLog: Raised if L is very large.
        """
        if L < 1:
            raise ValueError("Bandlimit {} must be greater than 0.".format(L))

        if L > 1024:
            warn(
                "Bandlimit {} is very large, computational price is large.".format(L)
            )

        # General class members
        self.L = L
        self.shape = (self.L, 2 * self.L - 1)

        # Define harmonic transforms and kernel mapping
        self.harmonics = op.HarmonicTransform(self.L)
        self.harmonic_kernel = self.compute_harmonic_kernel()

        # Intrinsic ellipticity dispersion
        self.var_e = 0.37 ** 2

        # Define realspace masking
        if mask is None:
            self.mask = np.ones(self.shape, dtype=bool)
        else:
            self.mask = mask.astype(bool)

        # Define observational covariance
        if ngal is None:
            self.inv_cov = self.mask_forward(np.ones(self.shape))
        else:
            self.inv_cov = self.ngal_to_inv_cov(ngal)

        if self.mask.shape != self.shape:
            raise ValueError("Shape of mask map is incorrect!")

    def dir_op(self, kappa):
        """Spherical weak lensing measurement operator

        Args:

                kappa (complex array): Convergence signal
        """
        # 1) Compute convergence harmonic coefficients
        klm = self.harmonics.forward(kappa, spin=0)
        # 2) Map to shear harmonic coefficients
        ylm = self.harmonic_mapping(klm)
        # 3) Compute shear realspace map
        y = self.harmonics.inverse(ylm, spin=2)
        # 4) Apply observational mask
        y_obs = self.mask_forward(y)
        # 5) Covariance weight the shear observations
        return self.cov_weight(y_obs)

    def adj_op(self, gamma):
        """Spherical weak lensing adjoint measurement operator

        Args:

                gamma (complex array): Shear Observations (cov weighted)
        """
        # 1) Covariance weight the shear observations
        y_obs = self.cov_weight(gamma)
        # 2) Grid masked shear onto a full-sky
        y = self.mask_adjoint(y_obs)
        # 3) Compute shear harmonic coefficients
        ylm = self.harmonics.inverse_adjoint(y, spin=2)
        # 4) Map to convergence harmonic coefficients
        klm = self.harmonic_mapping(ylm)
        # 5) Compute convergence realspace map
        return self.harmonics.forward_adjoint(klm, spin=0)

    def sks_estimate(self, gamma):
        """Computes spherical Kaiser-Squires estimator (for first estimate)

        Args:

                gamma (complex array): Shear Observations (full-sky)
        """
        ylm = self.harmonics.forward(gamma, spin=2)
        klm = self.harmonic_inverse_mapping(ylm)
        return self.harmonics.inverse(klm, spin=0)

    def compute_harmonic_kernel(self):
        """Compuptes harmonic space kernel mapping."""
        k = np.ones(self.L ** 2, dtype=float)
        index = 4
        for l in range(2, self.L):
            for m in range(-l, l + 1):
                el = float(l)
                k[index] = -1.0 * np.sqrt(((el + 2.0) * (el - 1.0)) / ((el + 1.0) * el))
                ++index
        return k

    def harmonic_mapping(self, flm):
        """Applys harmonic space mapping.

        Args:
                flm (complex array): harmonic coefficients.

        """
        out = flm * self.harmonic_kernel
        out[:4] = 0
        return out

    def harmonic_inverse_mapping(self, flm):
        """Applys harmonic space inverse mapping.

        Args:
                flm (complex array): harmonic coefficients.

        """
        out = flm / self.harmonic_kernel
        out[:4] = 0
        return out

    def mask_forward(self, f):
        """Applies given mask to a field.

        Args:

                f (complex array): Realspace Signal

        Raises:

                ValueError: Raised if signal is nan
                ValueError: Raised if signal is of incorrect shape.

        """
        if f is not f:
            raise ValueError("Signal is NaN.")

        if f.shape != self.shape:
            raise ValueError("Signal shape is incorrect for mw-sampling")

        return f[self.mask]

    def mask_adjoint(self, x):
        """Applies given mask adjoint to observations

        Args:

                x (complex array): Set of observations.

        Raises:

                ValueError: Raised if signal is nan

        """
        if x is not x:
            raise ValueError("Signal is NaN.")

        f = np.zeros(self.shape, dtype=complex)
        f[self.mask] = x
        return f

    def ngal_to_inv_cov(self, ngal):
        """Converts galaxy number density map to
        data covariance.

        Assumes no correlation between pixels.

        Args:
                ngal (real array): pixel space map of number of observations per pixel.

        """
        ngal_m = self.mask_forward(ngal)
        return np.sqrt((2.0 * ngal_m) / (self.var_e))

    def cov_weight(self, x):
        """Applies covariance weighting to observations.

        Assumes no correlation between pixels.

        Args:
                x (array): pixel space map to be inverse covariance weighted.

        """
        return x * self.inv_cov