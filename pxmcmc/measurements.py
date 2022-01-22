from scipy import sparse
import numpy as np
from warnings import warn
import pyssht


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


class WeakLensingHarmonic(Measurement):
    """
    Weak Gravitational Lensing spherical Forward model in spherical harmonic space

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
        self.shape = (self.L ** 2, )

        # Define harmonic transforms and kernel mapping
        self.harmonic_kernel = self.compute_harmonic_kernel()

        # Intrinsic ellipticity dispersion
        self.var_e = 0.37 ** 2

    def dir_op(self, klm):
        """Spherical weak lensing measurement operator

        Args:

                klm (complex array): Convergence signal in harmonic space
        """
        # Map to shear harmonic coefficients
        return self.harmonic_mapping(klm)

    def adj_op(self, glm):
        """Spherical weak lensing adjoint measurement operator

        Args:

                glm (complex array): Shear Observations in harmonic space
        """
        return self.harmonic_mapping(glm)

    def sks_estimate(self, glm):
        """Computes spherical Kaiser-Squires estimator (for first estimate)

        Args:

                glm (complex array): Shear Observations in harmonic space
        """
        return self.harmonic_inverse_mapping(glm)

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
