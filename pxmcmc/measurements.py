from scipy import sparse


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
