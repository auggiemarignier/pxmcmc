from scipy import sparse


class Measurement:
    """
    Measurement operators operate on some map X to create observations Y
    Adjoints operate on measurements to return some map
    X is a vector of size npix (number of pixels on the sphere)
    Y is a vector of size ndata (number of data points)
    """

    def __init__(self, ndata, npix):
        self.ndata = ndata
        self.npix = npix

    def forward(self):
        raise NotImplementedError

    def adjoint(self):
        raise NotImplementedError


class Identity(Measurement):
    def __init__(self, ndata, npix):
        super().__init__(ndata, npix)
        self.eye = sparse.eye(self.ndata, self.npix)
        self.eye_adj = self.eye.getH()

    def forward(self, X):
        assert len(X) == self.npix
        return self.eye.dot(X)

    def adjoint(self, Y):
        assert len(Y) == self.ndata
        return self.eye_adj.dot(Y)


class PathIntegral(Measurement):
    def __init__(self, datafile, ndata, npix, path_matrix_file=None):
        super().__init__(ndata, npix)
        self._read_datafile(datafile)
        self._get_path_matrix(path_matrix_file)

        self.path_matrix_adj = self.path_matrix.getH()

    def forward(self, X):
        assert len(X) == self.npix
        return self.path_matrix.dot(X)

    def adjoint(self, Y):
        assert len(Y) == self.ndata
        return self.path_matrix_adj.dot(Y)
