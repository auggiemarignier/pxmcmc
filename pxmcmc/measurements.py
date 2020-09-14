from scipy import sparse


class Measurement:
    """
    Measurement operators operate on some map X to create observations Y
    Adjoints operate on measurements to return some map
    X is a vector of size N (number of parameters in analysis setting)
    Y is a vector of size M (number of data points)
    """

    def __init__(self, M, N):
        self.M = M
        self.N = N

    def forward(self):
        raise NotImplementedError

    def adjoint(self):
        raise NotImplementedError


class Identity(Measurement):
    def __init__(self, M, N):
        super().__init__(M, N)
        self.eye = sparse.eye(self.M, self.N)
        self.eye_adj = self.eye.getH()

    def forward(self, X):
        assert len(X) == self.N
        return self.eye.dot(X)

    def adjoint(self, Y):
        assert len(Y) == self.M
        return self.eye_adj.dot(Y)


class PathIntegral(Measurement):
    def __init__(self, datafile, M, N, path_matrix_file=None):
        super().__init__(M, N)
        self._read_datafile(datafile)
        self._get_path_matrix(path_matrix_file)

        self.path_matrix_adj = self.path_matrix.getH()

    def forward(self, X):
        assert len(X) == self.N
        return self.path_matrix.dot(X)

    def adjoint(self, Y):
        assert len(Y) == self.M
        return self.path_matrix_adj.dot(Y)
