from scipy import sparse


class Measurement:
    """
    Measurement operators operate on some map X to create observations Y
    Adjoints operate on measurements to return some map
    X is a vector of size N
    Y is a vector of size M
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
    def __init__(self, M, N):
        super().__init__(M, N)