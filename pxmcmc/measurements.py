from scipy import sparse
import numpy as np
import os


class Measurement:
    """
    Measurement operators operate on some map X to create observations Y
    Adjoints operate on measurements to return some map
    X is a vector of size N (number of model parameters e.g. map pixels)
    Y is a vector of size M (number of observations i.e. data points)
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

    def _read_datafile(self, datafile):
        """
        Expects a file with the following columns for each path:
        start_lat, start_lon, stop_lat, stop_lon, data, error
        Coordinates given in degrees
        TODO: Check what happens when each data point has a different error
        TODO: Figure out what to do with minor/major and nsim
        """
        all_data = np.loadtxt(datafile)
        start_lat = all_data[:, 0]
        start_lon = all_data[:, 1]
        self.start = np.stack([start_lat, start_lon], axis=1)
        stop_lat = all_data[:, 2]
        stop_lon = all_data[:, 3]
        self.stop = np.stack([stop_lat, stop_lon], axis=1)
        self.data = all_data[:, 4]
        sig_d = all_data[:, 5]
        self.sig_d = np.max(sig_d)

    def _get_path_matrix(self, path_matrix_file):
        if path_matrix_file is None:
            path_matrix_file = f"path_matrix_{self.Nside}.npz"
        if os.path.exists(path_matrix_file):
            self._read_path_matrix_file(path_matrix_file)
        else:
            self._build_path_matrix_file(path_matrix_file)

    def _read_path_matrix_file(self, path_martix_file):
        self.path_matrix = sparse.load_npz(path_martix_file)

    def _build_path_matrix_file(self, path_matrix_file):
        from greatcirclepaths import GreatCirclePath
        from multiprocessing import Pool

        def build_path(start, stop):
            path = GreatCirclePath(start, stop, self.Nside)
            path.get_points(1000)
            path.fill()
            return path.map

        itrbl = [(start, stop) for (start, stop) in zip(self.start, self.stop)]
        with Pool() as p:
            result = p.starmap_async(build_path, itrbl)
            paths = result.get()
        self.path_matrix = sparse.csr_matrix(paths)
        sparse.save_npz(path_matrix_file, self.path_matrix)
