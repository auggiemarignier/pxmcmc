import numpy as np
import pys2let
import pyssht

from pxmcmc.utils import soft, weighted_s2


class L1:
    """
    L1-norm regulariser
    """

    def __init__(self, setting, fwd, adj, T):
        """
        setting = "analysis" or "synthesis"
        fwd = function handle for forward transform operator
        adj = function handle for adjoint transform operator
        T = threshold for soft thresholding
        """
        assert setting in ["analysis", "synthesis"]
        self.setting = setting
        self.fwd = fwd
        self.adj = adj
        self.T = T

    def prior(self, X):
        return sum(abs(X))

    def proxf(self, X):
        if self.setting == "synthesis":
            return self._proxf_synthesis(X)
        else:
            return self._proxf_analysis(X)

    def _proxf_synthesis(self, X):
        return soft(X, self.T)

    def _proxf_analysis(self, X):
        return X + self.fwd(soft(self.adj(X), self.T) - self.adj(X))


class S2_Wavelets_L1(L1):
    """
    L1 regulariser for wavelets on S2 (MW sampling)
    Performs some weighting to avoid overemphasizing pixels at the poles
    """

    def __init__(self, setting, fwd, adj, T, L, B, J_min):
        super().__init__(setting, fwd, adj, T)
        self.L = L
        B = B
        J_min = J_min
        J_max = pys2let.pys2let_j_max(B, L, J_min)
        self.nscales = J_max - J_min + 1

    def prior(self, X):
        X = self._weight_maps(X, self.L)
        return super().prior(X)

    def proxf(self, X):
        X = self._weight_maps(X, self.L)
        return super().proxf(X)

    def _weight_maps(self, X, L):
        X_w = np.zeros_like(X)
        if self.setting == "synthesis":
            map_size = pyssht.sample_length(L, Method="MW")
            for j in range(self.nscales):
                wav_map = X[j * map_size : (j + 1) * map_size]
                X_w[j * map_size : (j + 1) * map_size] = weighted_s2(wav_map, self.L)
        else:
            X_w = weighted_s2(X, self.L)
        return X_w
