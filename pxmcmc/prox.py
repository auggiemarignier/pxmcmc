from pxmcmc.utils import soft


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

    def proxf(self, X):
        if self.setting == "synthesis":
            return self._proxf_synthesis(X)
        else:
            return self._proxf_analysis(X)

    def _proxf_synthesis(self, X):
        return soft(X, self.T)

    def _proxf_analysis(self, X):
        return X + self.fwd(soft(self.adj(X), self.T) - self.adj(X))
