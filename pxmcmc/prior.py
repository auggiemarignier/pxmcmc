import numpy as np
import pys2let

from pxmcmc.utils import soft, mw_map_weights, _multires_bandlimits


class L1:
    """
    Base L1-norm prior.  The prox of this prior is soft thresholding.

    :param string setting: 'analysis' or 'synthesis'
    :param fwd: function handle for transform operator (e.g. :meth:`transforms.Transform.forward`)
    :param adj: function handle for adjoint transform operator (e.g. :meth:`transforms.Transform.forward_adjoint`)
    :param float T: threshold for the soft thresholding function

    .. todo::
       :code:`fwd` and :code:`adj` are only needed for analysis setting.  Make these optional arguments.
    """

    def __init__(self, setting, fwd, adj, T):
        assert setting in ["analysis", "synthesis"]
        self.setting = setting
        self.fwd = fwd
        self.adj = adj
        self.T = T

    def prior(self, X):
        """
        Calculates the logprior of mcmc sample

        :param X: MCMC sample
        :return: log prior
        """
        return sum(abs(X))

    def proxf(self, X):
        """
        Calculates the proximal map of the log prior

        :param X: MCMC sample
        :return: prox of log prior
        """
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
    L1 regulariser for wavelets on S2 (MW sampling).  Performs some weighting to avoid overemphasizing pixels at the poles.

    :param int L: angular bandlimit
    :param float B: wavelet scale parameter
    :param int J_min: minimum wavelet scale
    :param int dirs: azimuthal bandlimit for directional wavelets
    :param int spin: spin number of spherical signal
    """

    def __init__(self, setting, fwd, adj, T, L, B, J_min, dirs=1, spin=0):
        super().__init__(setting, fwd, adj, T)
        self.L = L
        self.B = B
        self.J_min = J_min
        self.J_max = pys2let.pys2let_j_max(B, L, J_min)
        self.nscales = self.J_max - J_min + 1
        self.dirs = dirs
        self.spin = spin
        if setting == "synthesis":
            bls = _multires_bandlimits(L, B, J_min, dirs, spin)
            self.map_weights = np.concatenate([mw_map_weights(el) for el in bls])
        else:
            self.map_weights = mw_map_weights(L)
        self.T *= self.map_weights

    def prior(self, X):
        return super().prior(self.map_weights * X)
