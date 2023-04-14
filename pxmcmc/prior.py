import numpy as np
import pys2let
import pyssht

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
            raise NotImplementedError
        self.T *= self.map_weights

    def prior(self, X):
        return super().prior(self.map_weights * X)


class S2_Wavelets_L1_Power_Weights(S2_Wavelets_L1):
    """
    L1 regulariser for wavelets on S2 (MW sampling).
    Includes weighting for pixel area, wavelet power wavelet decay
    See eqns 33&34 from Wallis et al 2017

    :param int L: angular bandlimit
    :param float B: wavelet scale parameter
    :param int J_min: minimum wavelet scale
    :param int dirs: azimuthal bandlimit for directional wavelets
    :param int spin: spin number of spherical signal
    :param float eta: wavelet decay tuning parameter
    """

    def __init__(self, setting, fwd, adj, T, L, B, J_min, dirs=1, spin=0, eta=1):
        super().__init__(setting, fwd, adj, T, L, B, J_min, dirs, spin)
        self.eta = eta
        if setting == "synthesis":
            self._get_weights()
        else:
            raise NotImplementedError
        self.T *= self.map_weights

    def prior(self, X):
        return super().prior(self.map_weights * X)

    def _get_weights(self):
        scaling_weights = self._calculate_scaling_weights()
        s = scaling_weights.flatten()
        wavelet_weights = self._calculate_wavelet_weights()
        w = np.concatenate([w.flatten() for w in wavelet_weights])
        self.map_weights = np.concatenate([s, w])

    def _calculate_scaling_weights(self):
        phi_l, _ = pys2let.wavelet_tiling(self.B, self.L, self.dirs, self.J_min, self.spin)
        scaling_power = np.vdot(phi_l, phi_l).real
        effective_L = np.nonzero(phi_l)[0].max() + 1
        nsamples = pyssht.sample_length(effective_L)
        weights = np.full(pyssht.sample_shape(effective_L), 2 * np.pi ** 2 / (scaling_power * nsamples))
        thetas, _ = pyssht.sample_positions(effective_L)
        weights = (weights.T * np.sin(thetas)).T
        return weights

    def _calculate_wavelet_weights(self):
        bls = _multires_bandlimits(self.L, self.B, self.J_min)
        _, psi_lm = pys2let.wavelet_tiling(self.B, self.L, self.dirs, self.J_min, self.spin)
        wavelet_powers = np.array(
            [np.vdot(lm, lm).real for lm in psi_lm.T]
        )
        psi_l = np.zeros((psi_lm.shape[1], self.L), dtype=complex)
        for j, psi in enumerate(psi_lm.T):
            psi_l[j, :] = np.array([psi[el ** 2 + el] for el in range(self.L)])
        peak_ls = np.array(
            [np.argmax(l) for l in psi_l]
        )
        all_weights = []
        for effective_L, power, peak_l in zip(bls[1:], wavelet_powers, peak_ls):
            nsamples = pyssht.sample_length(effective_L)
            weights = np.full(pyssht.sample_shape(effective_L), (2 * np.pi ** 2) * (peak_l ** self.eta) / (power * nsamples))
            thetas, _ = pyssht.sample_positions(effective_L)
            weights = (weights.T * np.sin(thetas)).T
            all_weights.append(weights)
        return np.array(all_weights, dtype=list)
