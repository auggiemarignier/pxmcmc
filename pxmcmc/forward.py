import pys2let
import numpy as np
from .utils import expand_mlm


class ForwardOperator:
    """
    Base Forward identity operator
    """
    def __init__(self, data, sig_d):
        self.data = data
        self.sig_d = sig_d

    def forward(self, X):
        return X

    def calc_gradg(self, preds):
        return (preds - self.data) / (self.sig_d ** 2)




class PathIntegral:
    def forward(self, X):
        """
        Forward modelling.  Takes a vector X containing the scaling and wavelet coefficients generated by the chain and predicts path averaged phase velocity.  Possible extension of this is to include finite frequency kernels.
        """
        wav_lm, scal_lm = expand_mlm(X, self.params.nscales)
        scal_lm_hp = pys2let.lm2lm_hp(scal_lm, self.params.L + 1)
        wav_lm_hp = np.zeros(
            [(self.params.L + 1) * (self.params.L + 2) // 2, self.params.nscales],
            dtype=np.complex,
        )
        for j in range(self.params.nscales):
            wav_lm_hp[:, j] = pys2let.lm2lm_hp(
                np.ascontiguousarray(wav_lm[:, j]), self.params.L + 1
            )
        clm_hp = pys2let.synthesis_axisym_lm_wav(
            wav_lm_hp, scal_lm_hp, self.params.B, self.params.L + 1, self.params.J_min
        )
        clm = np.real(pys2let.lm_hp2lm(clm_hp, self.params.L + 1))  # check complexity
        preds = np.matmul(clm, self.Ylmavs.T)
        return preds

    def calc_gradg(self, preds):
        """
        Calculates the gradient of the data fidelity term, which should guide the MCMC search.
        """
        diff = (self.p_weights ** 2) * (preds - self.data) / self.params.sig_d
        diffYlmavs = np.sum(diff * self.Ylmavs.T, axis=1)
        arrays = [diffYlmavs for _ in range(self.basis.shape[1])]
        diffYlmavs = np.concatenate(arrays)
        gradg = self.pf * diffYlmavs
        return gradg
    def _calc_prefactors(self):
        """
        Calculates prefactors of gradg which are constant throughout the chain, and so only need to be calculated once at the start.
        """
        prefactors = np.zeros(np.prod(self.basis.shape))
        for i, base in enumerate(self.basis.T):
            base_l0s = [base[l ** 2 + l] for l in range(self.params.L)]
            for ell in range(self.params.L):
                prefactors[
                    i * len(base) + ell ** 2 : i * len(base) + (ell + 1) ** 2
                ] = np.sqrt(4 * np.pi / (2 * ell + 1)) * np.real(base_l0s[ell])
        self.pf = prefactors