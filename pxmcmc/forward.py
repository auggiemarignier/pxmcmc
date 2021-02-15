from pys2let import mw_size
from scipy import sparse

from pxmcmc.measurements import Identity, PathIntegral
from pxmcmc.transforms import WaveletTransform


class ForwardOperator:
    """
    Base Forward operator. Combines a transform and a measurement operator
    Children of this class must define analysis/synthesis forward and gradg functions
    Children must also take data and sig_d in the constructor
    Number of model parameters must also be given
    """

    def __init__(
        self, data, sig_d, setting, transform=None, measurement=None, nparams=None
    ):
        self.data = data
        self.invcov = self._build_inverse_covariance_matrix(sig_d)
        if setting not in ["analysis", "synthesis"]:
            raise ValueError
        self.setting = setting
        if transform is not None:
            self.transform = transform
        if measurement is not None:
            self.measurement = measurement
        if nparams is not None:
            self.nparams = nparams

    def forward(self, X):
        if self.setting == "analysis":
            return self._forward_analysis(X)
        else:
            return self._forward_synthesis(X)

    def calc_gradg(self, preds):
        if self.setting == "analysis":
            return self._gradg_analysis(preds)
        else:
            return self._gradg_synthesis(preds)

    def _forward_analysis(self, X):
        return self.measurement.forward(X)

    def _forward_synthesis(self, X):
        realspace = self.transform.inverse(X)
        prediction = self.measurement.forward(realspace)
        return prediction

    def _gradg_analysis(self, preds):
        return self.measurement.adjoint(
            self.invcov.dot(sparse.csr_matrix(preds - self.data).T).toarray().flatten()
        )

    def _gradg_synthesis(self, preds):
        return self.transform.inverse_adjoint(self._gradg_analysis(preds))

    def _build_inverse_covariance_matrix(self, sig_d):
        if isinstance(sig_d, float) or isinstance(sig_d, int):
            return sparse.identity(len(self.data)).dot(1 / sig_d ** 2)
        elif sig_d.size == len(self.data) and len(sig_d.shape) == 1:
            return sparse.diags(1 / sig_d ** 2)
        elif len(sig_d.shape) == 2:
            return sparse.linalg.inv(sig_d)
        else:
            raise TypeError("sig_d must be a float scalar, vector or 2D matrix")


class WaveletTransformOperator(ForwardOperator):
    def __init__(self, data, sig_d, setting, L, B, J_min, dirs=1, spin=0):
        transform = WaveletTransform(
            L,
            B,
            J_min,
            dirs=dirs,
            spin=spin,
        )
        measurement = Identity(len(data), mw_size(L))

        if setting == "analysis":
            nparams = mw_size(L)
        else:
            nparams = transform.ncoefs

        super().__init__(
            data,
            sig_d,
            setting,
            transform=transform,
            measurement=measurement,
            nparams=nparams,
        )


class PathIntegralOperator(ForwardOperator):
    def __init__(self, pathmatrix, data, sig_d, setting, L, B, J_min, dirs=1, spin=0):
        transform = WaveletTransform(
            L,
            B,
            J_min,
            dirs=dirs,
            spin=spin,
        )
        measurement = PathIntegral(pathmatrix)

        if setting == "analysis":
            nparams = mw_size(L)
        else:
            nparams = transform.ncoefs

        super().__init__(
            data,
            sig_d,
            setting,
            transform=transform,
            measurement=measurement,
            nparams=nparams,
        )
