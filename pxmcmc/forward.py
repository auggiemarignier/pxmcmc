from pys2let import mw_size
from scipy import sparse

from pxmcmc.measurements import Identity, PathIntegral
from pxmcmc.transforms import WaveletTransform


class ForwardOperator:
    """
    Base Forward operator. Combines a transform and a measurement operator.

    :param data: observed data vector
    :param sig_d: observed data error.  Can be a single float, vector or covariance matrix.
    :param string setting: `analysis` or `synthesis`
    :param transform: type :class:`transforms.Transform` to transform between bases
    :param measurement: type :class:`measurements.Measurement` to predict observed data
    :param int nparams: number of sampled parameters/dimensions.  Depends on parameterisation and setting.
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
        """
        Forward modelling, with or without basis transformation depending on :code:`setting`.

        :param X: MCMC sample
        :return: data predictions
        """
        if self.setting == "analysis":
            return self._forward_analysis(X)
        else:
            return self._forward_synthesis(X)

    def calc_gradg(self, preds):
        """
        Calculates gradient of data fidelity.  Assumes Gaussian data errors
        
        :param preds: data predictions of current MCMC sample
        :return: gradient of Gaussian data fidelity
        """
        if self.setting == "analysis":
            return self._gradg_analysis(preds)
        else:
            return self._gradg_synthesis(preds)

    def _forward_analysis(self, X):
        return self.measurement.forward(X)

    def _forward_synthesis(self, X):
        return self.measurement.forward(self.transform.inverse(X))

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
    """
    Forward operator with a spherical wavelet transform and identity operator.

    :param int L: angular bandlimit
    :param float B: wavelet scale parameter
    :param int J_min: minimum wavelet scale
    :param int dirs: azimuthal bandlimit for directional wavelets
    :param int spin: spin number of spherical signal
    """
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
    """
    Forward operator with a spherical wavelet transform and a path integral measurement operator.

    .. todo::
       Since the measurement operator is just a matrix multiplication, can be renamed to something more generic.

    :param array pathmatrix: matrix describing a set of paths on the sphere
    :param int L: angular bandlimit
    :param float B: wavelet scale parameter
    :param int J_min: minimum wavelet scale
    :param int dirs: azimuthal bandlimit for directional wavelets
    :param int spin: spin number of spherical signal
    """
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
