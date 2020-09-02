from pys2let import mw_size

from pxmcmc.measurements import Identity
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
        self.sig_d = sig_d
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
        return self.measurement.adjoint((preds - self.data) / (self.sig_d ** 2))

    def _gradg_synthesis(self, preds):
        return self.transform.inverse_adjoint(
            self.measurement.adjoint((preds - self.data) / (self.sig_d ** 2))
        )


class WaveletTransformOperator(ForwardOperator):
    def __init__(self, data, sig_d, setting, L, B, J_min, Nside=None, dirs=1, spin=0):
        super().__init__(data, sig_d, setting)

        map_type = "pixel_mw"
        self.transform = WaveletTransform(
            L,
            B,
            J_min,
            Nside,
            dirs,
            spin,
            inv_in_type=map_type,
            inv_out_type=map_type,
            inv_adj_in_type=map_type,
            inv_adj_out_type=map_type,
        )
        self.measurement = Identity(len(data), mw_size(L))
        if setting == "analysis":
            self.nparams = mw_size(L)
        else:
            self.nparams = mw_size(L) * (self.transform.nscales + 1)
