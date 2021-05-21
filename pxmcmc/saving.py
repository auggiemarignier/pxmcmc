import os
import h5py


def save_mcmc(
    mcmc, params, outpath, filename="outputs", **kwargs,
):
    """
    Saves the MCMC run in a :code:`.hd5f` file. Any variable defined by a :class:`mcmc.PxMCMC` instance's :code:`tracking` option will be saved as a `dataset <https://docs.h5py.org/en/stable/high/dataset.html>`_.  Runtime parameters and other things that the user may wish to save are saved as  `attributes <https://docs.h5py.org/en/stable/high/attr.html>`_.

    :param mcmc: type :class:`mcmc.PxMCMC` after its :meth:`run` has been called.
    :param params: type :class:`mcmc.PxMCMCParams` used to setup the :code:`mcmc` instance.
    :param string outpath: directory in which file should be saved.
    :param string filename: filename without extension.
    :param \**kwargs: additional bits users want to save.
    """
    with h5py.File(os.path.join(outpath, f"{filename}.hdf5"), "w") as f:
        if hasattr(mcmc, "logPi"):
            f.create_dataset("logposterior", data=mcmc.logPi)
        if hasattr(mcmc, "preds"):
            f.create_dataset("predictions", data=mcmc.preds)
        if hasattr(mcmc, "chain"):
            f.create_dataset("chain", data=mcmc.chain)
        if hasattr(mcmc, "L2s"):
            f.create_dataset("L2s", data=mcmc.L2s)
        if hasattr(mcmc, "priors"):
            f.create_dataset("priors", data=mcmc.priors)
        if hasattr(mcmc, "acceptance_trace"):
            f.create_dataset("acceptances", data=mcmc.acceptance_trace, dtype="i1")
        if hasattr(mcmc, "deltas_trace"):
            f.create_dataset("deltas", data=mcmc.deltas_trace)

        for attr in params.__dict__.keys():
            f.attrs[attr] = getattr(params, attr)
        for k, v in kwargs.items():
            f.attrs[k] = v
