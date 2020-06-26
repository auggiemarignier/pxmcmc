import os
import h5py


def save_mcmc(mcmc, params, outpath, filename="outputs", **kwargs):
    with h5py.File(os.path.join(outpath, f"{filename}.hdf5"), "w") as f:
        f.create_dataset("logposterior", data=mcmc.logPi)
        f.create_dataset("predictions", data=mcmc.preds)
        f.create_dataset("chain", data=mcmc.chain)
        f.create_dataset("L2s", data=mcmc.L2s)
        f.create_dataset("L1s", data=mcmc.L1s)
        if hasattr(mcmc, "acceptance_trace"):
            f.create_dataset("acceptances", data=mcmc.acceptance_trace, dtype="i1")
        if hasattr(mcmc, "deltas_trace"):
            f.create_dataset("deltas", data=mcmc.deltas_trace)

        for attr in params.__dict__.keys():
            f.attrs[attr] = getattr(params, attr)
        for k, v in kwargs.items():
            f.attrs[k] = v
