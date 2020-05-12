import os
import h5py


def save_mcmc(mcmc, params, outpath, **kwargs):
    with h5py.File(os.path.join(outpath, "outputs.hdf5")) as f:
        f.create_dataset("logposterior", data=mcmc.logPi)
        f.create_dataset("predictions", data=mcmc.preds)
        f.create_dataset("chain", data=mcmc.chain)
        f.create_dataset("L2s", data=mcmc.L2s)
        f.create_dataset("L1s", data=mcmc.L1s)

        for attr in params.__dict__.keys():
            f.attrs[attr] = getattr(params, attr)
        for k, v in kwargs.items():
            f.attrs[k] = v
