from pxmcmc.mcmc import PxMCMC
import pytest


@pytest.fixture
def mcmc():
    return PxMCMC()


def test_PxMCMCConstructor(mcmc):
    for attr in [
        "algo",
        "lmda",
        "delta",
        "mu",
        "sig_d",
        "sig_m",
        "nsamples",
        "nburn",
        "ngap",
        "hard",
        "nparams"
    ]:
        assert hasattr(mcmc, attr)
