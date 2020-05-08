from pxmcmc.mcmc import PxMCMC


def test_PxMCMCConstructor():
    for attr in [
        "L",
        "B",
        "dirs",
        "spin",
        "J_min",
        "J_max",
        "nscales",
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
    ]:
        mcmc = PxMCMC()
        assert hasattr(mcmc, attr)
