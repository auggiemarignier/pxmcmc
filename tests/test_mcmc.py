import pytest
from pytest_cases import parametrize_with_cases

from pxmcmc.mcmc import MYULA, PxMALA, SKROCK, PxMCMCParams
from pxmcmc.measurements import Identity
from pxmcmc.transforms import IdentityTransform
from pxmcmc.forward import ForwardOperator
from pxmcmc.prox import L1


@pytest.fixture
def forwardop(simpledata, setting, sig_d):
    ndata = nparams = len(simpledata)
    transform = IdentityTransform()
    measurement = Identity(ndata, nparams)
    return ForwardOperator(
        simpledata.real, sig_d, setting, transform, measurement, nparams=nparams
    )


@pytest.fixture
def prox(forwardop, setting):
    return L1(
        setting, forwardop.transform.inverse, forwardop.transform.inverse_adjoint, 1
    )


@pytest.fixture
def mcmcparams():
    return PxMCMCParams(nsamples=100, nburn=10, ngap=5, verbosity=0, s=5)


def case_myula(forwardop, prox, mcmcparams):
    return MYULA(forwardop, prox, mcmcparams)


def case_pxmala(forwardop, prox, mcmcparams):
    return PxMALA(forwardop, prox, mcmcparams)


def case_skrock(forwardop, prox, mcmcparams):
    return SKROCK(forwardop, prox, mcmcparams)


@parametrize_with_cases("algo", cases=".")
def test_algorithms(algo):
    algo.run()
