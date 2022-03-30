import numpy as np
from scipy.stats import laplace
from pxmcmc.utils import chebyshev1, cheb1der


class PxMCMCParams:
    """
    Class to store tuning and runtime parameters

    :param lmda: prox parameter. tuned to make proxf abritrarily close to f
    :param delta: Forward-Euler approximation step-size (MYULA, PxMALA)
    :param mu: regularization parameter
    :param s: max order of Chebyshev polynomials (SKROCK)
    :param nsamples: number of desired samples to be saved
    :param nburn: burn-in size
    :param ngap: Thinning parameter=number of iterations between saved samples
    :param complex: :code:`True` if sampled parameters are complex. Default :code:`False`.
    :param verbosity: print every :code:`verbosity` samples to console
    :param track: list of variables to keep track of
    """
    def __init__(
        self,
        lmda=3e-5,
        delta=1e-5,
        s=1,
        mu=1,
        nsamples=int(1e6),
        nburn=int(1e3),
        ngap=int(1e2),
        complex=False,
        verbosity=100,
        track=["logposterior", "L2", "prior", "chain"],
    ):
        self.lmda = lmda
        self.delta = delta
        self.mu = mu
        self.s = s
        self.nsamples = nsamples
        self.nburn = nburn
        self.ngap = ngap
        self.complex = complex
        self.verbosity = verbosity
        self.track = track


class PxMCMC:
    """
    Base class with general PxMCMC functions.
    Children of this class must implement a run function.

    :param forward: type :class:`forward.ForwardOperator` for forward modelling and gradient calculation.
    :param prior: object that implements prior and proximal calculations e.g. :class:`prior.L1`.
    :param mcmcparams: type :class:`.PxMCMCParams`
    """

    def __init__(self, forward, prior, mcmcparams=PxMCMCParams()):
        self.forward = forward
        self.prior = prior
        for attr in mcmcparams.__dict__.keys():
            setattr(self, attr, getattr(mcmcparams, attr))
        self._initialise_tracking_arrays()

    def run(self):
        raise NotImplementedError

    def logpi(self, X, preds):
        """
        Calculates the log(posterior), L2-norm and prior-norm of a model.

        :param X: vector of model parameters
        :param preds: vector of data predictions from model X
        """
        L2 = (self.forward.data - preds).T.dot(
            self.forward.invcov.dot((self.forward.data - preds))
        )
        prior = self.prior.prior(X)
        logPi = -self.mu * prior - L2
        return logPi, L2, prior

    def _gradlogpi(self, X, preds=None):
        gradf = (X - self.prior.proxf(X)) / self.lmda
        if preds is None:
            preds = self.forward.forward(X)
        gradg = self.forward.calc_gradg(preds)
        return -gradf - gradg

    def _print_progress(self, i, logpi, **kwargs):
        print(
            f"{i+1:,}/{self.nsamples:,} - logposterior: {logpi:.8e} - "
            + " - ".join([f"{k}: {kwargs[k]:.8e}" for k in kwargs]),
        )

    def _initial_sample(self):
        # TODO: flexibility for different priors
        X_curr = laplace.rvs(size=self.forward.nparams)
        if self.complex:
            X_curr = X_curr + laplace.rvs(size=self.forward.nparams) * 1j
        curr_preds = self.forward.forward(X_curr)
        return X_curr, curr_preds

    def _initialise_tracking_arrays(self):
        if "logposterior" in self.track:
            self.logPi = np.zeros(self.nsamples)
        if "predictions" in self.track:
            self.preds = np.zeros(
                (self.nsamples, len(self.forward.data)), dtype=float
            )
        if "chain" in self.track:
            self.chain = np.zeros(
                (self.nsamples, self.forward.nparams),
                dtype=complex if self.complex else float,
            )
        if "L2" in self.track:
            self.L2s = np.zeros(self.nsamples, dtype=float)
        if "prior" in self.track:
            self.priors = np.zeros(self.nsamples, dtype=float)

    def _tracking(self, j, X_curr, curr_preds, logPi, L2, prior):
        if hasattr(self, "logPi"):
            self.logPi[j] = logPi
        if hasattr(self, "L2s"):
            self.L2s[j] = L2
        if hasattr(self, "priors"):
            self.priors[j] = prior
        if hasattr(self, "preds"):
            self.preds[j] = curr_preds
        if hasattr(self, "chain"):
            self.chain[j] = X_curr


class MYULA(PxMCMC):
    """
    Implements the MYULA chain
    """
    def __init__(self, forward, prox, mcmcparams=PxMCMCParams()):
        super().__init__(forward, prox, mcmcparams)

    def run(self):
        """
        Run the algorithm
        """
        i = 0  # total samples
        j = 0  # saved samples (excludes burn-in and thinned samples)
        X_curr, curr_preds = self._initial_sample()
        while j < self.nsamples:
            gradg = self.forward.calc_gradg(curr_preds)
            proxf = self.prior.proxf(X_curr)
            X_prop = self.chain_step(X_curr, proxf, gradg)
            prop_preds = self.forward.forward(X_prop)

            X_curr = X_prop
            curr_preds = prop_preds

            if i >= self.nburn:
                if self.ngap == 0 or (i - self.nburn) % self.ngap == 0:
                    logPi, L2, prior = self.logpi(X_curr, curr_preds)
                    self._tracking(j, X_curr, curr_preds, logPi, L2, prior)
                    j += 1
                if self.verbosity > 0 and (i + 1) % self.verbosity == 0:
                    self._print_progress(
                        j - 1,
                        self.logPi[j - 1],
                        L2=self.L2s[j - 1],
                        prior=self.priors[j - 1],
                    )
            else:
                if self.verbosity > 0 and (i + 1) % self.verbosity == 0:
                    print("Burning in...")
            i += 1

        print("\nDONE")

    def chain_step(self, X, proxf, gradg):
        """
        Takes a step in the chain.

        :param X: current sample
        :param proxf: prox of prior probablity
        :param grardg: gradient of data fidelity
        """
        w = np.random.randn(self.forward.nparams)
        if self.complex:
            w = w + np.random.randn(self.forward.nparams) * 1j
        return (
            (1 - self.delta / self.lmda) * X
            + (self.delta / self.lmda) * proxf
            - self.delta * gradg
            + np.sqrt(2 * self.delta) * w
        )


class PxMALA(MYULA):
    """
    Implements the PxMALA algorithm (MYULA + MH acceptance)

    :param bool tune_delta: if :code:`True`, tunes  :code:`delta` parameter to achieve an acceptance probablity of 0.5
    
    .. todo::

       Option to tune :code:`delta` to any desired acceptance probability.
    """
    def __init__(self, forward, prox, mcmcparams=PxMCMCParams(), tune_delta=True):
        super().__init__(forward, prox, mcmcparams)
        self.tune_delta = tune_delta

    def run(self):
        """
        Run the algorithm
        """
        self.acceptance_trace = []
        self.deltas_trace = [self.delta]
        i = 0
        j = 0
        X_curr, curr_preds = self._initial_sample()
        gradg_curr = self.forward.calc_gradg(curr_preds)
        proxf_curr = self.prior.proxf(X_curr)
        logpiXc, L2Xc, priorXc = self.logpi(X_curr, curr_preds)
        while j < self.nsamples:
            X_prop = self.chain_step(X_curr, proxf_curr, gradg_curr)
            prop_preds = self.forward.forward(X_prop)
            gradg_prop = self.forward.calc_gradg(prop_preds)
            proxf_prop = self.prior.proxf(X_prop)

            logtransXcXp = self.calc_logtransition(
                X_curr, X_prop, proxf_curr, gradg_curr
            )
            logtransXpXc = self.calc_logtransition(
                X_prop, X_curr, proxf_prop, gradg_prop
            )
            logpiXp, L2Xp, priorXp = self.logpi(X_prop, prop_preds)

            logalpha = logtransXpXc + logpiXp - logtransXcXp - logpiXc
            accept = np.log(np.random.rand()) < logalpha
            if accept:
                X_curr = X_prop
                curr_preds = prop_preds
                gradg_curr = gradg_prop
                proxf_curr = proxf_prop
                logpiXc = logpiXp
                L2Xc = L2Xp
                priorXc = priorXp
                self.acceptance_trace.append(1)
            else:
                self.acceptance_trace.append(0)

            if self.tune_delta:
                self._tune_delta(i)
                self.deltas_trace.append(self.delta)

            if i >= self.nburn:
                if (self.ngap == 0 or (i - self.nburn) % self.ngap == 0) and accept:
                    self._tracking(j, X_curr, curr_preds, logpiXc, L2Xc, priorXc)
                    j += 1
            if self.verbosity > 0 and (i + 1) % self.verbosity == 0:
                self._print_progress(
                    j - 1,
                    logpiXc,
                    L2=L2Xc,
                    prior=priorXc,
                    acceptanceRate=np.mean(self.acceptance_trace),
                )
            i += 1
        print("\nDONE")

    def _tune_delta(self, i):
        delta = self.delta * (1 + (self.acceptance_trace[i] - 0.5) / ((i + 1) ** 0.75))
        self.delta = min(max(delta, self.lmda * 1e-8), self.lmda / 2)

    def calc_logtransition(self, X1, X2, proxf, gradg):
        """
        Calculates the transition probability of stepping from model X1 to model X2 i.e. q(X2|X1).
        """
        gradlogpiX1 = -((X1 - proxf) / self.lmda) - gradg
        return (
            -(1 / 2 * self.delta)
            * np.sum((X2 - X1 - (self.delta / 2) * gradlogpiX1) ** 2) ** 2
        )


class SKROCK(PxMCMC):
    """
    Implements the SKROCK algorithm

    .. warning::
    
       This algorithm hasn't been used in practice yet from this code package
    """
    def __init__(self, forward, prox, mcmcparams=PxMCMCParams()):

        super().__init__(forward, prox, mcmcparams=mcmcparams)
        self.eta = 0.05
        self.omega_0 = 1 + self.eta / (self.s * self.s)
        self.omega_1 = chebyshev1(self.omega_0, self.s) / cheb1der(self.omega_0, self.s)
        self._recursion_coefs()

    def run(self):
        """
        Run the algorithm
        """
        i = 0  # total samples
        j = 0  # saved samples (excludes burn-in and thinned samples)
        X_curr, curr_preds = self._initial_sample()
        while j < self.nsamples:
            X_prop = self.chain_step(X_curr)
            prop_preds = self.forward.forward(X_prop)

            X_curr = X_prop
            curr_preds = prop_preds

            if i >= self.nburn:
                if self.ngap == 0 or (i - self.nburn) % self.ngap == 0:
                    logPi, L2, prior = self.logpi(X_curr, curr_preds)
                    self._tracking(j, X_curr, curr_preds, logPi, L2, prior)
                    j += 1
            if self.verbosity > 0 and (i + 1) % self.verbosity == 0:
                self._print_progress(
                    j - 1,
                    self.logPi[j - 1],
                    L2=self.L2s[j - 1],
                    prior=self.priors[j - 1],
                )
            i += 1

        print("\nDONE")

    def chain_step(self, X):
        """
        Takes a step in the chain.

        :param X: current sample
        """
        Z = np.random.randn(len(X))
        if self.complex:
            Z = Z + np.random.randn(len(X)) * 1j
        return self._K_recursion(X, self.s, Z)

    def _K_recursion(self, X, s, Z):
        if s == 0:
            return X
        elif s == 1:
            return (
                X
                + self.mus[1]
                * self.delta
                * self._gradlogpi(X + self.nus[1] * np.sqrt(2 * self.delta) * Z)
                + self.ks[1] * np.sqrt(2 * self.delta) * Z
            )
        else:
            return (
                self.mus[s]
                * self.delta
                * self._gradlogpi(self._K_recursion(X, s - 1, Z))
                + self.nus[s] * self._K_recursion(X, s - 1, Z)
                + self.ks[s]
                - self._K_recursion(X, s - 2, Z)
            )

    def _recursion_coefs(self):
        self.mus = np.zeros(self.s + 1)
        self.nus = np.zeros(self.s + 1)
        self.ks = np.zeros(self.s + 1)

        self.mus[1] = self.omega_1 / self.omega_0
        self.nus[1] = self.s * self.omega_1 / 2
        self.ks[1] = self.s * self.omega_1 / self.omega_0

        for j in range(2, self.s + 1):
            cheb_ratio = chebyshev1(self.omega_0, j - 1) / chebyshev1(self.omega_1, j)
            self.mus[j] = 2 * self.omega_1 * cheb_ratio
            self.nus[j] = 2 * self.omega_0 * cheb_ratio
            self.ks[j] = 1 - self.nus[0]
