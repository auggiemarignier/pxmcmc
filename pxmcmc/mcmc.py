import numpy as np
from .utils import soft


class PxMCMCParams:
    def __init__(
        self,
        lmda=3e-5,
        delta=1e-5,
        mu=1,
        sig_m=1,
        nsamples=int(1e6),
        nburn=int(1e3),
        ngap=int(1e2),
        complex=False,
        verbosity=100,
    ):
        self.lmda = lmda  # prox parameter. tuned to make proxf abritrarily close to f
        self.delta = delta  # Forward-Euler approximation step-size
        self.mu = mu  # regularization parameter
        self.sig_m = sig_m  # model parameter errors
        self.nsamples = nsamples  # number of desired samples
        self.nburn = nburn  # burn-in size
        self.ngap = ngap  # Thinning parameter=number of iterations between samples. reduces correlations between samples
        self.complex = complex
        self.verbosity = verbosity  # print every verbosity samples


class PxMCMC:
    def __init__(self, forward, mcmcparams=PxMCMCParams(), X_func=None):
        """
        Initialises proximal MCMC algorithm.  Sets up the wavelet basis functions.  Calculates prefactors of the gradg function which are constant throughout the chain.
        """
        self.forward = forward
        self.X_func = X_func
        for attr in mcmcparams.__dict__.keys():
            setattr(self, attr, getattr(mcmcparams, attr))
        self._initialise_tracking_arrays()

    def calc_proxf(self, X):
        """
        Calculates the prox of the sparsity regularisation term.
        """
        return soft(X, self.lmda * self.mu)

    def chain_step(self, X, proxf, gradg):
        """
        Takes a step in the chain.
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

    def logpi(self, X, preds):
        """
        Calculates the log(posterior), L2-norm and L1-norm of a model X.
        """
        L2 = sum(abs((self.forward.data - preds)) ** 2)
        L1 = sum(abs(X))
        logPi = -self.mu * L1 - L2 / (2 * self.forward.sig_d ** 2)
        return logPi, L2, L1

    def calc_logtransition(self, X1, X2, proxf, gradg):
        """
        Calculates the transition probability of stepping from model X1 to model X2 i.e. q(X2|X1).
        """
        gradlogpiX1 = -((X1 - proxf) / self.lmda) - gradg
        return (
            -(1 / 2 * self.delta)
            * np.sum((X2 - X1 - (self.delta / 2) * gradlogpiX1) ** 2) ** 2
        )

    def _print_progress(self, i, logpi, l2, l1):
        if i < self.nburn:
            print(f"\rBurning in", end="")
        else:
            print(
                f"\r{i+1:,}/{self.nsamples:,} - logposterior: {logpi:.8f} - L2: {l2:.8f} - L1: {l1:.8f}",
                end="",
            )

    def _initial_sample(self):
        X_curr = np.random.normal(0, self.sig_m, self.forward.nparams)
        if self.complex:
            X_curr = X_curr + np.random.normal(0, self.sig_m, self.forward.nparams) * 1j
        if self.X_func is not None:
            X_curr = self.X_func(X_curr)
        curr_preds = self.forward.forward(X_curr)
        return X_curr, curr_preds

    def _initialise_tracking_arrays(self):
        self.logPi = np.zeros(self.nsamples)
        self.preds = np.zeros(
            (self.nsamples, len(self.forward.data)),
            dtype=np.complex if self.complex else np.float,
        )
        self.chain = np.zeros(
            (self.nsamples, self.forward.nparams),
            dtype=np.complex if self.complex else np.float,
        )
        self.L2s = np.zeros(self.nsamples, dtype=np.float)
        self.L1s = np.zeros(self.nsamples, dtype=np.float)

    def myula(self):
        i = 0  # total samples
        j = 0  # saved samples (excludes burn-in and thinned samples)
        X_curr, curr_preds = self._initial_sample()
        while j < self.nsamples:
            gradg = self.forward.calc_gradg(curr_preds)
            proxf = self.calc_proxf(X_curr)
            X_prop = self.chain_step(X_curr, proxf, gradg)
            if self.X_func is not None:
                X_prop = self.X_func(X_prop)
            prop_preds = self.forward.forward(X_prop)

            X_curr = X_prop
            curr_preds = prop_preds

            if i >= self.nburn:
                if self.ngap == 0 or (i - self.nburn) % self.ngap == 0:
                    self.logPi[j], self.L2s[j], self.L1s[j] = self.logpi(
                        X_curr, curr_preds
                    )
                    self.preds[j] = curr_preds
                    self.chain[j] = X_curr
                    j += 1
            if (i + 1) % self.verbosity == 0:
                self._print_progress(
                    j - 1, self.logPi[j - 1], self.L2s[j - 1], self.L1s[j - 1]
                )
            i += 1

        print(f"\nDONE")

    def pxmala(self):
        i = 0
        j = 0
        X_curr, curr_preds = self._initial_sample()
        gradg_curr = self.forward.calc_gradg(curr_preds)
        proxf_curr = self.calc_proxf(X_curr)
        logpiXc, L2Xc, L1Xc = self.logpi(X_curr, curr_preds)
        while j < self.nsamples:
            X_prop = self.chain_step(X_curr, proxf_curr, gradg_curr)
            if self.X_func is not None:
                X_prop = self.X_func(X_prop)
            prop_preds = self.forward.forward(X_prop)
            gradg_prop = self.forward.calc_gradg(prop_preds)
            proxf_prop = self.calc_proxf(X_prop)

            logtransXcXp = self.calc_logtransition(
                X_curr, X_prop, proxf_curr, gradg_curr
            )
            logtransXpXc = self.calc_logtransition(
                X_prop, X_curr, proxf_prop, gradg_prop
            )
            logpiXp, L2Xp, L1Xp = self.logpi(X_prop, prop_preds)

            logalpha = logtransXpXc + logpiXp - logtransXcXp - logpiXc
            if np.log(np.random.rand()) < logalpha:
                X_curr = X_prop
                curr_preds = prop_preds
                gradg_curr = gradg_prop
                proxf_curr = proxf_prop
                logpiXc = logpiXp
                L2Xc = L2Xp
                L1Xc = L1Xp
                j += 1

            if i >= self.nburn:
                if self.ngap == 0 or (i - self.nburn) % self.ngap == 0:
                    self.logPi[j - 1] = logpiXc
                    self.L2s[j - 1] = L2Xc
                    self.L1s[j - 1] = L1Xc
                    self.preds[j - 1] = curr_preds
                    self.chain[j - 1] = X_curr
            if (i + 1) % self.verbosity == 0:
                self._print_progress(j - 1, logpiXc, L2Xc, L1Xc)
            i += 1
