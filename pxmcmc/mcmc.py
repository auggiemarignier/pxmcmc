import numpy as np
from .utils import hard, soft


class PxMCMCParams:
    def __init__(
        self,
        algo="MYULA",
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
        self.algo = algo  # algorithm choice: MYULA or PxMALA
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

    def logpi(self, X, data, preds):
        """
        Calculates the log(posterior) of a model X.  Takes in the predictions, preds, of X.
        """
        L2 = sum(abs((data - preds)) ** 2)
        L1 = sum(abs(X))
        logPi = -self.mu * L1 - L2 / (2 * self.forward.sig_d ** 2)
        return logPi, L2, L1

    def calc_logtransition(self, X1, X2, proxf, gradg):
        """
        Calculates the transition probability of stepping from model X1 to model X2 i.e. q(X2|X1).  TO BE REWRITTEN
        """
        gradlogpiX1 = -((X1 - proxf) / self.lamda) - gradg
        return (
            -(1 / 2 * self.delta)
            * np.sum((X2 - X1 - (self.delta / 2) * gradlogpiX1) ** 2) ** 2
        )  # not sure about sum of squares here

    def accept_prob(self, X_curr, curr_preds, X_prop, prop_preds, proxf, gradg):
        """
        Calculates the acceptance probability of the propsed model X_pop, as a ratio of the transtion probabilities times the ratio of the posteriors.  Strictly speaking, the returned value should be min(0,p)=min(1,e^p) but this makes no difference in the MH acceptance step.
        """
        logtransXcXp = self.calc_logtransition(X_curr, X_prop, proxf, gradg)
        logtransXpXc = self.calc_logtransition(X_prop, X_curr, proxf, gradg)
        logpiXc = self.logpi(X_curr, curr_preds)
        logpiXp = self.logpi(X_prop, prop_preds)
        p = np.real(logtransXpXc + logpiXp - logtransXcXp + logpiXc)
        assert not np.isnan(p)
        return p

    def MHaccept(self, X_curr, curr_preds, X_prop, prop_preds, proxf, gradg):
        """
        Metropolis-Hastings acceptance step.  Accept if the acceptance probability alpha is greater than a random number.
        """
        alpha = self.accept_prob(X_curr, curr_preds, X_prop, prop_preds, proxf, gradg)
        u = np.log(np.random.rand())
        return True if u <= alpha else False

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

    def mcmc(self):
        """
        Runs MCMC.  At present, logposteriors are becoming more and more negative and converging abnormally quickly.
        """
        logPi = np.zeros(self.nsamples)
        preds = np.zeros(
            (self.nsamples, len(self.forward.data)),
            dtype=np.complex if self.complex else np.float,
        )
        chain = np.zeros(
            (self.nsamples, self.forward.nparams),
            dtype=np.complex if self.complex else np.float,
        )
        L2s = np.zeros(self.nsamples, dtype=np.float)
        L1s = np.zeros(self.nsamples, dtype=np.float)

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

            if self.algo == "PxMALA":
                if self.MHaccept(X_curr, curr_preds, X_prop, prop_preds, proxf, gradg):
                    X_curr = X_prop
                    curr_preds = prop_preds
            if self.algo == "MYULA":
                X_curr = X_prop
                curr_preds = prop_preds

            if i >= self.nburn:
                if self.ngap == 0 or (i - self.nburn) % self.ngap == 0:
                    logPi[j], L2s[j], L1s[j] = self.logpi(
                        X_curr, self.forward.data, curr_preds
                    )
                    preds[j] = curr_preds
                    chain[j] = X_curr
                    j += 1
            if (i + 1) % self.verbosity == 0:
                self._print_progress(j - 1, logPi[j - 1], L2s[j - 1], L1s[j - 1])
            i += 1

        self.logPi = logPi
        self.preds = preds
        self.chain = chain
        self.L2s = L2s
        self.L1s = L1s
        print(f"\nDONE")
