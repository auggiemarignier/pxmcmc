import numpy as np
from .utils import hard, soft


class PxMCMCParams:
    def __init__(
        self,
        algo="MYULA",
        lmda=3e-5,
        delta=1e-5,
        mu=1,
        sig_d=0.6718,
        sig_m=1,
        nsamples=int(1e6),
        nburn=int(1e3),
        ngap=int(1e2),
        hard=False,
        nparams=1,
        complex=False,
    ):
        self.algo = algo  # algorithm choice: MYULA or PxMALA
        self.lmda = lmda  # prox parameter. tuned to make proxf abritrarily close to f
        self.delta = delta  # Forward-Euler approximation step-size
        self.mu = mu  # regularization parameter
        self.sig_d = sig_d  # data errors, could be estimated hierarchically
        self.sig_m = sig_m  # model parameter errors
        self.nsamples = nsamples  # number of desired samples
        self.nburn = nburn  # burn-in size
        self.ngap = ngap  # Thinning parameter=number of iterations between samples. reduces correlations between samples
        self.hard = hard  # if true, hard thresholds model parameters
        self.nparams = nparams
        self.complex = complex


class PxMCMC:
    def __init__(self, forward, mcmcparams=PxMCMCParams()):
        """
        Initialises proximal MCMC algorithm.  Sets up the wavelet basis functions.  Calculates prefactors of the gradg function which are constant throughout the chain.
        """
        self.forward = forward
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
        w = np.random.randn(self.nparams)
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
        return -self.mu * sum(abs(X)) - sum((data - preds) ** 2) / (2 * self.sig_d ** 2)

    def calc_logtransition(self, X1, X2, proxf, gradg):
        """
        Calculates the transition probability of stepping from model X1 to model X2 i.e. q(X2|X1).  TO BE REWRITTEN
        """
        gradlogpiX1 = -(1 / self.lmda) * (X1 - proxf) - gradg
        return -(1 / 2 * self.delta) * sum(
            (X2 - X1 - (self.delta / 2) * gradlogpiX1) ** 2
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

    def mcmc(self):
        """
        Runs MCMC.  At present, logposteriors are becoming more and more negative and converging abnormally quickly.
        """
        logPi = np.zeros(self.nsamples)
        preds = np.zeros((self.nsamples, len(self.forward.data)))
        chain = np.zeros(
            (self.nsamples, self.nparams),
            dtype=np.complex if self.complex else np.float,
        )
        X_curr = np.random.normal(0, self.sig_m, self.nparams)
        if self.complex:
            X_curr = X_curr + np.random.normal(0, self.sig_m, self.nparams) * 1j
        if self.hard:
            X_curr = hard(X_curr)
        curr_preds = self.forward.forward(X_curr)
        i = 0
        while i < self.nsamples:
            if i >= self.nburn:
                if self.ngap == 0 or (i - self.nburn) % self.ngap == 0:
                    logPi[i] = self.logpi(X_curr, self.forward.data, curr_preds)
                    preds[i] = curr_preds
                    chain[i] = X_curr
            gradg = self.forward.calc_gradg(curr_preds)
            proxf = self.calc_proxf(X_curr)
            X_prop = self.chain_step(X_curr, proxf, gradg)
            if self.hard:
                X_prop = hard(X_prop)
            prop_preds = self.forward.forward(X_prop)

            if self.algo == "PxMALA":
                if self.MHaccept(X_curr, curr_preds, X_prop, prop_preds, proxf, gradg):
                    X_curr = X_prop
                    curr_preds = prop_preds
                    # i += 1
            if self.algo == "MYULA":
                X_curr = X_prop
                curr_preds = prop_preds
            print(f"\r{i+1}/{self.nsamples} - logposterior: {logPi[i]} - best:{np.max(logPi)}")
            i += 1
        self.logPi = logPi
        self.preds = preds
        self.chain = chain
