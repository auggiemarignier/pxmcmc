import numpy as np
import pys2let
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
    ):
        self.algo = algo  # algorithm choice: MYULA or PxMALA
        self.lmda = (
            lmda  # proximity parameter. tuned to make proxf abritrarily close to f
        )
        self.delta = delta  # Forward-Euler approximation step-size
        self.mu = mu  # regularization parameter
        self.sig_d = sig_d  # data errors, could be estimated hierarchically
        self.sig_m = sig_m  # model parameter errors
        self.nsamples = nsamples  # number of desired samples
        self.nburn = nburn  # burn-in size
        self.ngap = ngap  # Thinning parameter=number of iterations between samples. reduces correlations between samples
        self.hard = hard  # if true, hard thresholds model parameters


class WaveletParams:
    def __init__(self, L=10, B=1.5, dirs=1, spin=0, J_min=2):
        self.L = L  # maximum angular order
        self.B = B  # wavelet parameter
        self.dirs = dirs  # number of directions for directional wavelets
        self.spin = spin  # spin of the field
        self.J_min = J_min  # minimum wavelet scale
        self.J_max = pys2let.pys2let_j_max(B, L, J_min)  # maximum wavelet scale
        self.nscales = self.J_max - self.J_min + 1  # number of wavelet scales


class PxMCMC:
    def __init__(self, mcmcparams=PxMCMCParams(), waveletparams=WaveletParams()):
        """
        Initialises proximal MCMC algorithm.  Sets up the wavelet basis functions.  Calculates prefactors of the gradg function which are constant throughout the chain.
        """
        for attr in mcmcparams.__dict__.keys():
            print(attr)
            setattr(self, attr, getattr(mcmcparams, attr))
        for attr in waveletparams.__dict__.keys():
            print(attr)
            setattr(self, attr, getattr(waveletparams, attr))

        B = self.B
        L = self.L
        dirs = self.dirs
        spin = self.spin
        J_min = self.J_min

        phi_l, psi_lm = pys2let.wavelet_tiling(B, L + 1, dirs, spin, J_min)
        psi_lm = psi_lm[:, J_min:]
        phi_lm = np.zeros(((L + 1) ** 2, 1), dtype=np.complex)
        for ell in range(L + 1):
            phi_lm[ell * ell + ell] = phi_l[ell]
        self.basis = np.concatenate((phi_lm, psi_lm), axis=1)

        self.n_lm = self.basis.shape[0]
        self.nb = self.basis.shape[1]
        self.nparams = self.n_lm * self.nb
        assert type(self.nparams) == int and self.nparams > 0

    def calc_proxf(self, X):
        """
        Calculates the prox of the sparsity regularisation term.
        """
        return soft(X, self.params.lmda * self.params.mu / 2)

    def chain_step(self, X, proxf, gradg):
        """
        Takes a step in the chain.
        """
        w = np.random.randn(self.nparams)
        return (
            (1 - self.params.delta / self.params.lmda) * X
            + (self.params.delta / self.params.lmda) * proxf
            - self.params.delta * gradg
            + np.sqrt(2 * self.params.delta) * w
        )

    def logpi(self, X, data, preds):
        """
        Calculates the log(posterior) of a model X.  Takes in the predictions, preds, of X.
        """
        return -self.params.mu * sum(abs(X)) - sum((data - preds) ** 2) / (
            2 * self.params.sig_d ** 2
        )

    def calc_logtransition(self, X1, X2, proxf, gradg):
        """
        Calculates the transition probability of stepping from model X1 to model X2 i.e. q(X2|X1).  TO BE REWRITTEN
        """
        gradlogpiX1 = -(1 / self.params.lmda) * (X1 - proxf) - gradg
        return -(1 / 2 * self.params.delta) * sum(
            (X2 - X1 - (self.params.delta / 2) * gradlogpiX1) ** 2
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

    def accept(self, alpha):
        """
        Metropolis-Hastings acceptance step.  Accept if the acceptance probability alpha is greater than a random number.
        """
        u = np.log(np.random.rand())
        return True if u <= alpha else False

    def mcmc(self, data):
        """
        Runs MCMC.  At present, logposteriors are becoming more and more negative and converging abnormally quickly.
        """
        logPi = np.zeros(self.params.nsamples + 1)
        preds = np.zeros((self.params.nsamples + 1, len(data)))
        chain = np.zeros((self.params.nsamples + 1, self.nparams), dtype=np.complex)
        X_curr = (
            np.random.normal(0, self.params.sig_m, self.nparams)
            + np.random.normal(0, self.params.sig_m, self.nparams) * 1j
        )
        if self.params.hard:
            X_curr = hard(X_curr)
        curr_preds = self.forward(X_curr)
        i = 0
        while i < self.params.nsamples:
            if i >= self.params.nburn:
                if (
                    self.params.ngap == 0
                    or (i - self.params.nburn) % self.params.ngap == 0
                ):
                    logPi[i] = self.logpi(X_curr, curr_preds)
                    preds[i] = curr_preds
                    chain[i] = X_curr
            gradg = self.calc_gradg(curr_preds)
            proxf = self.calc_proxf(X_curr)
            X_prop = self.chain_step(X_curr, proxf, gradg)
            if self.params.hard:
                X_prop = hard(X_prop)
            prop_preds = self.forward(X_prop)

            if self.algo == "PxMALA":
                alpha = self.accept_prob(
                    X_curr, curr_preds, X_prop, prop_preds, proxf, gradg
                )
                print(alpha)
                if self.accept(alpha):
                    X_curr = X_prop
                    curr_preds = prop_preds
                    # i += 1
            if self.algo == "MYULA":
                X_curr = X_prop
                curr_preds = prop_preds
            print(f"{i+1}/{self.params.nsamples} - logposterior: {logPi[i]}")
            i += 1
        self.logPi = logPi
        self.preds = preds
        self.chain = chain
