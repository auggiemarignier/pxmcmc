import numpy as np


def flatten_mlm(wav_lm, scal_lm):
    """
    Takes a set of wavelet and scaling coefficients and flattens them into a single vector
    """
    buff = wav_lm.ravel(order="F")
    mlm = np.concatenate((scal_lm, buff))
    return mlm


def expand_mlm(mlm, nscales):
    """
    Sepatates scaling and wavelet coefficients from a single vector to separate arrays.
    """
    v_len = mlm.size // (nscales + 1)
    assert v_len > 0
    scal_lm = mlm[:v_len]
    wav_lm = np.zeros((v_len, nscales), dtype=np.complex)
    for i in range(nscales):
        wav_lm[:, i] = mlm[(i + 1) * v_len : (i + 2) * v_len]
    return wav_lm, scal_lm


def soft(X, T=0.1):
    """
    Soft thresholding of a vector X with threshold T.  If Xi is less than T, then soft(Xi) = 0, otherwise soft(Xi) = Xi-T.
    """
    t = np.sign(X) * (np.abs(X) - T)
    t[np.abs(X) <= T] = 0
    return t


def hard(X, T=0.1):
    """
    Hard thresholding of a vector X with fraction threshold T. T is the fraction kept, i.e. the largest 100T% absolute values are kept, the others are thresholded to 0.
    TODO: What happens when all elements of X are equal?
    """
    X_srt = np.sort(abs(X))
    thresh_ind = int(T * len(X))
    thresh_val = X_srt[-thresh_ind]
    X[abs(X) < thresh_val] = 0
    return X
