import numpy as np
import pyssht

from pxmcmc.utils import _multires_bandlimits


def credible_interval_range(chain, alpha=0.05):
    """
    Calculates the range of the credible interval of MCMC samples at a given confidence level

    :param chain: :math:`N_{\mathrm{samples}}\\times N_{\mathrm{params}}` array containing MCMC samples
    :param float `alpha`: confidence level
    :return: array of credible interval ranges for each parameter
    """
    quantiles = np.quantile(chain, (alpha / 2, 1 - alpha / 2), axis=0)
    return np.diff(quantiles, axis=0)[0]


def wavelet_credible_interval_range(chain, L, B, J_min, alpha=0.05):
    """
    Maps the credible interval range at different wavelet scales. Assumes each sample in chain is a set of wavelet coefficients in multiresolution format. Maps are returned in MW (theta, phi) format.

    :param chain: :math:`N_{\mathrm{samples}}\\times N_{\mathrm{params}}` array containing MCMC samples
    :param int L: angular bandlimit
    :param float B: wavelet scale parameter
    :param int J_min: minimum wavelet scale
    :param float `alpha`: confidence level
    :return: array of credible interval ranges for each wavelet coefficient
    """
    bls = _multires_bandlimits(L, B, J_min)
    scale_start = 0
    wav_ci_ranges = []
    for i, bl in enumerate(bls):
        scale_length = pyssht.sample_length(bl)
        wav = chain[:, scale_start : scale_start + scale_length]
        wav_ci_ranges.append(
            credible_interval_range(wav, alpha).reshape(pyssht.sample_shape(bl))
        )
        scale_start += scale_length
    return wav_ci_ranges


def credible_region_threshold(logpis, alpha=0.05):
    """
    Calculates the log posterior threshold for a sample to be part of the credible set at a given confidence level.

    :param logpis: vector of log posterior probabilities of MCMC samples
    :param float `alpha`: confidence level
    :return: log posterior threshold
    """
    return np.quantile(logpis, 1 - alpha)


def in_credible_region(logpi, threshold):
    """:meta private:"""
    return True if logpi <= threshold else False
