import numpy as np


def credible_interval_range(predictions, alpha=0.05):
    quantiles = np.quantile(predictions, (alpha / 2, 1 - alpha / 2), axis=0)
    return np.diff(quantiles, axis=0)[0]


def credible_region_threshold(logpis, alpha=0.05):
    return np.quantile(logpis, 1 - alpha)


def in_credible_region(logpi, threshold):
    return True if logpi <= threshold else False
