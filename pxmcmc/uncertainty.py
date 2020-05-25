import numpy as np


def pixel_ci_range(predictions, alpha=0.05):
    quantiles = np.quantile(predictions, (alpha / 2, 1 - alpha / 2), axis=0)
    return np.diff(quantiles, axis=0)[0]
