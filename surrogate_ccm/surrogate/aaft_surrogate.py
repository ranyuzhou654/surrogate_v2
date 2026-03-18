"""Amplitude-Adjusted Fourier Transform (AAFT) surrogate."""

import numpy as np
from scipy.stats import rankdata

from .fft_surrogate import fft_surrogate


def aaft_surrogate(x, rng=None):
    """Generate an AAFT surrogate.

    1. Rank-order map x to Gaussian
    2. Create FFT surrogate of the Gaussian series
    3. Inverse rank-order map back to original amplitude distribution

    Preserves both the amplitude distribution and (approximately)
    the power spectrum.

    Parameters
    ----------
    x : ndarray, shape (T,)
        Input time series.
    rng : np.random.Generator, optional
        Random number generator.

    Returns
    -------
    surr : ndarray, shape (T,)
        AAFT surrogate time series.
    """
    if rng is None:
        rng = np.random.default_rng()

    T = len(x)

    # Step 1: Create rank-ordered Gaussian series
    ranks = rankdata(x)
    gaussian = np.sort(rng.normal(0, 1, T))
    x_gauss = gaussian[np.argsort(np.argsort(x))]

    # Step 2: FFT surrogate of the Gaussian-mapped series
    surr_gauss = fft_surrogate(x_gauss, rng=rng)

    # Step 3: Inverse rank-order mapping back to original values
    sorted_x = np.sort(x)
    surr = sorted_x[np.argsort(np.argsort(surr_gauss))]

    return surr
