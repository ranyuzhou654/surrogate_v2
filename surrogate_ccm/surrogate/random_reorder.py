"""Random reorder (permutation) surrogate."""

import numpy as np


def random_reorder_surrogate(x, rng=None):
    """Generate a random reorder surrogate by permuting the time series.

    Destroys all temporal structure while preserving the amplitude distribution.

    Parameters
    ----------
    x : ndarray, shape (T,)
        Input time series.
    rng : np.random.Generator, optional
        Random number generator.

    Returns
    -------
    surr : ndarray, shape (T,)
        Randomly permuted surrogate.
    """
    if rng is None:
        rng = np.random.default_rng()

    surr = x.copy()
    rng.shuffle(surr)
    return surr
