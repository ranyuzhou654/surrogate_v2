"""Time-shift (circular shift) surrogate."""

import numpy as np


def timeshift_surrogate(x, rng=None):
    """Generate a time-shift surrogate by circular shifting.

    Shifts the series by a random amount in [T/4, 3T/4] to preserve
    local temporal structure while breaking coupling relationships.

    Parameters
    ----------
    x : ndarray, shape (T,)
        Input time series.
    rng : np.random.Generator, optional
        Random number generator.

    Returns
    -------
    surr : ndarray, shape (T,)
        Time-shifted surrogate.
    """
    if rng is None:
        rng = np.random.default_rng()

    T = len(x)
    shift = rng.integers(T // 4, 3 * T // 4 + 1)
    return np.roll(x, shift)
