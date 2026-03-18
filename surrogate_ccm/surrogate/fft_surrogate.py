"""FFT phase-randomization surrogate."""

import numpy as np


def fft_surrogate(x, rng=None):
    """Generate an FFT surrogate by randomizing Fourier phases.

    Preserves the power spectrum (autocorrelation structure) but
    destroys nonlinear dependencies.

    Parameters
    ----------
    x : ndarray, shape (T,)
        Input time series.
    rng : np.random.Generator, optional
        Random number generator.

    Returns
    -------
    surr : ndarray, shape (T,)
        Surrogate time series.
    """
    if rng is None:
        rng = np.random.default_rng()

    T = len(x)
    X_fft = np.fft.rfft(x)

    # Random phases (preserve conjugate symmetry via rfft)
    phases = rng.uniform(0, 2 * np.pi, size=len(X_fft))
    phases[0] = 0  # DC component: no phase shift
    if T % 2 == 0:
        phases[-1] = 0  # Nyquist component: no phase shift

    X_surr = X_fft * np.exp(1j * phases)
    surr = np.fft.irfft(X_surr, n=T)

    return surr
