"""Iterative Amplitude-Adjusted Fourier Transform (iAAFT) surrogate."""

import numpy as np


def iaaft_surrogate(x, rng=None, max_iter=200, tol=1e-8):
    """Generate an iAAFT surrogate.

    Iteratively adjusts both the amplitude distribution and power spectrum
    to match the original series.

    Parameters
    ----------
    x : ndarray, shape (T,)
        Input time series.
    rng : np.random.Generator, optional
        Random number generator.
    max_iter : int
        Maximum number of iterations.
    tol : float
        Convergence tolerance on spectral difference.

    Returns
    -------
    surr : ndarray, shape (T,)
        iAAFT surrogate time series.
    """
    if rng is None:
        rng = np.random.default_rng()

    T = len(x)
    sorted_x = np.sort(x)
    target_amplitudes = np.abs(np.fft.rfft(x))

    # Initialize with a random shuffle
    surr = x.copy()
    rng.shuffle(surr)

    prev_spectrum = np.zeros_like(target_amplitudes)

    for _ in range(max_iter):
        # Step 1: Match power spectrum
        surr_fft = np.fft.rfft(surr)
        surr_phases = np.angle(surr_fft)
        surr_fft_adjusted = target_amplitudes * np.exp(1j * surr_phases)
        surr = np.fft.irfft(surr_fft_adjusted, n=T)

        # Step 2: Match amplitude distribution (rank-order mapping)
        rank_order = np.argsort(np.argsort(surr))
        surr = sorted_x[rank_order]

        # Check convergence
        current_spectrum = np.abs(np.fft.rfft(surr))
        diff = np.mean((current_spectrum - prev_spectrum) ** 2)
        if diff < tol:
            break
        prev_spectrum = current_spectrum

    return surr
