"""Phase surrogate for testing phase-coupling hypotheses.

Decomposes the signal into instantaneous amplitude and phase via the
Hilbert transform, then shuffles phase increments while preserving
the amplitude envelope. Effective for phase-coupled oscillatory systems
(e.g., Kuramoto) where spectral surrogates preserve too much structure.

Reference
---------
Lancaster, G. et al. (2018). Surrogate data for hypothesis testing of
physical systems. Physics Reports, 748, 1-60.
"""

import numpy as np
from scipy.signal import hilbert


def _estimate_dominant_period(x):
    """Estimate the dominant oscillation period from the power spectrum.

    Returns
    -------
    period : int
        Dominant period in samples, or 0 if no clear periodicity.
    """
    x_c = x - np.mean(x)
    fft_vals = np.fft.rfft(x_c)
    power = np.abs(fft_vals) ** 2
    power[0] = 0  # exclude DC
    if power.sum() < 1e-15:
        return 0
    peak_idx = np.argmax(power)
    if peak_idx == 0:
        return 0
    return len(x) // peak_idx


def phase_surrogate(x, rng=None, block_size="auto"):
    """Generate a phase surrogate by shuffling phase increments.

    Algorithm:
    1. Compute analytic signal via Hilbert transform
    2. Extract instantaneous amplitude A(t) and unwrapped phase φ(t)
    3. Compute phase increments Δφ(t) = φ(t+1) - φ(t)
    4. Shuffle Δφ in blocks of size `block_size`
    5. Reconstruct phase from shuffled increments
    6. Surrogate = A(t) * cos(φ_shuffled(t))

    Parameters
    ----------
    x : ndarray, shape (T,)
        Input time series (should be approximately oscillatory).
    rng : numpy.random.Generator, optional
        Random number generator.
    block_size : int or "auto"
        Size of blocks for shuffling phase increments.
        "auto" (default): estimate dominant period and use period // 2,
            preserving within-cycle phase dynamics while destroying
            inter-cycle coupling. Falls back to 1 if no periodicity.
        1 = fully independent shuffling.
        Larger values preserve short-range phase dynamics.

    Returns
    -------
    surrogate : ndarray, shape (T,)
        Phase surrogate time series.
    """
    if rng is None:
        rng = np.random.default_rng()

    x = np.asarray(x, dtype=float).ravel()
    T = len(x)

    # Resolve auto block_size
    if block_size == "auto":
        period = _estimate_dominant_period(x)
        block_size = max(period // 2, 1) if period >= 4 else 1

    # Subtract mean for cleaner Hilbert transform
    x_mean = np.mean(x)
    x_centered = x - x_mean

    # Analytic signal via Hilbert transform
    analytic = hilbert(x_centered)
    amplitude = np.abs(analytic)
    phase = np.unwrap(np.angle(analytic))

    # Phase increments
    dphase = np.diff(phase)

    # Shuffle increments (optionally in blocks)
    if block_size <= 1:
        dphase_shuffled = rng.permutation(dphase)
    else:
        n_blocks = len(dphase) // block_size
        remainder = len(dphase) % block_size
        blocks = [dphase[i * block_size:(i + 1) * block_size]
                  for i in range(n_blocks)]
        if remainder > 0:
            blocks.append(dphase[n_blocks * block_size:])
        rng.shuffle(blocks)
        dphase_shuffled = np.concatenate(blocks)

    # Reconstruct phase
    phase_surr = np.empty(T)
    phase_surr[0] = phase[0] + rng.uniform(0, 2 * np.pi)  # random start phase
    phase_surr[1:] = phase_surr[0] + np.cumsum(dphase_shuffled)

    # Reconstruct signal
    surrogate = amplitude * np.cos(phase_surr) + x_mean

    return surrogate
