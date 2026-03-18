"""Statistical hypothesis testing for surrogate-based inference."""

import numpy as np
from statsmodels.stats.multitest import multipletests


def compute_pvalue(rho_obs, rho_surr):
    """Compute rank-based p-value.

    p = (#{surrogates >= observed} + 1) / (N_surrogates + 1)

    Parameters
    ----------
    rho_obs : float
        Observed CCM correlation.
    rho_surr : ndarray, shape (N_s,)
        Surrogate CCM correlations.

    Returns
    -------
    pvalue : float
    """
    rho_surr = np.asarray(rho_surr)
    n_greater = np.sum(rho_surr >= rho_obs)
    return (n_greater + 1) / (len(rho_surr) + 1)


def compute_zscore(rho_obs, rho_surr):
    """Compute z-score of observed value against surrogate distribution.

    Parameters
    ----------
    rho_obs : float
        Observed CCM correlation.
    rho_surr : ndarray, shape (N_s,)
        Surrogate CCM correlations.

    Returns
    -------
    zscore : float
    """
    rho_surr = np.asarray(rho_surr)
    mu = np.mean(rho_surr)
    sigma = np.std(rho_surr)
    if sigma < 1e-12:
        return 0.0
    return (rho_obs - mu) / sigma


def fdr_correction(pvalues, alpha=0.05, method="fdr_bh"):
    """Apply FDR correction to a set of p-values.

    Parameters
    ----------
    pvalues : ndarray
        Array of p-values (can be any shape; flattened internally).
    alpha : float
        Significance level.
    method : str
        Correction method (default 'fdr_bh' for Benjamini-Hochberg).

    Returns
    -------
    rejected : ndarray (bool)
        Same shape as pvalues; True where null is rejected.
    pvalues_corrected : ndarray
        Corrected p-values, same shape.
    """
    shape = pvalues.shape
    pvals_flat = pvalues.ravel()

    # Handle NaNs by setting them to 1
    mask_valid = ~np.isnan(pvals_flat)
    pvals_clean = np.where(mask_valid, pvals_flat, 1.0)

    rejected_flat, corrected_flat, _, _ = multipletests(
        pvals_clean, alpha=alpha, method=method
    )

    # NaN positions are not rejected
    rejected_flat = rejected_flat & mask_valid

    return rejected_flat.reshape(shape), corrected_flat.reshape(shape)
