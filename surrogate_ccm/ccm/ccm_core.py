"""Core CCM algorithm implementation.

Convention: ccm(x, y, ...) cross-maps from M_x to predict y, testing "y causes x".
"""

import numpy as np
from scipy.spatial import KDTree

from .embedding import delay_embed


def ccm(x, y, E, tau, L=None):
    """Convergent Cross Mapping from M_x to predict y.

    Tests the hypothesis "y causes x" by checking if the attractor
    reconstructed from x contains information about y.

    Parameters
    ----------
    x : ndarray, shape (T,)
        Target series (effect candidate).
    y : ndarray, shape (T,)
        Source series (cause candidate) to predict.
    E : int
        Embedding dimension.
    tau : int
        Time delay.
    L : int, optional
        Library size. If None, uses full length.

    Returns
    -------
    rho : float
        Pearson correlation between predicted and actual y.
    """
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()

    # Create shadow manifold from x
    M_x = delay_embed(x, E, tau)
    T_eff = len(M_x)

    # Align y with embedded x (use the last time index of each embedding vector)
    offset = (E - 1) * tau
    y_aligned = y[offset : offset + T_eff]

    if L is None:
        L = T_eff

    L = min(L, T_eff)

    # Use first L points as library
    lib = M_x[:L]
    y_lib = y_aligned[:L]

    # Build KDTree on library
    k = E + 1
    tree = KDTree(lib)
    dists, idxs = tree.query(lib, k=k + 1)  # +1 to exclude self

    # Remove self (first neighbor is self with dist=0)
    dists = dists[:, 1:]
    idxs = idxs[:, 1:]

    # Exponential weights
    eps = 1e-12
    w = np.exp(-dists / (dists[:, 0:1] + eps))
    w = w / (w.sum(axis=1, keepdims=True) + eps)

    # Predict y
    y_pred = np.sum(w * y_lib[idxs], axis=1)

    # Pearson correlation
    rho = np.corrcoef(y_lib, y_pred)[0, 1]
    if np.isnan(rho):
        rho = 0.0

    return rho


def ccm_convergence(x, y, E, tau, n_points=20):
    """Compute CCM correlation across library sizes (convergence check).

    Parameters
    ----------
    x, y : ndarray
        Time series.
    E : int
        Embedding dimension.
    tau : int
        Time delay.
    n_points : int
        Number of library sizes to test.

    Returns
    -------
    L_values : ndarray
        Library sizes tested.
    rho_values : ndarray
        CCM correlation at each library size.
    """
    T_eff = len(x) - (E - 1) * tau
    L_min = E + 2
    L_max = T_eff

    if L_min >= L_max:
        return np.array([L_max]), np.array([ccm(x, y, E, tau, L_max)])

    L_values = np.unique(np.linspace(L_min, L_max, n_points, dtype=int))
    rho_values = np.array([ccm(x, y, E, tau, L) for L in L_values])

    return L_values, rho_values
