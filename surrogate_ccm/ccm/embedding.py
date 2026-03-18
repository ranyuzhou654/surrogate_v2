"""Delay embedding and automatic parameter selection."""

import numpy as np
from scipy.spatial import KDTree


def delay_embed(x, E, tau):
    """Create a time-delay embedding matrix.

    Parameters
    ----------
    x : ndarray, shape (T,)
        Input time series.
    E : int
        Embedding dimension.
    tau : int
        Time delay.

    Returns
    -------
    embedded : ndarray, shape (T_eff, E)
        Delay-embedded matrix where T_eff = T - (E-1)*tau.
    """
    x = np.asarray(x).ravel()
    T = len(x)
    T_eff = T - (E - 1) * tau
    if T_eff <= 0:
        raise ValueError(f"Time series too short (T={T}) for E={E}, tau={tau}")

    indices = np.arange(T_eff)[:, None] + np.arange(E)[None, :] * tau
    return x[indices]


def _mutual_information(x, tau, n_bins=64):
    """Compute auto mutual information at a given lag."""
    T = len(x)
    x1 = x[: T - tau]
    x2 = x[tau:]

    c_xy, xedges, yedges = np.histogram2d(x1, x2, bins=n_bins)
    c_xy = c_xy / c_xy.sum()
    c_x = c_xy.sum(axis=1)
    c_y = c_xy.sum(axis=0)

    mask = c_xy > 0
    mi = np.sum(
        c_xy[mask] * np.log(c_xy[mask] / (c_x[:, None] * c_y[None, :])[mask])
    )
    return mi


def _autocorrelation_tau(x, tau_max=50):
    """Find tau where autocorrelation first drops below 1/e.

    This is a robust fallback when MI has no local minimum (common for
    chaotic maps where MI decays monotonically).
    """
    x_centered = x - np.mean(x)
    var = np.var(x_centered)
    if var < 1e-12:
        return 1

    threshold = 1.0 / np.e
    T = len(x_centered)

    for t in range(1, min(tau_max + 1, T)):
        acf = np.mean(x_centered[: T - t] * x_centered[t:]) / var
        if acf < threshold:
            return max(t, 1)

    return 1


def select_tau(x, tau_max=50):
    """Select delay tau for embedding.

    Strategy:
    - Compute autocorrelation 1/e decay time (tau_acf).
    - If tau_acf <= 2 (fast decorrelation, typical for maps): return tau_acf.
    - Otherwise (slow decorrelation, typical for flows): search for MI
      first local minimum up to 2*tau_acf. If found, use it; else use tau_acf.

    Parameters
    ----------
    x : ndarray
        Input time series.
    tau_max : int
        Maximum lag to search.

    Returns
    -------
    tau : int
        Selected time delay (minimum 1).
    """
    x = np.asarray(x).ravel()

    # Step 1: autocorrelation 1/e decay time
    tau_acf = _autocorrelation_tau(x, tau_max)

    # For maps (fast decorrelation), autocorrelation is sufficient
    if tau_acf <= 2:
        return tau_acf

    # Step 2: for continuous systems, MI first minimum is more precise
    search_limit = min(tau_acf * 2, tau_max)
    mi_values = np.array(
        [_mutual_information(x, t) for t in range(1, search_limit + 1)]
    )

    for i in range(1, len(mi_values) - 1):
        if mi_values[i] < mi_values[i - 1] and mi_values[i] < mi_values[i + 1]:
            return i + 1

    return tau_acf


def _simplex_predict_rho(x, E, tau):
    """Compute simplex prediction accuracy with tp=tau prediction horizon.

    For maps (tau=1), this is one-step-ahead. For continuous flows (tau>1),
    predicting tau steps ahead is harder and forces proper attractor
    reconstruction, allowing meaningful E discrimination.
    """
    tp = tau  # prediction horizon = tau (standard in rEDM)
    emb = delay_embed(x, E, tau)
    T_eff = len(emb)

    # Target: x value tp steps after the last index of each embedding vector
    offset = (E - 1) * tau
    target_indices = np.arange(T_eff) + offset + tp
    valid = target_indices < len(x)
    emb = emb[valid]
    target_indices = target_indices[valid]
    T_eff = len(emb)

    if T_eff < 2 * (E + 2):
        return -1.0

    targets = x[target_indices]

    k = E + 1
    tree = KDTree(emb)
    dists, idxs = tree.query(emb, k=k + 1)
    dists = dists[:, 1:]  # remove self
    idxs = idxs[:, 1:]

    eps = 1e-12
    w = np.exp(-dists / (dists[:, 0:1] + eps))
    w = w / (w.sum(axis=1, keepdims=True) + eps)

    y_pred = np.sum(w * targets[idxs], axis=1)
    rho = np.corrcoef(targets, y_pred)[0, 1]
    return rho if not np.isnan(rho) else -1.0


def select_E(x, tau, E_max=10):
    """Select embedding dimension E using simplex projection.

    Tests E=2,...,E_max and picks the E that maximizes one-step-ahead
    prediction accuracy. This is the standard method used with CCM
    (Sugihara et al. 2012).

    Parameters
    ----------
    x : ndarray
        Input time series.
    tau : int
        Time delay.
    E_max : int
        Maximum dimension to test.

    Returns
    -------
    E : int
        Selected embedding dimension (minimum 2).
    """
    x = np.asarray(x).ravel()
    best_E = 2
    best_rho = -np.inf

    for E in range(2, E_max + 1):
        if len(x) - (E - 1) * tau < 2 * (E + 2):
            break
        rho = _simplex_predict_rho(x, E, tau)
        if rho > best_rho:
            best_rho = rho
            best_E = E

    return best_E


def select_parameters(x, tau_max=50, E_max=10):
    """Automatically select embedding parameters (E, tau).

    Parameters
    ----------
    x : ndarray
        Input time series.
    tau_max : int
        Maximum lag for MI / autocorrelation.
    E_max : int
        Maximum embedding dimension for simplex projection.

    Returns
    -------
    E : int
        Embedding dimension.
    tau : int
        Time delay.
    """
    tau = select_tau(x, tau_max)
    E = select_E(x, tau, E_max)
    return E, tau
