"""Coupled Logistic Map network generator."""

import numpy as np


class LogisticNetwork:
    """Coupled logistic map network.

    X_i(t+1) = (1 - eps) * f(X_i(t)) + eps * sum_j A[i,j] * f(X_j(t)) / k_in_i

    where f(x) = r * x * (1 - x) and A[i,j]=1 means j drives i.

    Parameters
    ----------
    adj : ndarray, shape (N, N)
        Adjacency matrix (A[i,j]=1 means j->i).
    coupling : float
        Coupling strength epsilon.
    r : float
        Logistic parameter (default 3.9).
    """

    def __init__(self, adj, coupling, r=3.9):
        self.adj = np.asarray(adj, dtype=float)
        self.coupling = coupling
        self.r = r
        self.N = adj.shape[0]

    def generate(self, T, transient=1000, seed=None, noise_std=0.0,
                 dyn_noise_std=0.0):
        """Generate coupled logistic map time series.

        Parameters
        ----------
        T : int
            Number of time steps to return (after transient).
        transient : int
            Steps to discard.
        seed : int, optional
            Random seed.
        noise_std : float
            Observation noise standard deviation.
        dyn_noise_std : float
            Dynamical noise standard deviation (added at each iteration).

        Returns
        -------
        data : ndarray, shape (T, N)
            Time series for each node.
        """
        rng = np.random.default_rng(seed)
        total = T + transient
        eps = self.coupling
        r = self.r

        # In-degree for normalization
        k_in = self.adj.sum(axis=1)  # shape (N,)
        k_in_safe = np.where(k_in > 0, k_in, 1.0)

        # Initialize
        X = rng.uniform(0.1, 0.9, size=self.N)
        data = np.empty((total, self.N))

        for t in range(total):
            data[t] = X
            fX = r * X * (1 - X)
            coupled = self.adj @ fX / k_in_safe
            # Nodes with no inputs get no coupling
            has_input = k_in > 0
            X_new = np.where(
                has_input,
                (1 - eps) * fX + eps * coupled,
                fX,
            )
            if dyn_noise_std > 0:
                X_new += rng.normal(0, dyn_noise_std, size=self.N)
            X = np.clip(X_new, 0.0, 1.0)

        data = data[transient:]

        if noise_std > 0:
            data = data + rng.normal(0, noise_std, size=data.shape)

        return data
