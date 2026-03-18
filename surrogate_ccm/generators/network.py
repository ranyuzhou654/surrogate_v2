"""Network topology generation (ER, WS, Ring).

Convention: A[i,j] = 1 means j -> i (j drives i).
"""

import networkx as nx
import numpy as np


def generate_network(topology, N, seed=None, **kwargs):
    """Generate a directed adjacency matrix.

    Parameters
    ----------
    topology : str
        One of 'ER', 'WS', 'ring'.
    N : int
        Number of nodes.
    seed : int, optional
        Random seed.
    **kwargs
        Topology-specific parameters:
        - ER: p (edge probability, default 0.3)
        - WS: k (nearest neighbors, default 4), p (rewiring prob, default 0.3)
        - ring: k (nearest neighbors per side, default 1)

    Returns
    -------
    adj : ndarray, shape (N, N)
        Adjacency matrix where A[i,j]=1 means j->i.
    """
    topology = topology.upper()

    if topology == "ER":
        p = kwargs.get("p", kwargs.get("er_p", 0.3))
        G = nx.erdos_renyi_graph(N, p, directed=True, seed=seed)

    elif topology == "WS":
        k = kwargs.get("k", kwargs.get("ws_k", 4))
        p = kwargs.get("p", kwargs.get("ws_p", 0.3))
        G = nx.watts_strogatz_graph(N, k, p, seed=seed).to_directed()

    elif topology == "RING":
        k = kwargs.get("k", 1)
        G = nx.DiGraph()
        G.add_nodes_from(range(N))
        for i in range(N):
            for offset in range(1, k + 1):
                G.add_edge((i - offset) % N, i)
                G.add_edge((i + offset) % N, i)

    else:
        raise ValueError(f"Unknown topology: {topology}")

    # Remove self-loops
    G.remove_edges_from(nx.selfloop_edges(G))

    # nx.to_numpy_array returns adj[i,j]=1 meaning i->j (row=source).
    # Our convention is A[i,j]=1 meaning j->i (row=receiver), so transpose.
    adj = nx.to_numpy_array(G, dtype=int).T
    return adj
