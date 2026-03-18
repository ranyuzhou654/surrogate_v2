"""Causal matrix heatmap plots."""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_causal_heatmap(
    matrix,
    title="Causal Matrix",
    cmap="RdBu_r",
    vmin=None,
    vmax=None,
    annot=True,
    fmt=".2f",
    ax=None,
    save_path=None,
):
    """Plot a heatmap of a causal matrix.

    Parameters
    ----------
    matrix : ndarray, shape (N, N)
        Matrix to visualize (CCM rho, p-values, z-scores, or binary).
    title : str
    cmap : str
        Colormap.
    vmin, vmax : float, optional
        Color range.
    annot : bool
        Annotate cells with values.
    fmt : str
        Format for annotations.
    ax : matplotlib.axes.Axes, optional
    save_path : str, optional
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 7))
    else:
        fig = ax.figure

    N = matrix.shape[0]
    labels = [f"X{i}" for i in range(N)]

    # Mask diagonal
    mask = np.eye(N, dtype=bool)

    sns.heatmap(
        matrix,
        mask=mask,
        annot=annot if N <= 15 else False,
        fmt=fmt if annot and N <= 15 else "",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        square=True,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title(title)
    ax.set_xlabel("Source (j)")
    ax.set_ylabel("Target (i)")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    return ax


def plot_comparison_heatmaps(
    ccm_matrix, pvalue_matrix, detected, ground_truth, save_path=None
):
    """Plot side-by-side heatmaps: CCM rho, p-values, detected, ground truth.

    Parameters
    ----------
    ccm_matrix : ndarray, shape (N, N)
    pvalue_matrix : ndarray, shape (N, N)
    detected : ndarray, shape (N, N)
    ground_truth : ndarray, shape (N, N)
    save_path : str, optional
    """
    fig, axes = plt.subplots(1, 4, figsize=(28, 6))

    plot_causal_heatmap(ccm_matrix, title="CCM ρ", cmap="RdBu_r", vmin=-1, vmax=1, ax=axes[0])
    plot_causal_heatmap(
        pvalue_matrix, title="p-values", cmap="RdYlGn", vmin=0, vmax=0.1, ax=axes[1]
    )
    plot_causal_heatmap(detected, title="Detected", cmap="Blues", vmin=0, vmax=1, fmt="d", ax=axes[2])
    plot_causal_heatmap(
        ground_truth, title="Ground Truth", cmap="Blues", vmin=0, vmax=1, fmt="d", ax=axes[3]
    )

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    return fig
