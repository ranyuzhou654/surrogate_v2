"""CCM convergence curve plots."""

import matplotlib.pyplot as plt
import numpy as np


def plot_convergence(
    L_values,
    rho_values,
    surr_L_values=None,
    surr_rho_mean=None,
    surr_rho_std=None,
    title=None,
    xlabel="Library size L",
    ylabel="CCM ρ",
    ax=None,
    save_path=None,
):
    """Plot CCM convergence curve with optional surrogate confidence band.

    Parameters
    ----------
    L_values : ndarray
        Library sizes.
    rho_values : ndarray
        Observed CCM correlations.
    surr_L_values : ndarray, optional
        Library sizes for surrogate band.
    surr_rho_mean : ndarray, optional
        Mean surrogate correlation at each L.
    surr_rho_std : ndarray, optional
        Std of surrogate correlations.
    title : str, optional
    ax : matplotlib.axes.Axes, optional
    save_path : str, optional
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.figure

    ax.plot(L_values, rho_values, "b-o", markersize=3, label="Observed", linewidth=2)

    if surr_rho_mean is not None and surr_rho_std is not None:
        L_s = surr_L_values if surr_L_values is not None else L_values
        ax.fill_between(
            L_s,
            surr_rho_mean - 2 * surr_rho_std,
            surr_rho_mean + 2 * surr_rho_std,
            alpha=0.3,
            color="gray",
            label="Surrogate 95% CI",
        )
        ax.plot(L_s, surr_rho_mean, "k--", linewidth=1, label="Surrogate mean")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    return ax
