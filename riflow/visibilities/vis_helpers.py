"""Helper functions for visibility-space diagnostics."""

import numpy as np
from scipy.stats import binned_statistic_2d

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def grid_visibilities(
    u_wl: np.ndarray,
    v_wl: np.ndarray,
    vis: np.ndarray,
    n_grid: int = 256,
    uv_max: float | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Grid complex visibilities onto a 2D UV plane.

    Both (u,v) and conjugate (-u,-v) points are included for symmetric coverage.
    Empty cells are set to 0.

    Parameters
    ----------
    u_wl   : (N,) u coordinates in wavelengths
    v_wl   : (N,) v coordinates in wavelengths
    vis    : (N,) complex visibilities
    n_grid : grid resolution along each axis
    uv_max : UV extent to use; computed from data if None

    Returns
    -------
    amp_grid : (n_grid, n_grid) |mean(vis)| per cell, 0 where empty
    std_grid : (n_grid, n_grid) std(vis) per cell, 0 where empty
    uv_max   : UV extent used (same units as u_wl / v_wl)
    """
    if uv_max is None:
        uv_max = float(np.max(np.abs(np.concatenate([u_wl, v_wl])))) * 1.05

    u_all   = np.concatenate([u_wl,  -u_wl])
    v_all   = np.concatenate([v_wl,  -v_wl])
    vis_all = np.concatenate([vis,    vis.conj()])

    uv_range = [[-uv_max, uv_max], [-uv_max, uv_max]]

    amp_grid = np.nan_to_num(binned_statistic_2d(
        u_all, v_all, vis_all,
        statistic=lambda x: np.abs(np.nanmean(x)),
        bins=n_grid, range=uv_range,
    ).statistic).real.T

    std_grid = np.nan_to_num(binned_statistic_2d(
        u_all, v_all, vis_all,
        statistic=np.nanstd,
        bins=n_grid, range=uv_range,
    ).statistic).real.T

    return amp_grid, std_grid, uv_max


def save_uv_grid(
    grids: list,
    subtitles: list,
    uv_max: float,
    save_path: str,
    suptitle: str = "",
    cbar_label: str = "|mean(vis)| [Jy]",
) -> None:
    """Save a grid of UV-plane plots as a single PNG.

    Parameters
    ----------
    grids      : list of (n_grid, n_grid) arrays from grid_visibilities
    subtitles  : one label per subplot
    uv_max     : UV extent in wavelengths (shared across all subplots)
    save_path  : output PNG path
    suptitle   : optional figure-level title
    cbar_label : colorbar axis label
    """
    n      = len(grids)
    n_cols = int(np.ceil(np.sqrt(n)))
    n_rows = int(np.ceil(n / n_cols))

    finite_vals = [g[np.isfinite(g)] for g in grids if np.any(np.isfinite(g))]
    all_vals = np.concatenate(finite_vals) if finite_vals else np.array([0.0, 1.0])
    vmin = float(np.percentile(all_vals, 1))
    vmax = float(np.percentile(all_vals, 99))

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(3.5 * n_cols + 0.6, 3.5 * n_rows),
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    axes_flat = axes.ravel()
    extent    = [-uv_max, uv_max, -uv_max, uv_max]

    im = None
    for i, (grid, sub) in enumerate(zip(grids, subtitles)):
        ax = axes_flat[i]
        im = ax.imshow(
            grid, origin="lower", cmap="inferno",
            vmin=vmin, vmax=vmax, extent=extent, aspect="equal",
        )
        ax.set_title(sub, fontsize=8)
        ax.grid(True, alpha=0.2, color="white", linewidth=0.5)
        ax.tick_params(labelsize=6)

    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.text(0.5, 0.01, "u [λ]", ha="center", fontsize=10)
    fig.text(0.01, 0.5, "v [λ]", va="center", rotation="vertical", fontsize=10)

    if suptitle:
        fig.suptitle(suptitle, fontsize=11)

    # Leave room for a dedicated colorbar column on the right
    plt.tight_layout(rect=[0.03, 0.03, 0.92, 0.97])

    if im is not None:
        cbar_ax = fig.add_axes([0.93, 0.1, 0.015, 0.8])
        fig.colorbar(im, cax=cbar_ax, label=cbar_label)
        cbar_ax.tick_params(labelsize=7)

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_uv_pair(
    amp_grid: np.ndarray,
    std_grid: np.ndarray,
    uv_max: float,
    save_path: str,
    suptitle: str = "",
) -> None:
    """Save a side-by-side |mean(vis)| and std(vis) UV plot as a single PNG.

    Each panel has its own colorbar and color scale.
    """
    extent = [-uv_max, uv_max, -uv_max, uv_max]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharex=True, sharey=True)

    for ax, grid, label in zip(
        axes,
        [amp_grid, std_grid],
        ["|mean(vis)| [Jy]", "std(vis) [Jy]"],
    ):
        vals = grid[np.isfinite(grid)]
        vmin = float(np.percentile(vals, 1)) if vals.size else 0.0
        vmax = float(np.percentile(vals, 99)) if vals.size else 1.0
        im = ax.imshow(
            grid, origin="lower", cmap="inferno",
            vmin=vmin, vmax=vmax, extent=extent, aspect="equal",
        )
        ax.set_title(label, fontsize=10)
        ax.grid(True, alpha=0.2, color="white", linewidth=0.5)
        ax.tick_params(labelsize=7)
        fig.colorbar(im, ax=ax, label=label, fraction=0.046, pad=0.04)

    fig.text(0.5, 0.01, "u [λ]", ha="center", fontsize=10)
    fig.text(0.01, 0.5, "v [λ]", va="center", rotation="vertical", fontsize=10)

    if suptitle:
        fig.suptitle(suptitle, fontsize=11)

    plt.tight_layout(rect=[0.03, 0.03, 1.0, 0.97])
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
