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


def grid_redundancy_chi2(
    u_wl: np.ndarray,
    v_wl: np.ndarray,
    vis: np.ndarray,
    sigma: float = 1.0,
    cell: float | None = None,
    n_grid: int = 256,
    uv_max: float | None = None,
    return_sums: bool = False,
):
    """Per-uv-cell redundancy reduced chi-squared.

    A sky image must predict a single visibility per uv-cell, so the best any
    image can do in a cell is the mean of the samples that fall there. The scatter
    of the samples *about that mean*, normalised by the thermal noise, is a reduced
    chi-squared that equals ~1 iff the cell is redundant and noise-limited (fully
    imageable and calibrated), and exceeds 1 when there is non-redundant
    per-baseline / per-antenna structure no image can fit (miscalibration, mutual
    coupling, differing element beams, ...).

    Samples are folded to the u>=0 half-plane (Hermitian: V(-u,-v)=V*(u,v)) so
    conjugate-redundant baselines pool together; the returned grids are mirrored
    back to the full plane for display. Set ``cell`` to the imaging uv-cell
    ``1/(2 sin(fov/2))`` so the chi-squared means "imageability at that field of
    view's resolution"; otherwise an explicit ``n_grid`` is used.

    IMPORTANT: this pools every sample passed in, so call it on ONE snapshot
    (single time + channel = identical sky) and combine across snapshots via
    ``return_sums=True`` + :func:`finalize_redundancy`. Pooling across time/channel
    lets earth-rotation / uv-gradient masquerade as non-redundancy.

    Parameters
    ----------
    u_wl, v_wl : (N,) uv coordinates in wavelengths (pre-flagged by the caller)
    vis        : (N,) complex visibilities
    sigma      : scalar thermal noise (per real/imag component); used for the
                 reduced chi-squared (ignored when ``return_sums`` -- apply it in
                 :func:`finalize_redundancy` instead)
    cell       : uv cell size [wavelengths]; if given, overrides ``n_grid``
    n_grid     : grid resolution per axis (used when ``cell`` is None)
    uv_max     : uv extent [wavelengths]; computed from the data if None

    Returns
    -------
    return_sums=False : (chi2_grid, mean_grid, count_grid, info) with grids
        (G,G) mirrored to the full plane; info has uv_max, n_grid, cell,
        chi2_global, imageable_frac.
    return_sums=True  : dict of raw per-cell sums (chi2_sum=sum|V-mean|^2, dof,
        cnt, betw_num, tot_num) + uv_max, n_grid, for accumulation.
    """
    vis = np.asarray(vis)
    u = np.asarray(u_wl, float)
    v = np.asarray(v_wl, float)
    good = np.isfinite(vis)
    u, v, vis = u[good], v[good], vis[good]
    if u.size == 0:
        G2 = n_grid * n_grid
        if return_sums:
            return dict(chi2_sum=np.zeros(G2), dof=np.zeros(G2), cnt=np.zeros(G2),
                        betw_num=0.0, tot_num=0.0, uv_max=0.0, n_grid=n_grid)
        g = np.full((n_grid, n_grid), np.nan)
        return g, np.zeros_like(g), np.zeros_like(g), dict(
            uv_max=0.0, n_grid=n_grid, cell=0.0, chi2_global=np.nan, imageable_frac=np.nan)

    if uv_max is None:
        uv_max = float(np.max(np.abs(np.concatenate([u, v])))) * 1.02
    if cell is not None:
        n_grid = max(1, int(np.ceil(2.0 * uv_max / cell)))
    G2 = n_grid * n_grid

    # Fold to the u>=0 half-plane (V(-u,-v)=V*(u,v)) so each physical sample
    # appears once and conjugate-redundant baselines pool together.
    flip = (u < 0) | ((u == 0) & (v < 0))
    u = np.where(flip, -u, u)
    v = np.where(flip, -v, v)
    vis = np.where(flip, np.conj(vis), vis)

    iu = np.floor((u + uv_max) / (2.0 * uv_max) * n_grid).astype(int)
    iv = np.floor((v + uv_max) / (2.0 * uv_max) * n_grid).astype(int)
    inb = (iu >= 0) & (iu < n_grid) & (iv >= 0) & (iv < n_grid)
    iu, iv, vis = iu[inb], iv[inb], vis[inb]
    idx = iu * n_grid + iv

    cnt = np.bincount(idx, minlength=G2)
    SVr = np.bincount(idx, weights=vis.real, minlength=G2)
    SVi = np.bincount(idx, weights=vis.imag, minlength=G2)
    mean = np.zeros(G2, dtype=complex)
    nz = cnt > 0
    mean[nz] = (SVr[nz] + 1j * SVi[nz]) / cnt[nz]

    # Raw within-cell scatter sum_i |V_i - mean|^2. Under sky+noise its
    # expectation is 2 sigma^2 (N-1) per cell (removing the complex mean costs one
    # complex dof). The noise normalisation (a single robust sigma) is applied in
    # finalize so it can be re-used / overridden and summed cleanly over snapshots.
    chi2_sum = np.bincount(idx, weights=np.abs(vis - mean[idx]) ** 2, minlength=G2)
    dof = 2.0 * np.maximum(cnt - 1, 0)
    betw_num = float((cnt * np.abs(mean) ** 2).sum())
    tot_num = float((np.abs(vis) ** 2).sum())

    if return_sums:
        return dict(chi2_sum=chi2_sum, dof=dof, cnt=cnt.astype(float),
                    betw_num=betw_num, tot_num=tot_num,
                    uv_max=float(uv_max), n_grid=int(n_grid))

    chi2_grid, cnt_grid, info = finalize_redundancy(
        chi2_sum, dof, cnt.astype(float), betw_num, tot_num, n_grid, float(sigma))
    info.update(uv_max=float(uv_max), cell=float(2.0 * uv_max / n_grid))
    mean_grid = _mirror_fill(np.abs(mean).reshape(n_grid, n_grid).T,
                             cnt.reshape(n_grid, n_grid).T == 0)
    return chi2_grid, mean_grid, cnt_grid, info


def _mirror_fill(grid2d, empty):
    """Fill empty (u<0 half-plane) cells from their point-reflected partner so the
    half-plane statistic displays as a symmetric full uv plane."""
    mirror = grid2d[::-1, ::-1]
    out = grid2d.copy()
    out[empty] = mirror[empty]
    return out


def finalize_redundancy(chi2_sum, dof, cnt, betw_num, tot_num, n_grid, sigma=1.0):
    """Turn accumulated per-snapshot sums (see ``grid_redundancy_chi2(..,
    return_sums=True)``) into a reduced-chi^2 map + headline scalars, applying a
    single robust noise ``sigma`` (per real/imag component).

    Reduced chi^2 per cell = sum|V-mean|^2 / (2 sigma^2 (N-1)); = 1 if the cell is
    redundant and noise-limited, > 1 for non-redundant excess. Returns
    ``(chi2_grid, count_grid, dict(chi2_global, imageable_frac))`` with grids
    mirrored to the full plane for ``origin='lower'`` display.
    """
    import numpy as np
    denom = dof * sigma ** 2
    chi2 = np.full(chi2_sum.shape, np.nan)
    m = denom > 0
    chi2[m] = chi2_sum[m] / denom[m]
    chi2_global = float(chi2_sum[m].sum() / denom[m].sum()) if m.any() else np.nan
    imageable = float(betw_num / tot_num) if tot_num > 0 else np.nan
    chi2_grid = _mirror_fill(chi2.reshape(n_grid, n_grid).T,
                             (dof.reshape(n_grid, n_grid).T == 0))
    cnt_grid = _mirror_fill(cnt.reshape(n_grid, n_grid).T,
                            (cnt.reshape(n_grid, n_grid).T == 0))
    return (chi2_grid, cnt_grid,
            dict(chi2_global=chi2_global, imageable_frac=imageable))


def save_uv_chi2_grid(
    grids: list,
    subtitles: list,
    uv_max: float,
    save_path: str,
    suptitle: str = "",
    vmax: float = 3.0,
    vcenter: float = 1.0,
    vmin: float = 0.0,
    cbar_label: str = "within-cell reduced χ²  (1 = redundant)",
) -> None:
    """Save UV-plane maps on a diverging scale centred on ``vcenter``.

    Defaults render the redundancy reduced-chi^2 (white at 1, red > 1
    non-redundant, blue < 1). Pass ``vcenter=0`` for the non-redundancy excess map
    (white at 0 = calibrated, red = excess). Mirrors :func:`save_uv_grid`'s layout.
    """
    from matplotlib.colors import TwoSlopeNorm

    n = len(grids)
    n_cols = int(np.ceil(np.sqrt(n)))
    n_rows = int(np.ceil(n / n_cols))
    norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
    extent = [-uv_max, uv_max, -uv_max, uv_max]

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(3.5 * n_cols + 0.6, 3.5 * n_rows),
                             squeeze=False, sharex=True, sharey=True)
    axes_flat = axes.ravel()
    im = None
    for i, (grid, sub) in enumerate(zip(grids, subtitles)):
        ax = axes_flat[i]
        im = ax.imshow(grid, origin="lower", cmap="RdBu_r", norm=norm,
                       extent=extent, aspect="equal")
        ax.set_title(sub, fontsize=8)
        ax.grid(True, alpha=0.2, color="grey", linewidth=0.5)
        ax.tick_params(labelsize=6)
    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.text(0.5, 0.01, "u [λ]", ha="center", fontsize=10)
    fig.text(0.01, 0.5, "v [λ]", va="center", rotation="vertical", fontsize=10)
    if suptitle:
        fig.suptitle(suptitle, fontsize=11)
    plt.tight_layout(rect=[0.03, 0.03, 0.92, 0.97])
    if im is not None:
        cbar_ax = fig.add_axes([0.93, 0.1, 0.015, 0.8])
        fig.colorbar(im, cax=cbar_ax, label=cbar_label)
        cbar_ax.tick_params(labelsize=7)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
