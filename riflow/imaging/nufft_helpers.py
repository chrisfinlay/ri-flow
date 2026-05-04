"""Helper functions for the nufft-gif imaging pipeline."""

import numpy as np
import jax.numpy as jnp
from jax_finufft import nufft1

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.wcs.utils import proj_plane_pixel_scales


def _parse_pixel_deg(scale_str: str) -> float:
    """Convert WSClean scale string ('30amin', '0.5adeg', ...) to degrees."""
    s = scale_str.strip()
    if s.endswith("amin"):
        return float(s[:-4]) / 60.0
    if s.endswith("asec"):
        return float(s[:-4]) / 3600.0
    if s.endswith("adeg"):
        return float(s[:-4])
    return float(s)  # assume degrees


def _build_wcs(ra_deg: float, dec_deg: float, pixel_deg: float, n_pix: int) -> WCS:
    """Build a SIN-projection WCS matching WSClean's output convention."""
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [n_pix // 2 + 1, n_pix // 2 + 1]   # 1-indexed centre
    wcs.wcs.crval = [ra_deg, dec_deg]
    wcs.wcs.cdelt = [-pixel_deg, pixel_deg]               # RA decreases left→right
    wcs.wcs.ctype = ["RA---SIN", "DEC--SIN"]
    wcs.wcs.cunit = ["deg", "deg"]
    return wcs


def _parse_channels(chan_str: str, n_chan: int) -> np.ndarray:
    """Parse a numpy-style channel selection string into an index array.

    Supports:
      ":"          → all channels  (default)
      "0:16"       → channels 0–15
      "::2"        → every other channel
      "0:32:2"     → even channels 0–30
      "0,1,5,10"   → explicit list
      "3"          → single channel (backward-compat with old freq_chan)
    """
    idx = np.arange(n_chan)
    s = chan_str.strip()
    try:
        if ":" in s:
            parts = [int(p) if p.strip() else None for p in s.split(":")]
            return idx[slice(*parts)]
        if "," in s:
            return np.array([int(x.strip()) for x in s.split(",")], dtype=int)
        return np.array([int(s)], dtype=int)
    except Exception as exc:
        raise ValueError(f"Cannot parse channel selection '{chan_str}': {exc}") from exc


def _dirty_image(
    u_wl: np.ndarray,
    v_wl: np.ndarray,
    vis: np.ndarray,
    n_pix: int,
    pixel_rad: float,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute dirty image for one time step using NUFFT type-1.

    Input u/v are in wavelengths.  The image convention follows WSClean:
      - rows  → Dec  (m direction, v conjugate)
      - cols  → RA   (l direction, u conjugate, *decreasing* left→right)
    """
    if weights is None:
        weights = np.ones(len(u_wl), dtype=np.float32)

    # Apply weights, then add Hermitian conjugates
    wvis = (vis * weights).astype(np.complex64)
    u_all = np.concatenate([u_wl, -u_wl]).astype(np.float32)
    v_all = np.concatenate([v_wl, -v_wl]).astype(np.float32)
    vis_all = np.concatenate([wvis, wvis.conj()])

    # Scale to NUFFT coordinates in [-π, π].
    # Convention matching WSClean/FITS (CDELT1 < 0, east to left):
    #   rows (first dim,  x = -2π·pr·v) → Dec/m, north (+v) → higher row  ✓
    #   cols (second dim, y = +2π·pr·u) → RA/l,  west (-RA) → higher col  ✓
    u_nufft = jnp.array(2.0 * np.pi * pixel_rad * u_all)
    v_nufft = jnp.array(2.0 * np.pi * pixel_rad * (-v_all))
    vis_jax = jnp.array(vis_all)

    # nufft1: first coord → rows (Dec/-v), second coord → cols (RA/u)
    im = nufft1((n_pix, n_pix), vis_jax, v_nufft, u_nufft)
    # Normalise by total weight (×2 for conjugates) so a unit point source → peak ≈ 1
    return np.array(jnp.real(im)) / (2.0 * float(weights.sum()))


def _aperture_pixel_coords(
    wcs_or_list: WCS | list[WCS],
    radec_all: np.ndarray,
    aperture_radius_deg: float,
    n_pix: int,
) -> tuple:
    """
    Pre-compute per-frame aperture pixel masks for all sources.

    wcs_or_list may be a single WCS (shared across all frames) or a list of
    per-frame WCS objects (one per time step, for multi-scan MSes where each
    integration has its own phase centre).

    Returns
    -------
    masks : list[ndarray]  shape (n_sources, n_times) of (n_pix_in_mask,) index arrays
    pix_xy : ndarray  shape (n_sources, n_times, 2)  — (x_centre, y_centre) per frame
    aperture_radius_pix : float
    """
    wcs0 = wcs_or_list[0] if isinstance(wcs_or_list, list) else wcs_or_list
    pixel_scales = proj_plane_pixel_scales(wcs0)[:2]
    aperture_radius_pix = aperture_radius_deg / np.mean(pixel_scales)

    n_sources, _, n_times = radec_all.shape
    yy, xx = np.mgrid[0:n_pix, 0:n_pix]

    pix_xy = np.empty((n_sources, n_times, 2))
    masks: list[list[np.ndarray]] = [[] for _ in range(n_sources)]

    for s in range(n_sources):
        for t in range(n_times):
            wcs_t = wcs_or_list[t] if isinstance(wcs_or_list, list) else wcs_or_list
            coord = SkyCoord(ra=radec_all[s, 0, t], dec=radec_all[s, 1, t], unit="deg")
            x_c, y_c = wcs_t.world_to_pixel(coord)
            pix_xy[s, t] = [x_c, y_c]
            dist2 = (xx - x_c) ** 2 + (yy - y_c) ** 2
            masks[s].append(np.where(dist2 <= aperture_radius_pix ** 2))

    return masks, pix_xy, aperture_radius_pix


def _estimate_noise_level(
    images: list,
    ap_masks: list,
    n_pix: int,
    n_samples: int = 1000,
    horizon_fraction: float = 0.9,
) -> float:
    """Estimate image noise from random off-source background pixels.

    Samples pixels within a circular horizon (horizon_fraction * n_pix / 2 radius)
    that are not covered by any source aperture in any timestep, then returns
    the std of those pixel values across all frames.
    """
    # Union of all aperture pixels across all sources and timesteps
    excluded = np.zeros((n_pix, n_pix), dtype=bool)
    for s_masks in ap_masks:
        for mask in s_masks:
            excluded[mask] = True

    # Circular horizon mask centred on the image
    cy, cx = n_pix / 2.0, n_pix / 2.0
    radius  = horizon_fraction * n_pix / 2.0
    yy, xx  = np.mgrid[0:n_pix, 0:n_pix]
    in_horizon = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2

    valid_idx = np.argwhere(in_horizon & ~excluded)  # (N, 2)

    if len(valid_idx) == 0:
        return float(np.std(images))  # fallback: whole-image std

    rng = np.random.default_rng(seed=42)
    if len(valid_idx) > n_samples:
        chosen    = rng.choice(len(valid_idx), size=n_samples, replace=False)
        valid_idx = valid_idx[chosen]

    rows, cols = valid_idx[:, 0], valid_idx[:, 1]
    pixel_vals = np.stack([im[rows, cols] for im in images])  # (n_times, n_samples)
    return float(pixel_vals.std())


def _make_frame_setup(wcs: WCS, vmin: float, vmax: float):
    """Create reusable figure/axes with WCS projection."""
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection=wcs)
    im_obj = ax.imshow(
        np.zeros((1, 1)), cmap="gray", vmin=vmin, vmax=vmax, origin="lower"
    )
    ax.coords.grid(True, color="blue", ls="dotted")
    ax.coords[0].set_axislabel("Right Ascension")
    ax.coords[1].set_axislabel("Declination")
    cax = ax.inset_axes((0.87, 0.05, 0.03, 0.4))
    cb = ax.figure.colorbar(im_obj, cax=cax, orientation="vertical")
    cb.ax.tick_params(labelsize=6)
    return fig, ax, im_obj


def _source_pixel_xy(
    wcs: WCS,
    radec_all: np.ndarray,
    time_sel: np.ndarray,
) -> np.ndarray:
    """Return pixel positions (n_sources, n_sel, 2) for sources at selected times."""
    n_sources, _, _ = radec_all.shape
    n_sel = len(time_sel)
    pix = np.empty((n_sources, n_sel, 2))
    for s in range(n_sources):
        for i, t in enumerate(time_sel):
            coord = SkyCoord(ra=radec_all[s, 0, t], dec=radec_all[s, 1, t], unit="deg")
            pix[s, i] = wcs.world_to_pixel(coord)
    return pix


def _draw_source_overlays(
    ax,
    pix_xy: np.ndarray,
    ap_radius_pix: float,
    source_titles: list,
    n_pix: int,
    fontsize: int = 7,
) -> None:
    """Draw aperture circle for Fornax and track dots for satellites on ax.

    pix_xy : (n_sources, n_sel, 2)
    """
    sat_r = max(1.5, ap_radius_pix / 6.0)
    mid = pix_xy.shape[1] // 2

    for s, title in enumerate(source_titles):
        is_fornax  = "fornax"  in title.lower()
        is_shifted = "shifted" in title.lower()

        if is_shifted:
            continue

        color = "cyan" if is_fornax else "red"

        if is_fornax:
            x_c, y_c = float(pix_xy[s, mid, 0]), float(pix_xy[s, mid, 1])
            if 0 <= x_c < n_pix and 0 <= y_c < n_pix:
                ax.add_patch(Circle((x_c, y_c), ap_radius_pix,
                                    edgecolor=color, facecolor="none", linewidth=1.5))
                ax.text(x_c + ap_radius_pix * 1.05, y_c, title,
                        color=color, fontsize=fontsize, ha="left", va="center")
        else:
            for i in range(pix_xy.shape[1]):
                x_c, y_c = float(pix_xy[s, i, 0]), float(pix_xy[s, i, 1])
                if 0 <= x_c < n_pix and 0 <= y_c < n_pix:
                    ax.add_patch(Circle((x_c, y_c), sat_r,
                                        edgecolor=color, facecolor=color,
                                        alpha=0.6, linewidth=0.0))
            x_m, y_m = float(pix_xy[s, mid, 0]), float(pix_xy[s, mid, 1])
            if 0 <= x_m < n_pix and 0 <= y_m < n_pix:
                ax.text(x_m + sat_r * 1.5, y_m, title,
                        color=color, fontsize=fontsize, ha="left", va="center")


def _save_image_png(
    image: np.ndarray,
    wcs: WCS,
    save_path: str,
    title: str = "",
    pix_xy: np.ndarray | None = None,
    ap_radius_pix: float = 0.0,
    source_titles: list | None = None,
) -> None:
    """Save a single integrated dirty image as a PNG with WCS axes."""
    n_pix = image.shape[0]
    vmin, vmax = np.percentile(image, [0.5, 99.5])
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection=wcs)
    im_obj = ax.imshow(image, cmap="gray", vmin=vmin, vmax=vmax, origin="lower")
    ax.coords.grid(True, color="blue", ls="dotted")
    ax.coords[0].set_axislabel("Right Ascension")
    ax.coords[1].set_axislabel("Declination")
    cax = ax.inset_axes((0.87, 0.05, 0.03, 0.4))
    cb = ax.figure.colorbar(im_obj, cax=cax, orientation="vertical")
    cb.ax.tick_params(labelsize=6)
    if pix_xy is not None and source_titles is not None and ap_radius_pix > 0:
        _draw_source_overlays(ax, pix_xy, ap_radius_pix, source_titles, n_pix, fontsize=8)
    if title:
        ax.set_title(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _save_image_grid(
    images: list,
    wcs: WCS,
    chan_freqs_mhz: np.ndarray,
    save_path: str,
    suptitle: str = "",
    pix_xy: np.ndarray | None = None,
    ap_radius_pix: float = 0.0,
    source_titles: list | None = None,
) -> None:
    """Save a grid of integrated dirty images (one per channel) as a PNG."""
    n_chan = len(images)
    n_cols = int(np.ceil(np.sqrt(n_chan)))
    n_rows = int(np.ceil(n_chan / n_cols))
    n_pix  = images[0].shape[0]

    all_vals = np.concatenate([im.ravel() for im in images])
    vmin, vmax = np.percentile(all_vals, [0.5, 99.5])

    fig = plt.figure(figsize=(4 * n_cols, 4 * n_rows))
    for i, (im, freq) in enumerate(zip(images, chan_freqs_mhz)):
        ax = fig.add_subplot(n_rows, n_cols, i + 1, projection=wcs)
        ax.imshow(im, cmap="gray", vmin=vmin, vmax=vmax, origin="lower")
        ax.set_title(f"{freq:.2f} MHz", fontsize=8)
        ax.coords.grid(True, color="blue", ls="dotted", alpha=0.5)
        ax.coords[0].set_axislabel("")
        ax.coords[1].set_axislabel("")
        ax.tick_params(labelsize=6)
        if pix_xy is not None and source_titles is not None and ap_radius_pix > 0:
            _draw_source_overlays(ax, pix_xy, ap_radius_pix, source_titles, n_pix, fontsize=6)

    for j in range(n_chan, n_rows * n_cols):
        fig.add_subplot(n_rows, n_cols, j + 1).set_visible(False)

    if suptitle:
        fig.suptitle(suptitle, fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
