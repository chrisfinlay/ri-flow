"""Helper functions for the nufft-gif imaging pipeline."""

import numpy as np
import jax.numpy as jnp
from jax_finufft import nufft1

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon

from matplotlib import colors as mcolors

from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.wcs.utils import proj_plane_pixel_scales


def _build_norm(data: np.ndarray, scale: str, pct_lo: float, pct_hi: float):
    """Build a matplotlib Normalize for a dirty image.

    scale:
      'linear' — straight percentile clip (default).
      'sqrt'   — PowerNorm(gamma=0.5); compresses bright pixels so faint
                 streaks come up. Negatives are clipped to 0.
      'power'  — PowerNorm(gamma=0.25); harder version of sqrt.
      'log'    — LogNorm; vmin is the (pct_lo)th percentile of positive pixels.
      'asinh'  — AsinhNorm; like log near bright, linear near zero (keeps
                 signed/near-zero structure). Falls back to linear pre-mpl-3.5.
      'symlog' — SymLogNorm, symmetric about 0; best for signed images such as
                 the `meanmed` (mean − median) projection.

    pct_lo/pct_hi set the clip percentiles (e.g. 0.5/99.5).
    """
    finite = data[np.isfinite(data)]
    vmin, vmax = np.percentile(finite, [pct_lo, pct_hi])
    if vmax <= vmin:
        vmax = vmin + 1e-6

    if scale == "linear":
        return mcolors.Normalize(vmin=vmin, vmax=vmax)
    if scale == "sqrt":
        return mcolors.PowerNorm(gamma=0.5, vmin=max(vmin, 0.0), vmax=vmax)
    if scale == "power":
        return mcolors.PowerNorm(gamma=0.25, vmin=max(vmin, 0.0), vmax=vmax)
    if scale == "log":
        pos = finite[finite > 0]
        lo = np.percentile(pos, pct_lo) if pos.size else vmax * 1e-3
        if not (lo > 0):
            lo = (vmax if vmax > 0 else 1.0) * 1e-3
        return mcolors.LogNorm(vmin=lo, vmax=max(vmax, lo * 10))
    if scale == "asinh":
        lw = float(np.percentile(np.abs(finite), 50))
        if lw <= 0:
            lw = (vmax - vmin) * 0.05 or 1.0
        try:
            return mcolors.AsinhNorm(linear_width=lw, vmin=vmin, vmax=vmax)
        except AttributeError:  # matplotlib < 3.5
            return mcolors.Normalize(vmin=vmin, vmax=vmax)
    if scale == "symlog":
        amax = float(np.percentile(np.abs(finite), pct_hi))
        if amax <= 0:
            amax = 1e-6
        linthresh = float(np.percentile(np.abs(finite), 50))
        linthresh = max(linthresh, amax * 1e-3)
        return mcolors.SymLogNorm(linthresh=linthresh, vmin=-amax, vmax=amax)
    raise ValueError(f"Unknown cmap scale '{scale}'")


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


def _aperture_noise_stats(
    images: list,
    ap_masks: list,
    n_pix: int,
    aperture_radius_pix: float,
    extra_excluded: np.ndarray | None = None,
    n_controls: int = 200,
    horizon_fraction: float = 0.9,
    level: float = 0.95,
    seed: int = 42,
) -> dict:
    """Per-timestep image noise + control-aperture nulls for the light curves.

    For each timestep this measures the background per-pixel std ``sigma(t)``.
    It then lays control apertures (same radius as the source apertures) over
    the off-source background and, at every timestep, measures both aperture
    statistics that the light curves use -- ``max|flux|`` and ``std`` -- in each,
    normalised by ``sigma(t)``, and pools across all timesteps and positions.
    Because only ``sigma(t)`` varies in time while the *shape* of the
    aperture-statistic distributions (set by the number of beams per aperture)
    is stationary, pooling gives many samples even when only a few control
    apertures fit in one frame.

    The ``max|flux|`` null is an extreme-value statistic, so its expected value
    and central band come from a single Gumbel fit; the ``std`` null is roughly
    chi-distributed about ``sigma`` and uses empirical quantiles. Both bands are
    central ``level`` intervals and scale per-timestep as ``sigma(t) *
    (dimensionless statistic)``.

    Parameters
    ----------
    ap_masks       : per-source list of per-time ``np.where`` index tuples (the
                     source apertures, excluded from the background).
    extra_excluded : optional (n_pix, n_pix) bool mask of additional sky to
                     avoid (e.g. known astronomical point sources).
    n_controls     : max number of clean control apertures to use.
    level          : central band level (0.95 -> 2.5th..97.5th percentiles).

    Returns
    -------
    dict with sigma_t, exp_max_t / max_lo_t / max_hi_t (max|flux| null) and
    std_exp_t / std_lo_t / std_hi_t (std null), all (n_times,), the dimensionless
    gumbel_loc / gumbel_scale, the band level, and n_controls actually used.
    """
    images = np.asarray(images)  # (n_times, n_pix, n_pix)
    n_times = images.shape[0]

    # Union of all source apertures (+ any extra avoid regions) to exclude.
    excluded = np.zeros((n_pix, n_pix), dtype=bool)
    for s_masks in ap_masks:
        for mask in s_masks:
            excluded[mask] = True
    if extra_excluded is not None:
        excluded |= extra_excluded

    # Circular horizon mask centred on the image.
    cy = cx = n_pix / 2.0
    radius = horizon_fraction * n_pix / 2.0
    yy, xx = np.mgrid[0:n_pix, 0:n_pix]
    in_horizon = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2
    background = in_horizon & ~excluded

    # Per-timestep background std sigma(t).
    bg_rows, bg_cols = np.where(background)
    if len(bg_rows) == 0:
        sigma_t = images.reshape(n_times, -1).std(axis=1)
    else:
        sigma_t = images[:, bg_rows, bg_cols].std(axis=1)
    sigma_t = sigma_t.astype(float)

    # Control-aperture offset template (a disc of the source aperture radius).
    R = int(np.ceil(aperture_radius_pix))
    dy, dx = np.mgrid[-R:R + 1, -R:R + 1]
    in_disc = dx ** 2 + dy ** 2 <= aperture_radius_pix ** 2
    off_y, off_x = dy[in_disc], dx[in_disc]

    # Candidate centres on a grid spaced ~ one aperture radius, then shuffled,
    # keeping only those whose full disc is inside the horizon and clear of any
    # source / avoid region.
    rng = np.random.default_rng(seed)
    step = max(1, int(round(aperture_radius_pix)))
    cand_y, cand_x = np.mgrid[R:n_pix - R:step, R:n_pix - R:step]
    cand = np.stack([cand_y.ravel(), cand_x.ravel()], axis=1)
    rng.shuffle(cand)

    pooled_max: list[np.ndarray] = []
    pooled_std: list[np.ndarray] = []
    n_used = 0
    safe_sigma = np.where(sigma_t > 0, sigma_t, np.nan)
    for c_y, c_x in cand:
        ry, rx = c_y + off_y, c_x + off_x
        if excluded[ry, rx].any() or not in_horizon[ry, rx].all():
            continue
        region = images[:, ry, rx]  # (n_times, n_ap)
        pooled_max.append(np.abs(region).max(axis=1) / safe_sigma)
        pooled_std.append(region.std(axis=1) / safe_sigma)
        n_used += 1
        if n_used >= n_controls:
            break

    p_lo, p_hi = (1.0 - level) / 2.0, 1.0 - (1.0 - level) / 2.0

    max_s = np.concatenate(pooled_max) if pooled_max else np.array([], dtype=float)
    max_s = max_s[np.isfinite(max_s)]
    std_s = np.concatenate(pooled_std) if pooled_std else np.array([], dtype=float)
    std_s = std_s[np.isfinite(std_s)]

    # max|flux| null: extreme-value statistic -> single Gumbel fit.
    if max_s.size >= 10:
        from scipy.stats import gumbel_r

        loc, scale = gumbel_r.fit(max_s)
        max_exp = loc + np.euler_gamma * scale
        max_lo, max_hi = gumbel_r.ppf([p_lo, p_hi], loc=loc, scale=scale)
    elif max_s.size > 0:
        # Too few clean controls to fit a Gumbel — fall back to empirical.
        loc = scale = np.nan
        max_exp = float(np.mean(max_s))
        max_lo, max_hi = np.quantile(max_s, [p_lo, p_hi])
    else:
        # No clean control apertures at all: crude white-noise extreme-value
        # guess from the aperture pixel count (overestimates N_eff because
        # dirty-image pixels are correlated, so treat as a loose upper bound).
        loc = scale = np.nan
        max_exp = float(np.sqrt(2.0 * np.log(max(len(off_y), 2))))
        max_lo, max_hi = max_exp - 1.0, max_exp + 3.0

    # std null: roughly chi-distributed about sigma -> empirical quantiles.
    if std_s.size > 0:
        std_exp = float(np.median(std_s))
        std_lo, std_hi = np.quantile(std_s, [p_lo, p_hi])
    else:
        std_exp, std_lo, std_hi = 1.0, 0.8, 1.2

    return {
        "sigma_t": sigma_t,
        "exp_max_t": sigma_t * max_exp,
        "max_lo_t": sigma_t * max_lo,
        "max_hi_t": sigma_t * max_hi,
        "std_exp_t": sigma_t * std_exp,
        "std_lo_t": sigma_t * std_lo,
        "std_hi_t": sigma_t * std_hi,
        "gumbel_loc": float(loc),
        "gumbel_scale": float(scale),
        "level": float(level),
        "n_controls": n_used,
    }


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


def _track_outline(pts: np.ndarray, half_width: float) -> np.ndarray | None:
    """Curved-capsule polygon outlining a satellite track.

    The track points are already time-ordered, so they form the centreline
    directly: it is offset by ±half_width along the local normal and closed
    with a semicircular cap at each end, giving a rounded tube that follows the
    arc of the track.

    pts : (n_sel, 2) time-ordered pixel positions along the track.
    half_width : perpendicular half-width of the tube (pixels).

    Returns (N, 2) closed-polygon vertices, or None if the track is degenerate.
    """
    p = np.asarray(pts, dtype=float)
    p = p[np.all(np.isfinite(p), axis=1)]
    # Drop consecutive duplicates so tangents are well defined.
    if len(p) >= 2:
        keep = np.concatenate([[True], np.any(np.diff(p, axis=0) != 0, axis=1)])
        p = p[keep]
    if len(p) < 2:
        return None

    # Local unit tangent (centred differences) and left-hand normal.
    tang = np.gradient(p, axis=0)
    norm = np.linalg.norm(tang, axis=1, keepdims=True)
    tang = tang / np.where(norm > 0, norm, 1.0)
    nrm = np.stack([-tang[:, 1], tang[:, 0]], axis=1)

    left  = p + half_width * nrm
    right = p - half_width * nrm

    phi = np.linspace(0.0, np.pi, 24)
    # End cap at p[-1]: +normal → −normal, bulging along +tangent.
    cap_end = [p[-1] + half_width * (np.cos(a) * nrm[-1] + np.sin(a) * tang[-1])
               for a in phi]
    # Start cap at p[0]: −normal → +normal, bulging along −tangent.
    cap_start = [p[0] + half_width * (np.cos(a) * (-nrm[0]) + np.sin(a) * (-tang[0]))
                 for a in phi]

    return np.concatenate([left, np.array(cap_end), right[::-1], np.array(cap_start)])


def _draw_source_overlays(
    ax,
    pix_xy: np.ndarray,
    ap_radius_pix: float,
    source_titles: list,
    n_pix: int,
    fontsize: int = 7,
    track_style: str = "capsule",
) -> None:
    """Draw aperture circle for Fornax and a track marker for satellites on ax.

    pix_xy : (n_sources, n_sel, 2)
    track_style : how to mark satellite tracks —
        'capsule' — single curved tube following the arc of the whole track
                    (default; the streak stays visible inside the thin border);
        'circles' — open circle of the aperture radius at each timestep;
        'dots'    — filled dot at each timestep (the original behaviour).
    'capsule' needs at least two distinct track points; for a degenerate
    (e.g. single-timestep) track it falls back to 'dots'.
    """
    sat_r = max(1.5, ap_radius_pix / 6.0)
    mid = pix_xy.shape[1] // 2

    def _draw_dots(s, color):
        for i in range(pix_xy.shape[1]):
            x_c, y_c = float(pix_xy[s, i, 0]), float(pix_xy[s, i, 1])
            if 0 <= x_c < n_pix and 0 <= y_c < n_pix:
                ax.add_patch(Circle((x_c, y_c), sat_r,
                                    edgecolor=color, facecolor=color,
                                    alpha=0.6, linewidth=0.0))

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
            continue

        # --- satellite track ---
        verts = _track_outline(pix_xy[s], ap_radius_pix) if track_style == "capsule" else None
        if verts is not None:
            ax.add_patch(Polygon(verts, closed=True, edgecolor=color,
                                 facecolor="none", linewidth=1.0, alpha=0.9))
        elif track_style == "circles":
            for i in range(pix_xy.shape[1]):
                x_c, y_c = float(pix_xy[s, i, 0]), float(pix_xy[s, i, 1])
                if 0 <= x_c < n_pix and 0 <= y_c < n_pix:
                    ax.add_patch(Circle((x_c, y_c), ap_radius_pix,
                                        edgecolor=color, facecolor="none",
                                        linewidth=0.7, alpha=0.7))
        else:  # dots, or capsule fallback for a degenerate track
            _draw_dots(s, color)

        x_m, y_m = float(pix_xy[s, mid, 0]), float(pix_xy[s, mid, 1])
        if 0 <= x_m < n_pix and 0 <= y_m < n_pix:
            ax.text(x_m + ap_radius_pix * 1.1, y_m, title,
                    color=color, fontsize=fontsize, ha="left", va="center")


def _save_image_png(
    image: np.ndarray,
    wcs: WCS,
    save_path: str,
    title: str = "",
    pix_xy: np.ndarray | None = None,
    ap_radius_pix: float = 0.0,
    source_titles: list | None = None,
    scale: str = "linear",
    cmap: str = "gray",
    pct: tuple = (0.5, 99.5),
    track_style: str = "capsule",
) -> None:
    """Save a single integrated dirty image as a PNG with WCS axes."""
    n_pix = image.shape[0]
    norm = _build_norm(image, scale, pct[0], pct[1])
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection=wcs)
    im_obj = ax.imshow(image, cmap=cmap, norm=norm, origin="lower")
    ax.coords.grid(True, color="blue", ls="dotted")
    ax.coords[0].set_axislabel("Right Ascension")
    ax.coords[1].set_axislabel("Declination")
    cax = ax.inset_axes((0.87, 0.05, 0.03, 0.4))
    cb = ax.figure.colorbar(im_obj, cax=cax, orientation="vertical")
    cb.ax.tick_params(labelsize=6)
    if pix_xy is not None and source_titles is not None and ap_radius_pix > 0:
        _draw_source_overlays(ax, pix_xy, ap_radius_pix, source_titles, n_pix,
                              fontsize=8, track_style=track_style)
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
    scale: str = "linear",
    cmap: str = "gray",
    pct: tuple = (0.5, 99.5),
    track_style: str = "capsule",
) -> None:
    """Save a grid of integrated dirty images (one per channel) as a PNG."""
    n_chan = len(images)
    n_cols = int(np.ceil(np.sqrt(n_chan)))
    n_rows = int(np.ceil(n_chan / n_cols))
    n_pix  = images[0].shape[0]

    all_vals = np.concatenate([im.ravel() for im in images])
    norm = _build_norm(all_vals, scale, pct[0], pct[1])

    fig = plt.figure(figsize=(4 * n_cols, 4 * n_rows))
    for i, (im, freq) in enumerate(zip(images, chan_freqs_mhz)):
        ax = fig.add_subplot(n_rows, n_cols, i + 1, projection=wcs)
        ax.imshow(im, cmap=cmap, norm=norm, origin="lower")
        ax.set_title(f"{freq:.2f} MHz", fontsize=8)
        ax.coords.grid(True, color="blue", ls="dotted", alpha=0.5)
        ax.coords[0].set_axislabel("")
        ax.coords[1].set_axislabel("")
        ax.tick_params(labelsize=6)
        if pix_xy is not None and source_titles is not None and ap_radius_pix > 0:
            _draw_source_overlays(ax, pix_xy, ap_radius_pix, source_titles, n_pix,
                                  fontsize=6, track_style=track_style)

    for j in range(n_chan, n_rows * n_cols):
        fig.add_subplot(n_rows, n_cols, j + 1).set_visible(False)

    if suptitle:
        fig.suptitle(suptitle, fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
