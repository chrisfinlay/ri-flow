#!/usr/bin/env python3
"""
nufft_gif.py - Create GIF directly from a Measurement Set using jax-finufft.

Replaces the two-step: `extract` (WSClean) + `mk-gif` workflow.

Usage:
    python nufft_gif.py -c img_starlink_1chan.yaml -d tab_rfi -tsx sgp4_bstar_var
"""

import time as _time

_t_start = _time.perf_counter()

import io
import os
import argparse
import warnings
import numpy as np

import jax.numpy as jnp
from jax_finufft import nufft1

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from PIL import Image

from astropy.wcs import WCS, FITSFixedWarning
from astropy.coordinates import SkyCoord
from astropy.wcs.utils import proj_plane_pixel_scales

from daskms import xds_from_ms, xds_from_table

from riflow import load_config
from riflow.config import prepend_suffix
from riflow.coords import get_tles, sat_radec, get_fornax_radec
from riflow.io.ms import read_ants_itrf
from riflow.extraction.light_curves import get_region_stats, plot_light_curves

warnings.filterwarnings("ignore", category=FITSFixedWarning)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def _parse_weight(weight_str: str) -> tuple[str, float | None]:
    """Parse a WSClean-style weight string into (scheme, robust).

    Examples: 'natural' → ('natural', None)
              'uniform'  → ('uniform', None)
              'briggs -0.5' → ('briggs', -0.5)
    """
    parts = weight_str.strip().lower().split()
    scheme = parts[0]
    if scheme == "briggs":
        return "briggs", float(parts[1]) if len(parts) > 1 else 0.0
    if scheme == "uniform":
        return "uniform", None
    return "natural", None


def _compute_weights(
    u_wl: np.ndarray,
    v_wl: np.ndarray,
    n_pix: int,
    pixel_rad: float,
    scheme: str,
    robust: float | None,
) -> np.ndarray:
    """Return per-baseline imaging weights (shape = len(u_wl)).

    The UV density grid uses the same cell size as the image pixel grid:
      uv_cell = 1 / (n_pix * pixel_rad)  [wavelengths]

    Schemes
    -------
    natural : all weights = 1
    uniform : w_k = 1 / n_k  (inverse cell occupancy)
    briggs  : w_k = 1 / (1 + n_k / f²)
              f² = (5 × 10^{-R})² × N / Σ n_i²
    """
    N = len(u_wl)
    if scheme == "natural":
        return np.ones(N, dtype=np.float32)

    # UV cell size matching the image grid
    uv_cell = 1.0 / (n_pix * pixel_rad)

    # Grid cell indices for both Hermitian halves
    u_idx   = np.clip(np.floor( u_wl / uv_cell + n_pix / 2).astype(int), 0, n_pix - 1)
    v_idx   = np.clip(np.floor( v_wl / uv_cell + n_pix / 2).astype(int), 0, n_pix - 1)
    u_idx_c = np.clip(np.floor(-u_wl / uv_cell + n_pix / 2).astype(int), 0, n_pix - 1)
    v_idx_c = np.clip(np.floor(-v_wl / uv_cell + n_pix / 2).astype(int), 0, n_pix - 1)

    density = np.zeros((n_pix, n_pix), dtype=np.float64)
    np.add.at(density, (v_idx,   u_idx),   1)
    np.add.at(density, (v_idx_c, u_idx_c), 1)
    n_k = density[v_idx, u_idx]

    if scheme == "uniform":
        w = 1.0 / np.maximum(n_k, 1.0)
    else:  # briggs
        sum_n2 = float(np.sum(density[density > 0] ** 2))
        f2 = (5.0 * 10.0 ** (-robust)) ** 2 * (2 * N) / sum_n2  # type: ignore[operator]
        w = 1.0 / (1.0 + f2 * n_k)

    return w.astype(np.float32)


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
    wcs: WCS, radec_all: np.ndarray, aperture_radius_deg: float, n_pix: int
) -> tuple[list[np.ndarray], np.ndarray]:
    """
    Pre-compute per-frame aperture pixel masks for all sources.

    Returns
    -------
    masks : list[ndarray]  shape (n_sources, n_times) of (n_pix_in_mask,) index arrays
    pix_xy : ndarray  shape (n_sources, n_times, 2)  — (x_centre, y_centre) per frame
    """
    pixel_scales = proj_plane_pixel_scales(wcs)[:2]
    aperture_radius_pix = aperture_radius_deg / np.mean(pixel_scales)

    n_sources, _, n_times = radec_all.shape
    yy, xx = np.mgrid[0:n_pix, 0:n_pix]  # (n_pix, n_pix)

    pix_xy = np.empty((n_sources, n_times, 2))
    masks: list[list[np.ndarray]] = [[] for _ in range(n_sources)]

    for s in range(n_sources):
        for t in range(n_times):
            coord = SkyCoord(ra=radec_all[s, 0, t], dec=radec_all[s, 1, t], unit="deg")
            x_c, y_c = wcs.world_to_pixel(coord)
            pix_xy[s, t] = [x_c, y_c]
            dist2 = (xx - x_c) ** 2 + (yy - y_c) ** 2
            masks[s].append(np.where(dist2 <= aperture_radius_pix ** 2))

    return masks, pix_xy, aperture_radius_pix


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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Create GIF from MS using jax-finufft (replaces extract + mk-gif)."
    )
    parser.add_argument("-c", "--config_path", required=True, help="Config YAML path.")
    parser.add_argument("-d", "--data_types", default="tab_rfi",
                        help="Comma-separated data type keys from config.")
    parser.add_argument("-ms", "--ms_path", default=None, help="Override MS path.")
    parser.add_argument("-i", "--image_path", default=None,
                        help="Override output image directory.")
    parser.add_argument("-tsx", "--tab_suffix", default=None,
                        help="TABASCAL solution suffix (e.g. sgp4_bstar_var).")
    parser.add_argument("-img", "--img_suffix", default=None,
                        help="Image parameter suffix (e.g. 'briggs05', 'uniform').")
    parser.add_argument("-g", "--gif_suffix", default=None, help="Extra GIF suffix.")
    parser.add_argument("-n", "--norad_ids", default=None,
                        help="Extra NORAD IDs (comma-separated).")
    parser.add_argument("-r", "--radius_deg", type=float, default=None,
                        help="Aperture circle radius in degrees.")
    parser.add_argument("-st", "--spacetrack_path", default=None,
                        help="Space-Track login YAML path.")
    args = parser.parse_args()

    config = load_config(args.config_path)

    # --- Resolve paths ---
    ms_path = os.path.abspath(args.ms_path or config["data"]["ms_path"])
    image_path = os.path.abspath(args.image_path or config["data"]["image_path"])
    freq_chan = int(config["data"].get("freq_chan", 0))

    spacetrack_path = args.spacetrack_path or config["gif"].get("spacetrack_path") or ""
    if spacetrack_path:
        spacetrack_path = os.path.abspath(spacetrack_path)

    norad_ids: list[int] = list(config["gif"].get("norad_ids", []))
    if args.norad_ids:
        norad_ids += [int(x) for x in args.norad_ids.split(",")]
    norad_ids = list(dict.fromkeys(norad_ids))  # deduplicate, preserve order

    aperture_radius_deg: float = args.radius_deg or float(
        config["gif"].get("marker_radius_deg", 3.0)
    )

    # --- Image parameters from config ---
    img_params = config["image"]["params"]
    n_pix = int(str(img_params["size"]).split()[0])
    pixel_deg = _parse_pixel_deg(str(img_params["scale"]))
    n_times = int(img_params["intervals-out"])
    pixel_rad = np.deg2rad(pixel_deg)

    weight_str = str(img_params.get("weight", "natural"))
    weight_scheme, briggs_robust = _parse_weight(weight_str)

    tab_suffix = prepend_suffix(args.tab_suffix)
    img_suffix = prepend_suffix(args.img_suffix)
    gif_suffix = prepend_suffix(args.gif_suffix)

    data_types = args.data_types.split(",")
    names = {
        key: config[key]["data_col"]
        for key in config.keys()
        if key not in ["data", "image", "extract", "gif"]
    }

    save_path = os.path.join(image_path, "gifs")
    os.makedirs(save_path, exist_ok=True)

    print(f"MS path   : {ms_path}")
    print(f"Image path: {image_path}")
    print(f"n_pix={n_pix}, pixel_deg={pixel_deg}°, n_times={n_times}")
    print(f"Weighting : {weight_str} (scheme={weight_scheme}, robust={briggs_robust})")

    # -----------------------------------------------------------------------
    # 1. Read MS (once, shared across all data types)
    # -----------------------------------------------------------------------
    t0 = _time.perf_counter()
    print("\n[1/4] Reading Measurement Set...")

    xds = xds_from_ms(ms_path)[0]
    xds_spw = xds_from_table(ms_path + "::SPECTRAL_WINDOW")[0]
    xds_field = xds_from_table(ms_path + "::FIELD")[0]

    uvw_all = xds.UVW.data.compute()                         # (N, 3)
    times_all = xds.TIME.data.compute()                      # (N,)
    flags_all = xds.FLAG.data.compute()[:, freq_chan, 0]     # (N,)
    freq_hz = float(xds_spw.CHAN_FREQ.data.compute()[0, freq_chan])
    phase_dir_rad = xds_field.PHASE_DIR.data.compute()[0, 0]  # (2,)

    phase_ra_deg = float(np.rad2deg(phase_dir_rad[0]))
    phase_dec_deg = float(np.rad2deg(phase_dir_rad[1]))

    lam = 299792458.0 / freq_hz                              # wavelength in metres

    # Group row indices by time step (maintains time order)
    unique_times = np.unique(times_all)
    assert len(unique_times) == n_times, (
        f"MS has {len(unique_times)} unique times but config expects {n_times}."
    )
    times_mjd = unique_times / (24 * 3600)

    # Sort rows by time to allow reshape if uniform row counts
    sort_idx = np.argsort(times_all, kind="stable")
    _, counts = np.unique(times_all, return_counts=True)
    uniform_rows = bool(np.all(counts == counts[0]))
    rows_per_time = int(counts[0])

    if uniform_rows:
        # Reshape into (n_times, rows_per_time, ...) for fast slicing
        uvw_t = (uvw_all[sort_idx] / lam).reshape(n_times, rows_per_time, 3)
        flags_t = flags_all[sort_idx].reshape(n_times, rows_per_time)
        time_idx = None  # use reshape slices instead
    else:
        uvw_t = uvw_all / lam
        flags_t = flags_all
        time_idx = [np.where(times_all == t)[0] for t in unique_times]

    t1 = _time.perf_counter()
    print(f"  Done in {t1 - t0:.2f}s | freq={freq_hz/1e6:.3f} MHz "
          f"| phase=({phase_ra_deg:.4f}°, {phase_dec_deg:.4f}°)")

    # -----------------------------------------------------------------------
    # 2. Get satellite positions (TLE propagation via Skyfield)
    # -----------------------------------------------------------------------
    print("\n[2/4] Fetching TLEs and propagating satellite positions...")
    t0 = _time.perf_counter()

    from tabsim.jax.coordinates import mjd_to_jd  # noqa: F401 (needed for get_tles)

    ants_itrf = read_ants_itrf(ms_path)
    obs_xyz = np.mean(ants_itrf, axis=0)

    radec_sats = None
    sat_labels: list[str] = []

    if norad_ids and spacetrack_path:
        tles, valid_norad_ids = get_tles(
            spacetrack_path, norad_ids, float(np.mean(times_mjd))
        )
        radec_sats = np.stack(
            [sat_radec(tle, times_mjd, obs_xyz) for tle in tles], axis=0
        )  # (n_sats, 2, n_times)
        sat_labels = [str(nid) for nid in valid_norad_ids]
        print(f"  Satellites: {sat_labels}")
    else:
        print("  No NORAD IDs or Space-Track path — skipping satellite circles.")

    fornax_radec = get_fornax_radec(n_times)  # (1, 2, n_times)

    if radec_sats is not None:
        radec_all = np.concatenate([radec_sats, fornax_radec], axis=0)
        titles = sat_labels + ["Fornax A"]
    else:
        radec_all = fornax_radec
        titles = ["Fornax A"]

    n_sources = len(titles)
    t1 = _time.perf_counter()
    print(f"  Done in {t1 - t0:.2f}s")

    # -----------------------------------------------------------------------
    # 3. Build WCS (shared for all frames & data types)
    # -----------------------------------------------------------------------
    wcs = _build_wcs(phase_ra_deg, phase_dec_deg, pixel_deg, n_pix)

    # -----------------------------------------------------------------------
    # 4. Process each data type
    # -----------------------------------------------------------------------
    for data_type in data_types:
        assert data_type in names, (
            f"'{data_type}' not found in config. Available: {list(names.keys())}"
        )
        data_col = names[data_type]
        gif_name = f"{data_type}{tab_suffix}{img_suffix}{gif_suffix}"
        print(f"\n[3/4] Imaging data column '{data_col}' -> '{gif_name}'")

        # Read visibilities for this column
        t0 = _time.perf_counter()
        vis_raw = xds[data_col].data.compute()[:, freq_chan, 0]  # (N,) complex
        if uniform_rows:
            vis_t_arr = vis_raw[sort_idx].reshape(n_times, rows_per_time)
        t1 = _time.perf_counter()
        print(f"  Visibility read: {t1 - t0:.2f}s")

        # --- Compute dirty images for all time steps ---
        t0 = _time.perf_counter()
        images = []
        for t_idx in range(n_times):
            if uniform_rows:
                u_t = uvw_t[t_idx, :, 0]
                v_t = uvw_t[t_idx, :, 1]
                vis_t = vis_t_arr[t_idx]
                flag_t = flags_t[t_idx]
            else:
                row_idx = time_idx[t_idx]  # type: ignore[index]
                u_t = uvw_t[row_idx, 0]
                v_t = uvw_t[row_idx, 1]
                vis_t = vis_raw[row_idx]
                flag_t = flags_t[row_idx]

            valid = ~flag_t.astype(bool)
            if valid.sum() == 0:
                images.append(np.zeros((n_pix, n_pix), dtype=np.float32))
                continue

            u_v, v_v, vis_v = u_t[valid], v_t[valid], vis_t[valid]
            w = _compute_weights(u_v, v_v, n_pix, pixel_rad, weight_scheme, briggs_robust)
            im = _dirty_image(u_v, v_v, vis_v, n_pix, pixel_rad, w)
            images.append(im)

            if t_idx == 0:
                t_jit = _time.perf_counter()
                print(f"  JIT compile (1st image): {t_jit - t0:.2f}s")

        t1 = _time.perf_counter()
        print(f"  All {n_times} dirty images: {t1 - t0:.2f}s")

        # --- Render GIF ---
        print(f"\n[4/4] Rendering {n_times} frames...")
        t0 = _time.perf_counter()

        data_arr = np.stack(images)
        vmin, vmax = np.percentile(data_arr, [0.5, 99.5])
        light_curves = np.empty((n_sources, n_times, 3))
        frames = []
        times_sec = (times_mjd - times_mjd[0]) * 86400.0

        # Pre-compute aperture masks and pixel centres (avoids repeated WCS calls)
        t_pre = _time.perf_counter()
        ap_masks, pix_xy, ap_radius_pix = _aperture_pixel_coords(
            wcs, radec_all, aperture_radius_deg, n_pix
        )
        print(f"  Aperture pre-compute: {_time.perf_counter() - t_pre:.2f}s")

        # Build figure once, update data each frame (saves ~50% of matplotlib overhead)
        fig, ax, im_obj = _make_frame_setup(wcs, vmin, vmax)
        buf = io.BytesIO()

        for f_idx, im_data in enumerate(images):
            # Update image data in-place
            im_obj.set_data(im_data)
            im_obj.set_extent([-0.5, n_pix - 0.5, -0.5, n_pix - 0.5])

            # Remove previous circles/labels (all artists after fixed ones)
            while len(ax.patches) > 0:
                ax.patches[-1].remove()
            while len(ax.texts) > 0:
                ax.texts[-1].remove()

            for s_idx in range(n_sources):
                mask = ap_masks[s_idx][f_idx]
                x_c, y_c = pix_xy[s_idx, f_idx]

                # Aperture stats (fast: pre-indexed)
                if len(mask[0]) > 0:
                    region = im_data[mask]
                    light_curves[s_idx, f_idx] = [region.min(), region.mean(), region.max()]
                else:
                    light_curves[s_idx, f_idx] = np.nan

                circ = Circle(
                    (x_c, y_c), ap_radius_pix,
                    edgecolor="red", facecolor="none", linewidth=1.5,
                )
                ax.add_patch(circ)
                ax.text(
                    x=x_c + 1.0 * ap_radius_pix,
                    y=y_c + 1.0 * ap_radius_pix,
                    s=titles[s_idx],
                    color="red", fontsize=9, ha="left", va="bottom",
                )

            # Save to in-memory buffer (avoids file I/O)
            buf.seek(0)
            buf.truncate()
            fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
            buf.seek(0)
            frames.append(Image.open(buf).copy().convert("P"))

        plt.close(fig)

        gif_path = os.path.join(save_path, f"{gif_name}.gif")
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=20,
            loop=0,
        )
        print(f"GIF saved as {gif_path}")

        np.save(os.path.join(save_path, f"{gif_name}_light_curves.npy"), light_curves)
        plot_light_curves(
            times_sec,
            light_curves,
            titles,
            os.path.join(save_path, f"{gif_name}.png"),
        )

        t1 = _time.perf_counter()
        print(f"  Frame render + GIF: {t1 - t0:.2f}s")

    t_total = _time.perf_counter() - _t_start
    print(f"\nTotal wall time: {t_total:.2f}s")


if __name__ == "__main__":
    main()
