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
from riflow.io.ms import read_ants_itrf, recopy_tab_results
from riflow.extraction.light_curves import get_region_stats, plot_light_curves, plot_spectrogram, plot_spectrum
from riflow.imaging.weights import parse_weight, compute_weights
from riflow.imaging.wstack import dirty_image_wstack, estimate_n_wplanes

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


# Weight helpers now live in riflow.imaging.weights; imported above.
# Local aliases for backward compat within this module.
_parse_weight   = parse_weight
_compute_weights = compute_weights


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
    wcs_or_list: "WCS | list[WCS]",
    radec_all: np.ndarray,
    aperture_radius_deg: float,
    n_pix: int,
) -> tuple[list[np.ndarray], np.ndarray]:
    """
    Pre-compute per-frame aperture pixel masks for all sources.

    wcs_or_list may be a single WCS (shared across all frames) or a list of
    per-frame WCS objects (one per time step, for multi-scan MSes where each
    integration has its own phase centre).

    Returns
    -------
    masks : list[ndarray]  shape (n_sources, n_times) of (n_pix_in_mask,) index arrays
    pix_xy : ndarray  shape (n_sources, n_times, 2)  — (x_centre, y_centre) per frame
    """
    wcs0 = wcs_or_list[0] if isinstance(wcs_or_list, list) else wcs_or_list
    pixel_scales = proj_plane_pixel_scales(wcs0)[:2]
    aperture_radius_pix = aperture_radius_deg / np.mean(pixel_scales)

    n_sources, _, n_times = radec_all.shape
    yy, xx = np.mgrid[0:n_pix, 0:n_pix]  # (n_pix, n_pix)

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
    parser.add_argument("-ws", "--wstack", action="store_true", default=False,
                        help="Enable w-stacking correction for wide-field imaging.")
    parser.add_argument("-nw", "--n_wplanes", type=int, default=None,
                        help="Number of w-stacking planes (default: auto-estimated from data).")
    parser.add_argument("-rc", "--recopy", action="store_true", default=False,
                        help="Re-copy TABASCAL results from the matching zarr into the MS "
                             "before imaging.  Requires -tsx.")
    parser.add_argument("-td", "--tab_data", default="map",
                        help="Zarr result type: 'map' (default) or 'init'.")
    parser.add_argument("-mn", "--model_name", default="Custom",
                        help="TABASCAL model name encoded in the zarr filename (default: 'Custom').")
    parser.add_argument("-ch", "--channels", default=None,
                        help="Channel selection using numpy indexing, e.g. ':', '0:16', "
                             "'::2', '0,1,5'.  Default: all channels.")
    parser.add_argument("-mo", "--mode", default="mfs",
                        choices=["mfs", "perchan"],
                        help="Imaging mode: 'mfs' — all selected channels combined into "
                             "one image per timestep (default); 'perchan' — one GIF per channel.")
    args = parser.parse_args()

    config = load_config(args.config_path)

    # --- Resolve paths ---
    ms_path    = os.path.abspath(args.ms_path    or config["data"]["ms_path"])
    image_path = os.path.abspath(args.image_path or config["data"]["image_path"])

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
    pixel_rad = np.deg2rad(pixel_deg)

    weight_str = str(img_params.get("weight", "natural"))
    weight_scheme, briggs_robust = _parse_weight(weight_str)

    # W-stacking: CLI flag takes priority, then config, default off
    use_wstack = args.wstack or bool(img_params.get("wstack", False))
    n_wplanes_cfg = args.n_wplanes or img_params.get("n_wplanes", None)
    if n_wplanes_cfg is not None:
        n_wplanes_cfg = int(n_wplanes_cfg)

    # Channel selection string — CLI > config > legacy freq_chan > all channels
    _cfg_freq_chan = config["data"].get("freq_chan")
    chan_sel_str = (
        args.channels
        or str(img_params.get("channels", ""))
        or (str(int(_cfg_freq_chan)) if _cfg_freq_chan is not None else None)
        or ":"
    )
    mode = args.mode or str(img_params.get("mode", "mfs"))

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
    print(f"n_pix={n_pix}, pixel_deg={pixel_deg}°")
    print(f"Weighting : {weight_str} (scheme={weight_scheme}, robust={briggs_robust})")
    print(f"W-stacking: {'ON' if use_wstack else 'OFF (flat-sky 2D)'}")

    # -----------------------------------------------------------------------
    # 0. Optionally re-copy TABASCAL results from zarr → MS columns
    # -----------------------------------------------------------------------
    if args.recopy:
        sim_dir = config["data"].get("sim_dir") or None
        print(f"\n[0/4] Re-copying TABASCAL results (tab_data='{args.tab_data}', "
              f"model='{args.model_name}', suffix='{tab_suffix}')...")
        recopy_tab_results(
            ms_path=ms_path,
            tab_suffix=tab_suffix,
            data_col=config["data"]["data_col"],
            sim_dir=sim_dir,
            tab_data=args.tab_data,
            model_name=args.model_name,
        )

    # -----------------------------------------------------------------------
    # 1. Read MS (once, shared across all data types)
    # -----------------------------------------------------------------------
    t0 = _time.perf_counter()
    print("\n[1/4] Reading Measurement Set...")

    xds_spw   = xds_from_table(ms_path + "::SPECTRAL_WINDOW")[0]
    xds_field = xds_from_table(ms_path + "::FIELD")[0]
    phase_dirs_all = xds_field.PHASE_DIR.data.compute()      # (n_fields, 1, 2) radians

    # --- Channel selection ---
    all_chan_freqs = xds_spw.CHAN_FREQ.data.compute()[0]      # (n_chan_total,)
    n_chan_total   = len(all_chan_freqs)
    chan_sel       = _parse_channels(chan_sel_str, n_chan_total)  # index array
    chan_freqs     = all_chan_freqs[chan_sel]                  # Hz, selected
    chan_lams      = 299792458.0 / chan_freqs                  # metres
    n_chan_sel     = len(chan_sel)
    lam_min        = float(np.min(chan_lams))                 # shortest λ (highest freq)

    # daskms partitions by FIELD_ID: multi-field MS → one dataset per scan,
    # single-field MS → one dataset containing all rows.
    # UVW is kept in metres; lambda scaling is applied per-channel during imaging.
    all_datasets = xds_from_ms(ms_path, columns=["UVW", "TIME", "FLAG"])
    multi_field  = len(all_datasets) > 1

    if multi_field:
        all_datasets = sorted(
            all_datasets,
            key=lambda ds: float(ds.TIME.data[0].compute()),
        )
        n_times      = len(all_datasets)
        fid_per_time = np.array([int(ds.attrs["FIELD_ID"]) for ds in all_datasets])
        times_mjd    = np.array([float(ds.TIME.data[0].compute())
                                 for ds in all_datasets]) / (24 * 3600)
        # uvw_t_m: (n_times, rows_per_time, 3) metres
        uvw_t_m = np.stack([ds.UVW.data.compute() for ds in all_datasets])
        # flags_t: (n_times, rows_per_time, n_chan_sel)
        flags_t = np.stack([ds.FLAG.data.compute()[:, chan_sel, 0]
                            for ds in all_datasets])
        rows_per_time = uvw_t_m.shape[1]
        uniform_rows  = True
        sort_idx      = None
        time_idx      = None
        print(f"  Multi-scan MS: {n_times} partitions (one per scan).")
    else:
        xds       = all_datasets[0]
        uvw_all   = xds.UVW.data.compute()                   # metres
        times_all = xds.TIME.data.compute()
        flags_all = xds.FLAG.data.compute()[:, chan_sel, 0]  # (n_rows, n_chan_sel)

        unique_times  = np.unique(times_all)
        n_times       = len(unique_times)
        times_mjd     = unique_times / (24 * 3600)
        fid_per_time  = np.zeros(n_times, dtype=int)

        sort_idx = np.argsort(times_all, kind="stable")
        _, counts = np.unique(times_all, return_counts=True)
        uniform_rows  = bool(np.all(counts == counts[0]))
        rows_per_time = int(counts[0])

        if uniform_rows:
            # uvw_t_m: (n_times, rows_per_time, 3) metres
            uvw_t_m = uvw_all[sort_idx].reshape(n_times, rows_per_time, 3)
            # flags_t: (n_times, rows_per_time, n_chan_sel)
            flags_t = flags_all[sort_idx].reshape(n_times, rows_per_time, n_chan_sel)
            time_idx = None
        else:
            uvw_t_m  = uvw_all                                # (n_rows, 3) metres
            flags_t  = flags_all                              # (n_rows, n_chan_sel)
            time_idx = [np.where(times_all == t)[0] for t in unique_times]

    # Report w-range and auto-estimate n_wplanes (use lam_min for worst-case w/λ)
    if use_wstack:
        w_all_m    = uvw_t_m[:, :, 2].ravel() if uniform_rows else uvw_t_m[:, 2]
        w_all_wl   = w_all_m / lam_min
        w_min_val, w_max_val = float(w_all_wl.min()), float(w_all_wl.max())
        if n_wplanes_cfg is None:
            n_wplanes_cfg = estimate_n_wplanes(w_all_wl, n_pix, pixel_rad)
        print(f"  W-range (λ_min={lam_min:.3f} m): [{w_min_val:.1f}, {w_max_val:.1f}] λ "
              f"| n_wplanes={n_wplanes_cfg}")

    phase_ra_per_time  = np.rad2deg(phase_dirs_all[fid_per_time, 0, 0])
    phase_dec_per_time = np.rad2deg(phase_dirs_all[fid_per_time, 0, 1])
    phase_ra_deg  = float(np.mean(phase_ra_per_time))
    phase_dec_deg = float(np.mean(phase_dec_per_time))

    t1 = _time.perf_counter()
    freq_range_str = (f"{chan_freqs[0]/1e6:.3f} MHz" if n_chan_sel == 1
                      else f"{chan_freqs[0]/1e6:.3f}–{chan_freqs[-1]/1e6:.3f} MHz "
                           f"({n_chan_sel} chans, {mode.upper()})")
    print(f"  Done in {t1 - t0:.2f}s | n_times={n_times} | {freq_range_str} "
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
    # 3. Build WCS
    # -----------------------------------------------------------------------
    # For a multi-scan MS each frame has its own phase centre, so source
    # pixel positions must be computed per-frame.  A single reference WCS
    # (mean phase centre) is used for the matplotlib figure axes so the
    # coordinate grid remains stable across frames.
    wcs_ref = _build_wcs(phase_ra_deg, phase_dec_deg, pixel_deg, n_pix)
    if multi_field:
        wcs = [
            _build_wcs(float(phase_ra_per_time[t]), float(phase_dec_per_time[t]),
                       pixel_deg, n_pix)
            for t in range(n_times)
        ]
    else:
        wcs = wcs_ref

    # -----------------------------------------------------------------------
    # 4. Process each data type
    # -----------------------------------------------------------------------
    for data_type in data_types:
        assert data_type in names, (
            f"'{data_type}' not found in config. Available: {list(names.keys())}"
        )
        data_col = names[data_type]
        tab_data_suffix = prepend_suffix(args.tab_data) if tab_suffix else ""
        gif_base = f"{data_type}{tab_suffix}{img_suffix}{gif_suffix}{tab_data_suffix}"
        print(f"\n[3/4] Imaging data column '{data_col}' -> '{gif_base}'")

        # Read visibilities for this column — all selected channels
        t0 = _time.perf_counter()
        if multi_field:
            vis_datasets = sorted(
                xds_from_ms(ms_path, columns=[data_col]),
                key=lambda ds: int(ds.attrs["FIELD_ID"]),
            )
            vis_t_arr = np.stack([
                ds[data_col].data.compute()[:, chan_sel, 0] for ds in vis_datasets
            ])  # (n_times, rows_per_time, n_chan_sel)
            vis_raw = None
        else:
            xds_vis = xds_from_ms(ms_path, columns=[data_col])[0]
            vis_raw = xds_vis[data_col].data.compute()[:, chan_sel, 0]  # (n_rows, n_chan_sel)
            if uniform_rows:
                vis_t_arr = vis_raw[sort_idx].reshape(n_times, rows_per_time, n_chan_sel)
            else:
                vis_t_arr = None
        t1 = _time.perf_counter()
        print(f"  Visibility read: {t1 - t0:.2f}s")

        # --- Build imaging runs ---
        # MFS:     one pass concatenating all selected channels → one GIF
        # perchan: one pass per channel  → one GIF per channel
        if mode == "mfs":
            imaging_runs: list[tuple[np.ndarray, str]] = [(np.arange(n_chan_sel), "")]
        else:
            imaging_runs = [
                (np.array([ci]), f"_ch{chan_sel[ci]}")
                for ci in range(n_chan_sel)
            ]

        perchan_lcs: list[np.ndarray] = []  # accumulate for spectrogram

        for run_chans, run_suffix in imaging_runs:
            run_gif_name = f"{gif_base}{run_suffix}"
            run_lams = chan_lams[run_chans]  # wavelengths (m) for this run's channels

            # --- Compute dirty images for all time steps ---
            t0 = _time.perf_counter()
            images = []
            for t_idx in range(n_times):
                # Concatenate channels into one (u,v,w,vis) set (MFS) or use one channel
                u_list: list[np.ndarray] = []
                v_list: list[np.ndarray] = []
                w_list: list[np.ndarray] = []
                vis_list: list[np.ndarray] = []

                for ci_idx, lam_c in zip(run_chans, run_lams):
                    if uniform_rows:
                        u_m   = uvw_t_m[t_idx, :, 0]
                        v_m   = uvw_t_m[t_idx, :, 1]
                        w_m   = uvw_t_m[t_idx, :, 2]
                        vis_c  = vis_t_arr[t_idx, :, ci_idx]   # type: ignore[index]
                        flag_c = flags_t[t_idx, :, ci_idx]
                    else:
                        row_idx = time_idx[t_idx]              # type: ignore[index]
                        u_m   = uvw_t_m[row_idx, 0]
                        v_m   = uvw_t_m[row_idx, 1]
                        w_m   = uvw_t_m[row_idx, 2]
                        src   = vis_t_arr if vis_t_arr is not None else vis_raw
                        vis_c  = src[row_idx, ci_idx]          # type: ignore[index]
                        flag_c = flags_t[row_idx, ci_idx]

                    valid_c = ~flag_c.astype(bool)
                    if valid_c.sum() == 0:
                        continue
                    u_list.append(u_m[valid_c] / lam_c)
                    v_list.append(v_m[valid_c] / lam_c)
                    w_list.append(w_m[valid_c] / lam_c)
                    vis_list.append(vis_c[valid_c])

                if not u_list:
                    images.append(np.zeros((n_pix, n_pix), dtype=np.float32))
                    continue

                u_v   = np.concatenate(u_list)
                v_v   = np.concatenate(v_list)
                w_v   = np.concatenate(w_list)
                vis_v = np.concatenate(vis_list)

                wgt = _compute_weights(u_v, v_v, n_pix, pixel_rad, weight_scheme, briggs_robust)
                if use_wstack:
                    im = dirty_image_wstack(
                        u_v, v_v, w_v, vis_v, n_pix, pixel_rad,
                        weights=wgt, n_wplanes=n_wplanes_cfg,
                    )
                else:
                    im = _dirty_image(u_v, v_v, vis_v, n_pix, pixel_rad, wgt)
                images.append(im)

                if t_idx == 0:
                    t_jit = _time.perf_counter()
                    print(f"  JIT compile (1st image): {t_jit - t0:.2f}s")

            t1 = _time.perf_counter()
            print(f"  All {n_times} dirty images: {t1 - t0:.2f}s")

            # --- Render GIF ---
            print(f"\n[4/4] Rendering {n_times} frames -> '{run_gif_name}'...")
            t0 = _time.perf_counter()

            data_arr = np.stack(images)
            vmin, vmax = np.percentile(data_arr, [0.5, 99.5])
            light_curves = np.empty((n_sources, n_times, 3))
            frames = []
            times_sec = (times_mjd - times_mjd[0]) * 86400.0

            # Pre-compute aperture masks and pixel centres (avoids repeated WCS calls).
            # For multi-scan MSes, wcs is a per-frame list so each source position
            # is projected onto its own scan's phase centre.
            t_pre = _time.perf_counter()
            ap_masks, pix_xy, ap_radius_pix = _aperture_pixel_coords(
                wcs, radec_all, aperture_radius_deg, n_pix
            )
            print(f"  Aperture pre-compute: {_time.perf_counter() - t_pre:.2f}s")

            # Build figure once, update data each frame (saves ~50% of matplotlib overhead).
            # Always use the reference WCS for the figure axes so the coordinate grid
            # is stable even when individual frames have different phase centres.
            fig, ax, im_obj = _make_frame_setup(wcs_ref, vmin, vmax)
            buf = io.BytesIO()

            for f_idx, im_data in enumerate(images):
                im_obj.set_data(im_data)
                im_obj.set_extent([-0.5, n_pix - 0.5, -0.5, n_pix - 0.5])

                while len(ax.patches) > 0:
                    ax.patches[-1].remove()
                while len(ax.texts) > 0:
                    ax.texts[-1].remove()

                for s_idx in range(n_sources):
                    mask = ap_masks[s_idx][f_idx]
                    x_c, y_c = pix_xy[s_idx, f_idx]

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

                buf.seek(0)
                buf.truncate()
                fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
                buf.seek(0)
                frames.append(Image.open(buf).copy().convert("P"))

            plt.close(fig)

            gif_path = os.path.join(save_path, f"{run_gif_name}.gif")
            frames[0].save(
                gif_path,
                save_all=True,
                append_images=frames[1:],
                duration=20,
                loop=0,
            )
            print(f"GIF saved as {gif_path}")

            np.save(os.path.join(save_path, f"{run_gif_name}_light_curves.npy"), light_curves)
            plot_light_curves(
                times_sec,
                light_curves,
                titles,
                os.path.join(save_path, f"{run_gif_name}.png"),
            )
            if mode == "perchan":
                perchan_lcs.append(light_curves)

            t1 = _time.perf_counter()
            print(f"  Frame render + GIF: {t1 - t0:.2f}s")

        # --- Spectrogram + spectrum (perchan mode, more than one channel) ---
        if mode == "perchan" and len(perchan_lcs) > 1:
            lc_stack = np.stack(perchan_lcs)  # (n_chan_sel, n_sources, n_times, 3)
            freqs_mhz = chan_freqs / 1e6
            spec_path = os.path.join(save_path, f"{gif_base}_spectrogram.png")
            plot_spectrogram(times_sec, freqs_mhz, lc_stack, titles, spec_path)
            print(f"Spectrogram saved as {spec_path}")
            spect_path = os.path.join(save_path, f"{gif_base}_spectrum.png")
            plot_spectrum(freqs_mhz, lc_stack, titles, spect_path)
            print(f"Spectrum saved as {spect_path}")

    t_total = _time.perf_counter() - _t_start
    print(f"\nTotal wall time: {t_total:.2f}s")


if __name__ == "__main__":
    main()
