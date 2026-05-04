#!/usr/bin/env python3
"""
nufft_gif.py - Create GIF directly from a Measurement Set using jax-finufft.

Replaces the two-step: `extract` (WSClean) + `mk-gif` workflow.

Usage:
    nufft-gif -c img_starlink_1chan.yaml -d tab_rfi -tsx sgp4_bstar_var
"""

import argparse
import os


# ---------------------------------------------------------------------------
# CLI — only argparse and os at module level so --help is instantaneous
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
    parser.add_argument("-ng", "--no-gif", action="store_true", default=False,
                        help="Skip GIF rendering; only produce light curve plots and "
                             "spectrograms.  Imaging is still performed to extract aperture "
                             "statistics, but no matplotlib frames are rendered.")
    parser.add_argument("-pcl", "--per-chan-lc", action="store_true", default=False,
                        help="In perchan mode, save one light curve PNG per channel "
                             "(original behaviour).  Default is a single grid figure "
                             "with one subplot per frequency channel.")
    parser.add_argument("-fs", "--frame-shift", type=float, default=None,
                        help="Add a time-shifted copy of each satellite track as a background "
                             "sky reference.  Value in (-1, 1) sets the shift as a fraction of "
                             "the total track length (e.g. 0.1 → shift by 10%% of timesteps). "
                             "The light curve is shifted back after extraction so it aligns in "
                             "time with the original satellite track.")
    args = parser.parse_args()
    _run(args)


# ---------------------------------------------------------------------------
# Processing — all heavy imports deferred here so --help is instantaneous
# ---------------------------------------------------------------------------

def _run(args):
    import time as _time
    _t_start = _time.perf_counter()

    import io
    import warnings
    import numpy as np

    # nufft_helpers imports jax, matplotlib (sets Agg backend), and astropy
    from riflow.imaging.nufft_helpers import (
        _parse_pixel_deg, _build_wcs, _parse_channels,
        _dirty_image, _aperture_pixel_coords, _estimate_noise_level, _make_frame_setup,
    )
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    from PIL import Image

    from daskms import xds_from_ms, xds_from_table

    from riflow import load_config
    from riflow.config import prepend_suffix
    from riflow.coords import get_tles, sat_radec, get_fornax_radec
    from riflow.io.ms import read_ants_itrf, recopy_tab_results
    from riflow.extraction.light_curves import (
        plot_light_curves, plot_perchan_lc_grid, plot_spectrogram, plot_spectrum,
    )
    from riflow.imaging.weights import parse_weight, compute_weights
    from riflow.imaging.wstack import dirty_image_wstack, estimate_n_wplanes
    from astropy.wcs import FITSFixedWarning
    from tqdm import tqdm

    warnings.filterwarnings("ignore", category=FITSFixedWarning)

    # -----------------------------------------------------------------------
    # Setup
    # -----------------------------------------------------------------------

    config = load_config(args.config_path)

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

    img_params = config["image"]["params"]
    n_pix      = int(str(img_params["size"]).split()[0])
    pixel_deg  = _parse_pixel_deg(str(img_params["scale"]))
    pixel_rad  = np.deg2rad(pixel_deg)

    weight_str = str(img_params.get("weight", "natural"))
    weight_scheme, briggs_robust = parse_weight(weight_str)

    use_wstack    = args.wstack or bool(img_params.get("wstack", False))
    n_wplanes_cfg = args.n_wplanes or img_params.get("n_wplanes", None)
    if n_wplanes_cfg is not None:
        n_wplanes_cfg = int(n_wplanes_cfg)

    chan_sel_str = args.channels or str(img_params.get("channels", "")) or ":"
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

    save_path    = os.path.join(image_path, "gifs")
    lc_data_path = os.path.join(save_path, "lc_data")
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(lc_data_path, exist_ok=True)

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

    all_chan_freqs = xds_spw.CHAN_FREQ.data.compute()[0]      # (n_chan_total,)
    n_chan_total   = len(all_chan_freqs)
    chan_sel       = _parse_channels(chan_sel_str, n_chan_total)
    chan_freqs     = all_chan_freqs[chan_sel]                  # Hz, selected
    chan_lams      = 299792458.0 / chan_freqs                  # metres
    n_chan_sel     = len(chan_sel)
    lam_min        = float(np.min(chan_lams))

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
        uvw_t_m      = np.stack([ds.UVW.data.compute() for ds in all_datasets])
        flags_t      = np.stack([ds.FLAG.data.compute()[:, chan_sel, 0]
                                 for ds in all_datasets])
        rows_per_time = uvw_t_m.shape[1]
        uniform_rows  = True
        sort_idx      = None
        time_idx      = None
        print(f"  Multi-scan MS: {n_times} partitions (one per scan).")
    else:
        xds       = all_datasets[0]
        uvw_all   = xds.UVW.data.compute()
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
            uvw_t_m  = uvw_all[sort_idx].reshape(n_times, rows_per_time, 3)
            flags_t  = flags_all[sort_idx].reshape(n_times, rows_per_time, n_chan_sel)
            time_idx = None
        else:
            uvw_t_m  = uvw_all
            flags_t  = flags_all
            time_idx = [np.where(times_all == t)[0] for t in unique_times]

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
    obs_xyz   = np.mean(ants_itrf, axis=0)

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

    # Optional time-shifted copies of satellite tracks for sky background estimation
    frame_shift = args.frame_shift
    shift_steps = 0
    if frame_shift is not None and radec_sats is not None:
        assert -1 < frame_shift < 1, "--frame-shift must be strictly between -1 and 1"
        shift_steps = int(n_times * frame_shift)
        radec_shifted = np.roll(radec_sats, shift_steps, axis=-1)
        shifted_labels = [f"{lbl} Shifted" for lbl in sat_labels]
    else:
        radec_shifted = None
        shifted_labels = []

    fornax_radec = get_fornax_radec(n_times)  # (1, 2, n_times)

    parts  = []
    titles = []
    if radec_sats is not None:
        parts.append(radec_sats)
        titles += sat_labels
    if radec_shifted is not None:
        parts.append(radec_shifted)
        titles += shifted_labels
    parts.append(fornax_radec)
    titles.append("Fornax A")
    radec_all = np.concatenate(parts, axis=0)

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
        data_col        = names[data_type]
        tab_data_suffix = prepend_suffix(args.tab_data) if tab_suffix else ""
        gif_base        = f"{data_type}{tab_suffix}{img_suffix}{gif_suffix}{tab_data_suffix}"
        print(f"\n[3/4] Imaging data column '{data_col}' -> '{gif_base}'")

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
            vis_raw = xds_vis[data_col].data.compute()[:, chan_sel, 0]
            if uniform_rows:
                vis_t_arr = vis_raw[sort_idx].reshape(n_times, rows_per_time, n_chan_sel)
            else:
                vis_t_arr = None
        t1 = _time.perf_counter()
        print(f"  Visibility read: {t1 - t0:.2f}s")

        # MFS:     one pass concatenating all selected channels → one GIF
        # perchan: one pass per channel  → one GIF per channel
        if mode == "mfs":
            mfs_suffix = f"_mfs-chan_{chan_sel[0]}-{chan_sel[-1]}"
            imaging_runs: list[tuple[np.ndarray, str]] = [(np.arange(n_chan_sel), mfs_suffix)]
        else:
            imaging_runs = [
                (np.array([ci]), f"_ch{chan_sel[ci]}")
                for ci in range(n_chan_sel)
            ]

        perchan_lcs:   list[np.ndarray] = []
        perchan_noise: list[float]     = []

        if mode == "perchan":
            print(f"  Saving GIFs as {os.path.join(save_path, gif_base)}_ch<N>.gif")
        chan_desc = "Channels" if mode == "perchan" else "MFS"
        for run_chans, run_suffix in tqdm(imaging_runs, desc=chan_desc, unit="chan"):
            run_gif_name = f"{gif_base}{run_suffix}"
            run_lams     = chan_lams[run_chans]

            # --- Compute dirty images for all time steps ---
            images = []
            for t_idx in range(n_times):
                u_list:   list[np.ndarray] = []
                v_list:   list[np.ndarray] = []
                w_list:   list[np.ndarray] = []
                vis_list: list[np.ndarray] = []

                for ci_idx, lam_c in zip(run_chans, run_lams):
                    if uniform_rows:
                        u_m    = uvw_t_m[t_idx, :, 0]
                        v_m    = uvw_t_m[t_idx, :, 1]
                        w_m    = uvw_t_m[t_idx, :, 2]
                        vis_c  = vis_t_arr[t_idx, :, ci_idx]   # type: ignore[index]
                        flag_c = flags_t[t_idx, :, ci_idx]
                    else:
                        row_idx = time_idx[t_idx]              # type: ignore[index]
                        u_m    = uvw_t_m[row_idx, 0]
                        v_m    = uvw_t_m[row_idx, 1]
                        w_m    = uvw_t_m[row_idx, 2]
                        src    = vis_t_arr if vis_t_arr is not None else vis_raw
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

                wgt = compute_weights(u_v, v_v, n_pix, pixel_rad, weight_scheme, briggs_robust)
                if use_wstack:
                    im = dirty_image_wstack(
                        u_v, v_v, w_v, vis_v, n_pix, pixel_rad,
                        weights=wgt, n_wplanes=n_wplanes_cfg,
                    )
                else:
                    im = _dirty_image(u_v, v_v, vis_v, n_pix, pixel_rad, wgt)
                images.append(im)


            # --- Extract light curves + render GIF ---
            no_gif = args.no_gif

            data_arr     = np.stack(images)
            vmin, vmax   = np.percentile(data_arr, [0.5, 99.5])
            light_curves = np.empty((n_sources, n_times, 2))
            times_sec    = (times_mjd - times_mjd[0]) * 86400.0

            ap_masks, pix_xy, ap_radius_pix = _aperture_pixel_coords(
                wcs, radec_all, aperture_radius_deg, n_pix
            )
            noise_level = _estimate_noise_level(images, ap_masks, n_pix)

            if no_gif:
                for f_idx, im_data in enumerate(images):
                    for s_idx in range(n_sources):
                        mask = ap_masks[s_idx][f_idx]
                        if len(mask[0]) > 0:
                            region = im_data[mask]
                            light_curves[s_idx, f_idx] = [region.std(), np.max(np.abs(region))]
                        else:
                            light_curves[s_idx, f_idx] = np.nan
            else:
                # Build figure once, update data each frame (saves ~50% of matplotlib overhead).
                # Always use the reference WCS so the coordinate grid is stable across frames.
                frames = []
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
                            light_curves[s_idx, f_idx] = [region.std(), np.max(np.abs(region))]
                        else:
                            light_curves[s_idx, f_idx] = np.nan

                        ax.add_patch(Circle(
                            (x_c, y_c), ap_radius_pix,
                            edgecolor="red", facecolor="none", linewidth=1.5,
                        ))
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

            # Roll shifted tracks back so they align in time with the satellite
            if shift_steps and shifted_labels:
                for s_idx, title in enumerate(titles):
                    if "Shifted" in title:
                        light_curves[s_idx] = np.roll(
                            light_curves[s_idx], -shift_steps, axis=0
                        )

            np.savez(
                os.path.join(lc_data_path, f"{run_gif_name}_light_curves.npz"),
                light_curves=light_curves,
                noise_level=np.array(noise_level),
            )
            if mode == "mfs" or args.per_chan_lc:
                plot_light_curves(
                    times_sec, light_curves, titles,
                    os.path.join(save_path, f"{run_gif_name}.png"),
                    noise_level=noise_level,
                )
            if mode == "perchan":
                perchan_lcs.append(light_curves)
                perchan_noise.append(noise_level)


        chan_range_suffix = f"_chan_{chan_sel[0]}-{chan_sel[-1]}"

        # --- Per-channel light curve grid (perchan mode, default) ---
        if mode == "perchan" and not args.per_chan_lc and len(perchan_lcs) > 0:
            grid_path = os.path.join(save_path, f"{gif_base}_perchan_lc_grid{chan_range_suffix}.png")
            plot_perchan_lc_grid(
                times_sec, perchan_lcs, chan_freqs / 1e6, titles, grid_path,
                noise_levels=perchan_noise,
            )
            print(f"Per-channel LC grid saved as {grid_path}")

        # --- Spectrogram + spectrum (perchan mode, more than one channel) ---
        if mode == "perchan" and len(perchan_lcs) > 1:
            lc_stack    = np.stack(perchan_lcs)
            freqs_mhz   = chan_freqs / 1e6
            noise_level = float(np.median(perchan_noise))
            for stat_idx, stat_tag in ((0, "std"), (1, "maxabs")):
                sp = os.path.join(save_path, f"{gif_base}_spectrogram_{stat_tag}{chan_range_suffix}.png")
                plot_spectrogram(times_sec, freqs_mhz, lc_stack, titles, sp,
                                 stat_idx=stat_idx, noise_level=noise_level)
                print(f"Spectrogram ({stat_tag}) saved as {sp}")
            spect_path = os.path.join(save_path, f"{gif_base}_spectrum{chan_range_suffix}.png")
            plot_spectrum(freqs_mhz, lc_stack, titles, spect_path, noise_level=noise_level)
            print(f"Spectrum saved as {spect_path}")

    print(f"\nTotal wall time: {_time.perf_counter() - _t_start:.2f}s")


if __name__ == "__main__":
    main()
