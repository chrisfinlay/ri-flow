#!/usr/bin/env python3
"""
vis_analysis.py — Visibility-space diagnostics from a Measurement Set.

Produces gridded UV-plane plots for rapid quality assessment without imaging.

Two plot types are generated per data column:
  1. Per-channel grid  — one subplot per selected frequency channel, all times.
  2. Per-time-chunk grid — one subplot per time chunk (all selected channels
     combined), splitting the observation into a configurable number of chunks.
  3. Full grid — single two-panel plot (|mean(vis)| and std) over all selected
     times and channels.

Usage:
    vis-analysis -c config.yaml -d real -ch 0:32 --uv-chunks 25
"""

import argparse
import os


def main():
    parser = argparse.ArgumentParser(
        description="Visibility-space diagnostics (gridded UV plots) from a Measurement Set."
    )
    parser.add_argument("-c", "--config_path", required=True, help="Config YAML path.")
    parser.add_argument("-d", "--data_types", default="real",
                        help="Comma-separated data column keys from config (default: real).")
    parser.add_argument("-ms", "--ms_path", default=None, help="Override MS path.")
    parser.add_argument("-i", "--image_path", default=None,
                        help="Override output directory.")
    parser.add_argument("-tsx", "--tab_suffix", default=None,
                        help="TABASCAL solution suffix (e.g. sgp4_bstar_var).")
    parser.add_argument("-rc", "--recopy", action="store_true", default=False,
                        help="Re-copy TABASCAL results from the matching zarr into the MS "
                             "before analysis.  Requires -tsx.")
    parser.add_argument("-td", "--tab_data", default="map",
                        help="Zarr result type: 'map' (default) or 'init'.")
    parser.add_argument("-mn", "--model_name", default="Custom",
                        help="TABASCAL model name encoded in the zarr filename (default: 'Custom').")
    parser.add_argument("-ch", "--channels", default=None,
                        help="Channel selection, numpy-style: ':', '0:16', '::2', '0,4,8'. "
                             "Default: all channels.")
    parser.add_argument("-tr", "--time-range", default=None,
                        help="Time index selection, numpy-style: '0:50', '::2'. "
                             "Default: all timesteps.")
    parser.add_argument("-uvc", "--uv-chunks", type=int, default=25,
                        help="Number of time chunks for the per-chunk UV grid (default: 25).")
    parser.add_argument("-ng", "--n-grid", type=int, default=256,
                        help="UV grid resolution along each axis (default: 256).")
    parser.add_argument("-sx", "--suffix", default=None,
                        help="Extra suffix appended to all output filenames.")
    args = parser.parse_args()
    _run(args)


def _run(args):
    import numpy as np
    import warnings

    from daskms import xds_from_ms, xds_from_table
    from astropy.wcs import FITSFixedWarning

    from tqdm import tqdm

    from riflow import load_config
    from riflow.config import prepend_suffix
    from riflow.imaging.nufft_helpers import _parse_channels
    from riflow.visibilities.vis_helpers import grid_visibilities, save_uv_grid, save_uv_pair

    warnings.filterwarnings("ignore", category=FITSFixedWarning)

    # -----------------------------------------------------------------------
    # Setup
    # -----------------------------------------------------------------------
    config = load_config(args.config_path)

    ms_path    = os.path.abspath(args.ms_path    or config["data"]["ms_path"])
    image_path = os.path.abspath(args.image_path or config["data"]["image_path"])

    tab_suffix   = prepend_suffix(args.tab_suffix)
    extra_suffix = prepend_suffix(args.suffix)

    data_types = args.data_types.split(",")
    names = {
        key: config[key]["data_col"]
        for key in config.keys()
        if key not in ["data", "image", "extract", "gif"]
    }

    n_grid       = args.n_grid
    n_chunks     = args.uv_chunks
    chan_sel_str = args.channels or ":"

    print(f"MS path   : {ms_path}")

    # -----------------------------------------------------------------------
    # 0. Optionally re-copy TABASCAL results from zarr → MS columns
    # -----------------------------------------------------------------------
    if args.recopy:
        from riflow.io.ms import recopy_tab_results
        sim_dir = config["data"].get("sim_dir") or None
        print(f"\n[0/3] Re-copying TABASCAL results (tab_data='{args.tab_data}', "
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
    # 1. Read MS metadata
    # -----------------------------------------------------------------------
    print("\n[1/3] Reading Measurement Set...")

    xds_spw        = xds_from_table(ms_path + "::SPECTRAL_WINDOW")[0]
    all_chan_freqs  = xds_spw.CHAN_FREQ.data.compute()[0]
    n_chan_total    = len(all_chan_freqs)
    chan_sel        = _parse_channels(chan_sel_str, n_chan_total)
    chan_freqs      = all_chan_freqs[chan_sel]
    chan_lams       = 299792458.0 / chan_freqs
    n_chan_sel      = len(chan_sel)

    all_datasets = xds_from_ms(ms_path, columns=["UVW", "TIME", "FLAG"])
    multi_field  = len(all_datasets) > 1

    if multi_field:
        all_datasets = sorted(all_datasets,
                               key=lambda ds: float(ds.TIME.data[0].compute()))
        n_times      = len(all_datasets)
        times_mjd    = np.array([float(ds.TIME.data[0].compute())
                                  for ds in all_datasets]) / (24 * 3600)
        uvw_t_m      = np.stack([ds.UVW.data.compute() for ds in all_datasets])
        flags_t      = np.stack([ds.FLAG.data.compute()[:, chan_sel, 0]
                                  for ds in all_datasets])
        rows_per_time = uvw_t_m.shape[1]
        uniform_rows  = True
        sort_idx      = None
    else:
        xds        = all_datasets[0]
        uvw_all    = xds.UVW.data.compute()
        times_all  = xds.TIME.data.compute()
        flags_all  = xds.FLAG.data.compute()[:, chan_sel, 0]

        unique_times  = np.unique(times_all)
        n_times       = len(unique_times)
        times_mjd     = unique_times / (24 * 3600)

        sort_idx      = np.argsort(times_all, kind="stable")
        _, counts     = np.unique(times_all, return_counts=True)
        uniform_rows  = bool(np.all(counts == counts[0]))
        rows_per_time = int(counts[0])

        if uniform_rows:
            uvw_t_m = uvw_all[sort_idx].reshape(n_times, rows_per_time, 3)
            flags_t = flags_all[sort_idx].reshape(n_times, rows_per_time, n_chan_sel)
        else:
            print("  Non-uniform rows per timestep — skipping UV diagnostics.")
            return

    times_sec = (times_mjd - times_mjd[0]) * 86400.0

    # Apply time selection
    time_sel_str = args.time_range or ":"
    time_sel     = _parse_channels(time_sel_str, n_times)
    uvw_t_m      = uvw_t_m[time_sel]
    flags_t      = flags_t[time_sel]
    times_sec    = times_sec[time_sel]
    n_times_sel  = len(time_sel)

    chan_range_suffix = f"_chan_{chan_sel[0]}-{chan_sel[-1]}"
    time_range_suffix = f"_tidx_{time_sel[0]}-{time_sel[-1]}"

    # Global UV max: longest baseline at highest frequency (shortest wavelength)
    lam_min  = float(np.min(chan_lams))
    uv_m_max = float(np.max(np.abs(uvw_t_m[:, :, :2])))
    uv_max   = uv_m_max / lam_min * 1.05

    print(f"  {n_times_sel} timesteps × {uvw_t_m.shape[1]} baselines × "
          f"{n_chan_sel} channels | uv_max={uv_max:.0f} λ")

    # -----------------------------------------------------------------------
    # 2. Read visibilities and compute UV grids per data type
    # -----------------------------------------------------------------------
    for data_type in data_types:
        assert data_type in names, (
            f"'{data_type}' not found in config. Available: {list(names.keys())}"
        )
        data_col      = names[data_type]
        tab_data_suffix = prepend_suffix(args.tab_data) if tab_suffix else ""
        base          = f"{data_type}{tab_suffix}{extra_suffix}{tab_data_suffix}"

        vis_root  = os.path.join(image_path, "vis_analysis")
        tab_dir   = args.tab_suffix or ""
        save_path = os.path.join(vis_root, tab_dir, data_type) if tab_dir \
                    else os.path.join(vis_root, data_type)
        os.makedirs(save_path, exist_ok=True)

        print(f"\n[2/3] Reading '{data_col}' → {save_path}/")

        if multi_field:
            vis_datasets = sorted(
                xds_from_ms(ms_path, columns=[data_col]),
                key=lambda ds: int(ds.attrs["FIELD_ID"]),
            )
            vis_t_arr = np.stack([
                ds[data_col].data.compute()[:, chan_sel, 0] for ds in vis_datasets
            ])[time_sel]
        else:
            xds_vis   = xds_from_ms(ms_path, columns=[data_col])[0]
            vis_raw   = xds_vis[data_col].data.compute()[:, chan_sel, 0]
            vis_t_arr = vis_raw[sort_idx].reshape(n_times, rows_per_time, n_chan_sel)[time_sel]

        # -------------------------------------------------------------------
        # 3a. Per-channel UV grid
        # -------------------------------------------------------------------
        chan_amp_grids = []
        chan_std_grids = []
        chan_subtitles = [f"{f/1e6:.2f} MHz" for f in chan_freqs]

        u_m_flat = uvw_t_m[:, :, 0].ravel()
        v_m_flat = uvw_t_m[:, :, 1].ravel()

        for ci in tqdm(range(n_chan_sel), desc="Per-channel UV", unit="chan"):
            lam    = float(chan_lams[ci])
            valid  = ~flags_t[:, :, ci].ravel().astype(bool)
            vis_ci = vis_t_arr[:, :, ci].ravel()
            amp_g, std_g, _ = grid_visibilities(
                u_m_flat[valid] / lam, v_m_flat[valid] / lam,
                vis_ci[valid], n_grid=n_grid, uv_max=uv_max,
            )
            chan_amp_grids.append(amp_g)
            chan_std_grids.append(std_g)

        for grids, stat, cbar_label in (
            (chan_amp_grids, "abs", "|mean(vis)| [Jy]"),
            (chan_std_grids, "std", "std(vis) [Jy]"),
        ):
            path = os.path.join(
                save_path,
                f"{base}_uv_perchan_{stat}{chan_range_suffix}{time_range_suffix}.png",
            )
            save_uv_grid(
                grids, chan_subtitles, uv_max, path,
                suptitle=f"{base} — {cbar_label} per channel",
                cbar_label=cbar_label,
            )
            print(f"  Per-channel UV grid ({stat}) saved as {path}")

        # -------------------------------------------------------------------
        # 3b. Per-time-chunk UV grid (all channels combined)
        # -------------------------------------------------------------------
        chunk_size      = max(1, n_times_sel // n_chunks)
        chunk_amp_grids = []
        chunk_std_grids = []
        chunk_labels    = []

        for k in tqdm(range(n_chunks), desc="Time-chunk UV", unit="chunk"):
            t0 = k * chunk_size
            t1 = min(t0 + chunk_size, n_times_sel)
            if t0 >= n_times_sel:
                break

            u_list, v_list, vis_list = [], [], []
            for ci in range(n_chan_sel):
                lam   = float(chan_lams[ci])
                u_m   = uvw_t_m[t0:t1, :, 0].ravel()
                v_m   = uvw_t_m[t0:t1, :, 1].ravel()
                valid = ~flags_t[t0:t1, :, ci].ravel().astype(bool)
                vis_c = vis_t_arr[t0:t1, :, ci].ravel()
                u_list.append(u_m[valid] / lam)
                v_list.append(v_m[valid] / lam)
                vis_list.append(vis_c[valid])

            amp_g, std_g, _ = grid_visibilities(
                np.concatenate(u_list), np.concatenate(v_list),
                np.concatenate(vis_list), n_grid=n_grid, uv_max=uv_max,
            )
            chunk_amp_grids.append(amp_g)
            chunk_std_grids.append(std_g)
            chunk_labels.append(f"{times_sec[t0]:.0f}–{times_sec[t1-1]:.0f} s")

        for grids, stat, cbar_label in (
            (chunk_amp_grids, "abs", "|mean(vis)| [Jy]"),
            (chunk_std_grids, "std", "std(vis) [Jy]"),
        ):
            path = os.path.join(
                save_path,
                f"{base}_uv_timechunks{n_chunks}_{stat}{chan_range_suffix}{time_range_suffix}.png",
            )
            save_uv_grid(
                grids, chunk_labels, uv_max, path,
                suptitle=f"{base} — {cbar_label} per time chunk",
                cbar_label=cbar_label,
            )
            print(f"  Time-chunk UV grid ({stat}) saved as {path}")

        # -------------------------------------------------------------------
        # 3c. Full time+frequency UV grid (two subplots: abs and std)
        # -------------------------------------------------------------------
        u_list, v_list, vis_list = [], [], []
        for ci in range(n_chan_sel):
            lam   = float(chan_lams[ci])
            valid = ~flags_t[:, :, ci].ravel().astype(bool)
            vis_c = vis_t_arr[:, :, ci].ravel()
            u_list.append(u_m_flat[valid] / lam)
            v_list.append(v_m_flat[valid] / lam)
            vis_list.append(vis_c[valid])

        amp_full, std_full, _ = grid_visibilities(
            np.concatenate(u_list), np.concatenate(v_list),
            np.concatenate(vis_list), n_grid=n_grid, uv_max=uv_max,
        )
        path = os.path.join(
            save_path,
            f"{base}_uv_full{chan_range_suffix}{time_range_suffix}.png",
        )
        save_uv_pair(
            amp_full, std_full, uv_max, path,
            suptitle=f"{base} — full UV coverage",
        )
        print(f"  Full UV grid saved as {path}")


if __name__ == "__main__":
    main()
