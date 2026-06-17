#!/usr/bin/env python3
"""
uv_redundancy.py — UV-plane redundancy / imageability diagnostic from a Measurement Set.

Motivation
----------
A sky image can predict only ONE visibility per uv-cell, so the best any image can
do in a cell is the (inverse-variance) mean of the samples there. On a redundant
array (e.g. EDA2: ~256 antennas -> 32640 baselines into ~2300 uv-cells), the
scatter of the samples *about that mean*, normalised by the thermal noise, is a
reduced chi-squared that should be ~1 if the array is redundant and calibrated.
Values >> 1 mean non-redundant per-baseline / per-antenna structure (miscalibration,
mutual coupling, differing element beams) that NO sky image can represent -- the
fundamental reason a dense-image sky model (tabascal ImageSky) floors well above
the noise while a per-baseline visibility GP can fit the data.

This script grids that per-uv-cell reduced chi-squared and a few companion maps so
the problem is visible at a glance, and prints a single headline number per data
column (the global within-cell reduced chi-squared and the imageable power
fraction). Re-run it on the columns written by successive tabascal runs -- e.g.
after introducing gain/redundant calibration -- and watch the map and chi-squared
fall toward 1 as the non-redundant structure is removed.

Usage
-----
    # explicit columns (compare data before/after a calibration run)
    uv-redundancy -ms path/to.ms --cols REAL_DATA,CORRECTED_DATA -ch 0:27

    # via a riflow config (column names resolved from config like vis-analysis)
    uv-redundancy -c config.yaml -d real,corrected -ch 0:27

The uv-cell is sized to the imaging resolution 1/(2 sin(fov/2)) (=0.5 lambda at
fov 180), so the chi-squared literally means "imageability at that field of view".
"""

import argparse
import os


def main():
    p = argparse.ArgumentParser(
        description="UV-plane redundancy / imageability diagnostic (reduced chi^2 per uv-cell)."
    )
    p.add_argument("-c", "--config_path", default=None,
                   help="Optional riflow config YAML (for ms_path/image_path and column names).")
    p.add_argument("-ms", "--ms_path", default=None, help="Measurement Set path (overrides config).")
    p.add_argument("--cols", default=None,
                   help="Comma-separated MS column names to analyse (e.g. REAL_DATA,CORRECTED_DATA). "
                        "Takes precedence over -d.")
    p.add_argument("-d", "--data_types", default="real",
                   help="Comma-separated config data-type keys (used when --cols is absent).")
    p.add_argument("-i", "--image_path", default=None, help="Output directory (default: <ms dir>).")
    p.add_argument("-ch", "--channels", default=None,
                   help="Channel selection, numpy-style ':', '0:16', '::2', '0,4,8' (default: all).")
    p.add_argument("--fov-deg", type=float, default=180.0,
                   help="Field of view setting the uv-cell = 1/(2 sin(fov/2)) lambda (default 180).")
    p.add_argument("-ng", "--n-grid", type=int, default=None,
                   help="Explicit uv grid resolution (overrides the fov-derived cell).")
    p.add_argument("--sigma", type=float, default=None,
                   help="Per-visibility noise (real/imag std) to use instead of the MS SIGMA column.")
    p.add_argument("--vmax", type=float, default=3.0, help="Colour scale max for the chi^2 maps.")
    p.add_argument("--geom-floor", action="store_true",
                   help="Also subtract the geometric (sub-cell sky-gradient) floor: grid each "
                        "channel to a dirty image and degrid (vis2dirty->dirty2vis), an "
                        "image-consistent round-trip whose within-cell chi^2 is the floor. "
                        "Outputs a non-redundancy map = chi^2(data) - chi^2(round-trip) - noise.")
    p.add_argument("--noise-floor", type=float, default=1.0,
                   help="Within-cell chi^2 attributable to thermal noise, subtracted in the "
                        "non-redundancy map (1.0 for calibrated noisy data; pass 0 for a "
                        "noiseless model column, e.g. vis_ast/TAB_AST_DATA).")
    p.add_argument("--n-pix", type=int, default=180, help="Image size for the geom-floor round-trip.")
    p.add_argument("--eps", type=float, default=1e-6, help="Gridder accuracy for the geom-floor round-trip.")
    p.add_argument("-sx", "--suffix", default="", help="Extra suffix appended to output filenames.")
    args = p.parse_args()
    _run(args)


def _run(args):
    import warnings
    import numpy as np
    from daskms import xds_from_ms, xds_from_table

    from riflow.imaging.nufft_helpers import _parse_channels
    from riflow.visibilities.vis_helpers import (
        grid_redundancy_chi2, finalize_redundancy, save_uv_chi2_grid,
    )

    warnings.filterwarnings("ignore")
    C = 299792458.0

    geom = None
    if args.geom_floor:
        import jax
        jax.config.update("jax_enable_x64", True)
        import jax.numpy as jnp
        from jax_nufft import dirty2vis, vis2dirty
        from tabascal.imaging import make_image_plan

        def geom_roundtrip(uvw_rows, freq, vis_rows):
            """Image-consistent round-trip vis2dirty->dirty2vis for one channel,
            LS-scaled to the data. Its within-cell scatter is the geometric floor."""
            plan = make_image_plan(jnp.asarray(uvw_rows), np.array([freq]),
                                   args.fov_deg, args.n_pix, args.eps,
                                   divide_by_n=False).plan
            v = jnp.asarray(vis_rows).astype(complex)[:, None]
            vrt = np.asarray(dirty2vis(plan, vis2dirty(plan, v)))[:, 0]
            a = np.vdot(vrt, vis_rows) / np.vdot(vrt, vrt)
            return a * vrt
        geom = geom_roundtrip

    # --- resolve ms / columns / output dir --------------------------------
    cfg = None
    if args.config_path:
        from riflow import load_config
        cfg = load_config(args.config_path)
    ms_path = os.path.abspath(args.ms_path or cfg["data"]["ms_path"])

    if args.cols:
        columns = [c.strip() for c in args.cols.split(",")]
    elif cfg is not None:
        names = {k: cfg[k]["data_col"] for k in cfg
                 if k not in ("data", "image", "extract", "gif")}
        columns = [names[k] for k in args.data_types.split(",")]
    else:
        raise SystemExit("Provide --cols, or -c config with -d data_types.")

    image_path = os.path.abspath(
        args.image_path or (cfg["data"]["image_path"] if cfg and "image_path" in cfg["data"]
                            else os.path.dirname(ms_path)))
    out_dir = os.path.join(image_path, "uv_redundancy")
    os.makedirs(out_dir, exist_ok=True)
    sx = (f"_{args.suffix}" if args.suffix and not args.suffix.startswith("_") else args.suffix)

    # --- read metadata -----------------------------------------------------
    freqs = xds_from_table(ms_path + "::SPECTRAL_WINDOW")[0].CHAN_FREQ.data.compute()[0]
    n_chan_total = len(freqs)
    chan_sel = _parse_channels(args.channels or ":", n_chan_total)
    fsel = np.asarray(freqs)[chan_sel]
    n_chan = len(chan_sel)

    xds = xds_from_ms(ms_path, columns=["UVW", "FLAG", "SIGMA", "TIME"])[0]
    uvw = xds.UVW.data.compute()                                    # (nrow, 3)
    flag = xds.FLAG.data.compute()[:, chan_sel, 0]                  # (nrow, n_chan)
    times = xds.TIME.data.compute()
    # A single robust noise level (per real/imag component) for the chi^2
    # normalisation -- the MS SIGMA is per-row and often unreliable (zeros / wide
    # spread), so a per-sample weighting is fragile; one scalar matches the
    # "scatter / sigma ~ 1" interpretation. Override with --sigma.
    if args.sigma is not None:
        sigma = float(args.sigma)
        print(f"  noise sigma = {sigma:.3g} (from --sigma)")
    else:
        sig_col = np.asarray(xds.SIGMA.data.compute())[:, 0].astype(float)
        pos = sig_col[(sig_col > 0) & np.isfinite(sig_col)]
        sigma = float(np.median(pos)) if pos.size else 1.0
        print(f"  noise sigma = {sigma:.3g} (median of positive MS SIGMA)")

    # Group rows by timestep: the redundancy test must pool *within* a snapshot
    # (same time + channel = identical sky), else earth-rotation / uv-gradient
    # masquerades as non-redundancy.
    uniq, t_idx = np.unique(times, return_inverse=True)
    n_time = len(uniq)
    counts = np.bincount(t_idx)
    if not np.all(counts == counts[0]):
        raise SystemExit("Non-uniform rows per timestep; not supported.")
    order = np.argsort(t_idx, kind="stable")
    n_bl = counts[0]
    uvw_t = uvw[order].reshape(n_time, n_bl, 3)
    flag_t = flag[order].reshape(n_time, n_bl, n_chan)

    lam = C / fsel
    uv_max = float(np.max(np.abs(uvw[:, :2]))) / float(lam.min()) * 1.02
    cell = None if args.n_grid else 1.0 / (2.0 * np.sin(np.deg2rad(args.fov_deg) / 2.0))
    n_grid = args.n_grid or int(np.ceil(2.0 * uv_max / cell))
    G2 = n_grid * n_grid
    print(f"MS: {ms_path}")
    print(f"{n_time} times x {n_bl} bl x {n_chan} chan | uv_max={uv_max:.1f} lam | "
          f"cell={2*uv_max/n_grid:.3f} lam ({n_grid}^2 grid, fov={args.fov_deg:g})")

    summary = []
    for col in columns:
        data = xds_from_ms(ms_path, columns=[col])[0][col].data.compute()[:, chan_sel, 0]
        data_t = data[order].reshape(n_time, n_bl, n_chan)

        # per-channel accumulators of the within-snapshot sums
        cs = np.zeros((n_chan, G2)); df = np.zeros((n_chan, G2)); ct = np.zeros((n_chan, G2))
        bn = np.zeros(n_chan); tn = np.zeros(n_chan)
        for t in range(n_time):
            for ci in range(n_chan):
                v = ~flag_t[t, :, ci].astype(bool)
                if not v.any():
                    continue
                s = grid_redundancy_chi2(
                    uvw_t[t, v, 0] / lam[ci], uvw_t[t, v, 1] / lam[ci],
                    data_t[t, v, ci],
                    uv_max=uv_max, n_grid=n_grid, return_sums=True)
                cs[ci] += s["chi2_sum"]; df[ci] += s["dof"]; ct[ci] += s["cnt"]
                bn[ci] += s["betw_num"]; tn[ci] += s["tot_num"]

        chi2_grids, subt = [], []
        for ci in range(n_chan):
            g, _, info = finalize_redundancy(cs[ci], df[ci], ct[ci], bn[ci], tn[ci],
                                             n_grid, sigma)
            chi2_grids.append(g)
            subt.append(f"{fsel[ci]/1e6:.1f} MHz  χ²={info['chi2_global']:.2f}")
        chi2, cntg, info = finalize_redundancy(
            cs.sum(0), df.sum(0), ct.sum(0), bn.sum(), tn.sum(), n_grid, sigma)

        base = f"{col}{sx}"
        if n_chan > 1:
            save_uv_chi2_grid(chi2_grids, subt, uv_max,
                              os.path.join(out_dir, f"{base}_chi2_perchan.png"),
                              suptitle=f"{col}: within-uv-cell redundancy χ² per channel",
                              vmax=args.vmax)
        save_uv_chi2_grid([chi2], [f"all {n_chan} chan  χ²={info['chi2_global']:.2f}  "
                                   f"imageable={100*info['imageable_frac']:.0f}%"],
                          uv_max, os.path.join(out_dir, f"{base}_chi2_combined.png"),
                          suptitle=f"{col}: within-uv-cell redundancy reduced χ²  "
                                   f"(1 = redundant/imageable)", vmax=args.vmax)

        chi2_geom_global = nonredund_global = None
        if geom is not None:
            # Geometric floor: image-consistent round-trip per channel, then the
            # same per-snapshot within-cell statistic. Non-redundancy = data - geom
            # - noise_floor (noise_floor=0 for a noiseless model column).
            csg = np.zeros((n_chan, G2)); dfg = np.zeros((n_chan, G2)); ctg = np.zeros((n_chan, G2))
            for ci in range(n_chan):
                vrt = geom(uvw, float(fsel[ci]), data[:, ci])     # MS row order
                vrt_t = vrt[order].reshape(n_time, n_bl)
                for t in range(n_time):
                    v = ~flag_t[t, :, ci].astype(bool)
                    if not v.any():
                        continue
                    s = grid_redundancy_chi2(
                        uvw_t[t, v, 0] / lam[ci], uvw_t[t, v, 1] / lam[ci],
                        vrt_t[t, v], uv_max=uv_max, n_grid=n_grid, return_sums=True)
                    csg[ci] += s["chi2_sum"]; dfg[ci] += s["dof"]; ctg[ci] += s["cnt"]
            chi2g, _, infog = finalize_redundancy(
                csg.sum(0), dfg.sum(0), ctg.sum(0), 0.0, 1.0, n_grid, sigma)
            chi2_geom_global = infog["chi2_global"]
            nonredund = chi2 - chi2g - args.noise_floor          # per-cell excess map
            nonredund_global = info["chi2_global"] - chi2_geom_global - args.noise_floor
            save_uv_chi2_grid([chi2g], [f"geometric floor  χ²={chi2_geom_global:.2f}"],
                              uv_max, os.path.join(out_dir, f"{base}_geomfloor_combined.png"),
                              suptitle=f"{col}: geometric (sub-cell gradient) floor "
                                       f"[vis2dirty->dirty2vis]", vmax=args.vmax)
            save_uv_chi2_grid([nonredund], [f"non-redundancy  χ²_excess={nonredund_global:.2f}"],
                              uv_max, os.path.join(out_dir, f"{base}_nonredund_combined.png"),
                              suptitle=f"{col}: calibratable non-redundancy "
                                       f"(data - geom - noise={args.noise_floor:g})",
                              vmax=args.vmax, vcenter=0.0, vmin=-1.0,
                              cbar_label="non-redundancy χ² excess  (0 = calibrated)")

        summary.append((col, info["chi2_global"], info["imageable_frac"],
                        chi2_geom_global, nonredund_global))
        msg = (f"  [{col}]  redundancy χ²={info['chi2_global']:.3f}  "
               f"imageable={100*info['imageable_frac']:.1f}%")
        if geom is not None:
            msg += (f"  geom-floor χ²={chi2_geom_global:.3f}  "
                    f"non-redundancy χ²={nonredund_global:.3f}")
        print(msg)

    print("\n=== summary (calibration should drive non-redundancy χ² -> 0, imageable% up) ===")
    for col, c2, imf, c2g, nr in summary:
        line = f"  {col:20s}  χ²={c2:7.3f}  imageable={100*imf:5.1f}%"
        if nr is not None:
            line += f"  geom-floor={c2g:6.3f}  non-redundancy={nr:7.3f}"
        print(line)
    print(f"\nplots -> {out_dir}/")


if __name__ == "__main__":
    main()
