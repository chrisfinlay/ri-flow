#!/usr/bin/env python3
"""
flux_calibrate.py - Simple array-wide flux calibration from Fornax A.

Derives a smooth amplitude calibration k(t, ν) that puts the data on an
approximate Jy scale, using Fornax A (the one source common to all fields) as
the flux reference.  This is a *scalar*, direction-independent calibration:

    V_cal(t, ν) = k(t, ν) · V_obs(t, ν),   k = S_model_FornaxA(ν) / S_obs_FornaxA(t, ν)

The calibration solution is modelled as AT MOST LINEAR in time and in frequency
(a plane), because the underlying instrumental gain is expected to be smooth —
any rapid variation in the Fornax measurement is taken to be noise or satellite
artefacts and is rejected by a robust (iteratively sigma-clipped) fit.

It cannot recover per-antenna gains (per-antenna bandpass, phases, the
direction-dependent station beam) because the rest of the field is unmodelled.
See "Limitations / better options" below before relying on it.

Workflow
--------
1.  Run `nufft-gif -mo perchan` on the column you want to calibrate so the
    per-channel Fornax A light curves are written to <...>/lc_data/*.npz.
2.  Run this script to fit k(t, ν) vs a Fornax A spectral model, write a
    diagnostic plot + the solution, and (optionally, --apply) write a
    flux-scaled column back to the MS.

Robustness to satellite contamination
-------------------------------------
Satellites passing through / near Fornax A's aperture contaminate its measured
flux. Each (time, channel) sample is screened before the smooth fit:

  * out-of-FOV samples (NaN) are dropped;
  * a time/channel is flagged if any satellite aperture |flux| exceeds
    --sat-factor × the Fornax median in that channel (sidelobe / overlap risk);
  * a sample is flagged if Fornax itself spikes above median + --spike-nsigma·MAD
    in that channel;
  * with --tle-flag, times where a satellite is within --tle-radius-deg of
    Fornax A (from TLE propagation) are flagged for all channels;
  * the robust plane fit then sigma-clips (--fit-nsigma) any remaining outliers.

This covers: clean data (nothing flagged); weak satellites passing close (time
flags); strong satellites in specific channels (those samples / the whole
channel drop out and the smooth fit interpolates); and strong satellites across
the whole band (too little survives -> declared UNCALIBRATABLE, see
--min-good-frac / --allow-uncalibratable).

Fornax A model
--------------
A log-polynomial spectrum, by default a power law:

    log10 S(ν) = log10(S_ref) + alpha · log10(ν / ν_ref)

The DEFAULTS BELOW ARE APPROXIMATE (lobe-dominated, ~MWA-era integrated flux)
and should be replaced with a literature spectrum for your band — e.g.
McKinley et al. 2015 (MNRAS 446, 3478) or McKinley et al. 2020 (Nat.Astron. 4,
1182) for MWA Fornax A measurements.  Override with --fornax-flux / --fornax-freq
/ --fornax-index, or pass an explicit log-polynomial with --fornax-logpoly.

Limitations / better initial-calibration options (before TABASCAL)
------------------------------------------------------------------
* The absolute Jy scale folds in the station-beam response at Fornax (no beam
  model); the *relative* bandpass shape is as good as the assumed spectral index.
  Smooth time variation is absorbed into the gain — fine when it is genuine
  electronic drift, but it will also absorb a slow beam transit, so keep
  --time-order 0 if the Fornax level shows a strong (beam) trend in time.
* For a real gain calibration of an all-sky aperture array like EDA2, the
  highest-value next step is sky-model self-calibration (image -> CLEAN model ->
  per-antenna complex gain solve -> iterate; phase self-cal alone is a big win),
  or predicting visibilities from a Global Sky Model (pygdsm GSM2016/LFSM) plus a
  GLEAM A-team catalogue through the station beam and solving gains against that.
* EDA2's layout is non-redundant, so redundant calibration does not apply.

Usage
-----
    flux-calibrate -c img_solrad_polxx.yaml -d real \
        --fornax-flux 750 --fornax-freq 154 --fornax-index -0.77
    flux-calibrate -c img_solrad_polxx.yaml -d real --tle-flag
    flux-calibrate -c img_solrad_polxx.yaml -d real --apply
"""

import argparse
import glob
import os


def _fornax_model_jy(freq_mhz, logpoly, ref_freq_mhz):
    """Log-polynomial Fornax A spectrum. logpoly = [c0, c1, c2, ...] where
    log10(S) = sum_i c_i * (log10(nu/nu_ref))**i."""
    import numpy as np
    x = np.log10(np.asarray(freq_mhz, dtype=float) / ref_freq_mhz)
    log_s = sum(c * x**i for i, c in enumerate(logpoly))
    return 10.0**log_s


def _mad_sigma(x):
    """Robust (median, sigma) via the MAD. x must be finite and non-empty."""
    import numpy as np
    med = float(np.median(x))
    sig = 1.4826 * float(np.median(np.abs(x - med)))
    return med, sig


def _channel_flags(fornax_lc, sat_lc, sat_factor, spike_nsigma):
    """Per-time clean mask for one channel.

    Drops out-of-FOV samples, times where a satellite aperture is bright
    (sat_lc > sat_factor × Fornax median) and Fornax spikes (> median +
    spike_nsigma·MAD).  Returns (clean, sat_flag, spike_flag, fornax_median,
    sat_level)."""
    import numpy as np
    n = fornax_lc.size
    in_fov   = np.isfinite(fornax_lc)
    base_med = float(np.nanmedian(fornax_lc[in_fov])) if in_fov.any() else np.nan
    sat_level = (float(np.nanmedian(sat_lc[np.isfinite(sat_lc)]))
                 if np.isfinite(sat_lc).any() else 0.0)

    sat_flag = np.zeros(n, bool)
    if sat_factor and sat_factor > 0 and np.isfinite(base_med) and base_med > 0:
        sat_flag = np.nan_to_num(sat_lc, nan=0.0) > sat_factor * base_med

    spike_flag = np.zeros(n, bool)
    base = in_fov & ~sat_flag
    if base.sum() >= 3:
        med, sig = _mad_sigma(fornax_lc[base])
        if sig > 0:
            spike_flag = in_fov & (fornax_lc > med + spike_nsigma * sig)

    clean = in_fov & ~sat_flag & ~spike_flag
    return clean, sat_flag, spike_flag, base_med, sat_level


def _robust_plane(t, nu, k, time_order, freq_order, nsigma, n_iter=5):
    """Robust least-squares fit of k as an affine (degree ≤1 per axis) function
    of time and frequency, iteratively sigma-clipping outliers.

    t, nu, k : 1-D arrays of clean samples.
    Returns a dict of explicit coefficients + normalisation and the used mask:
        k(t, ν) = c0 + ct·(t - t0)/ts + cnu·(ν - n0)/ns
    """
    import numpy as np
    t = np.asarray(t, float); nu = np.asarray(nu, float); k = np.asarray(k, float)
    t0 = float(np.median(t)); ts = float(np.ptp(t) / 2) or 1.0
    n0 = float(np.median(nu)); ns = float(np.ptp(nu) / 2) or 1.0

    use_t, use_n = time_order >= 1, freq_order >= 1
    cols = [np.ones_like(k)]
    if use_t: cols.append((t - t0) / ts)
    if use_n: cols.append((nu - n0) / ns)
    A = np.vstack(cols).T

    used = np.ones(k.size, bool)
    coef = np.linalg.lstsq(A, k, rcond=None)[0]
    for _ in range(n_iter):
        if used.sum() < A.shape[1] + 1:
            break
        coef = np.linalg.lstsq(A[used], k[used], rcond=None)[0]
        resid = k - A @ coef
        med, sig = _mad_sigma(resid[used])
        if sig <= 0:
            break
        new = np.abs(resid - med) <= nsigma * sig
        if int(new.sum()) == int(used.sum()):
            used = new
            break
        used = new

    c0 = float(coef[0]); j = 1
    ct = float(coef[j]) if use_t else 0.0; j += int(use_t)
    cnu = float(coef[j]) if use_n else 0.0
    return dict(c0=c0, ct=ct, cnu=cnu, t0=t0, ts=ts, n0=n0, ns=ns,
                used=used, n_used=int(used.sum()), n_pts=int(k.size))


def _plane_predict(coef, t, nu):
    """Evaluate the fitted plane. t, nu broadcast against each other."""
    import numpy as np
    return (coef["c0"]
            + coef["ct"] * (np.asarray(t, float) - coef["t0"]) / coef["ts"]
            + coef["cnu"] * (np.asarray(nu, float) - coef["n0"]) / coef["ns"])


def _angsep_deg(ra1, dec1, ra2, dec2):
    """Great-circle angular separation [deg]; inputs in degrees."""
    import numpy as np
    r1, d1, r2, d2 = map(np.deg2rad, (ra1, dec1, ra2, dec2))
    cos = np.sin(d1) * np.sin(d2) + np.cos(d1) * np.cos(d2) * np.cos(r1 - r2)
    return np.rad2deg(np.arccos(np.clip(cos, -1.0, 1.0)))


def _tle_proximity_flag(ms_path, norad_ids, spacetrack_path, radius_deg):
    """(n_times,) bool flag where any satellite is within radius_deg of Fornax A,
    plus the (n_times,) times in MJD. Propagates TLEs (needs Space-Track)."""
    import numpy as np
    from riflow.io.ms import read_times, read_ants_itrf
    from riflow.coords import get_tles, sat_radec
    from tabsim.jax.coordinates import mjd_to_jd  # noqa: F401 (needed for get_tles)

    times_mjd = read_times(ms_path)
    obs_xyz   = np.mean(read_ants_itrf(ms_path), axis=0)
    tles, valid = get_tles(spacetrack_path, norad_ids, float(np.mean(times_mjd)))
    fornax_ra, fornax_dec = 51.75, -37.2  # matches riflow.coords.get_fornax_radec

    flag = np.zeros(times_mjd.size, bool)
    min_sep = np.full(times_mjd.size, np.inf)
    for tle in tles:
        rd  = sat_radec(tle, times_mjd, obs_xyz)          # (2, n_times) deg
        sep = _angsep_deg(rd[0], rd[1], fornax_ra, fornax_dec)
        flag |= sep < radius_deg
        min_sep = np.minimum(min_sep, sep)
    return flag, times_mjd, valid, min_sep


def main():
    p = argparse.ArgumentParser(
        description="Array-wide flux calibration k(t, ν) from Fornax A.",
    )
    p.add_argument("-c", "--config_path", required=True, help="Config YAML path.")
    p.add_argument("-d", "--data_type", default="real",
                   help="Config data-type key whose lc_data Fornax flux is used "
                        "to derive the calibration (default: real).")
    p.add_argument("-ms", "--ms_path", default=None, help="Override MS path.")
    p.add_argument("-i", "--image_path", default=None, help="Override image dir.")
    p.add_argument("--lc-dir", default=None,
                   help="Directory with *_light_curves.npz (default: derived from "
                        "config like nufft-gif).")
    p.add_argument("-tsx", "--tab_suffix", default=None, help="TABASCAL suffix.")
    p.add_argument("-img", "--img_suffix", default=None, help="Image param suffix.")
    p.add_argument("-g", "--gif_suffix", default=None, help="Extra GIF suffix.")
    p.add_argument("-td", "--tab_data", default="map", help="Zarr result type.")

    # --- Fornax A spectral model ---
    p.add_argument("--fornax-flux", type=float, default=750.0,
                   help="Fornax A flux [Jy] at --fornax-freq (default 750, APPROX).")
    p.add_argument("--fornax-freq", type=float, default=154.0,
                   help="Reference frequency [MHz] for --fornax-flux (default 154).")
    p.add_argument("--fornax-index", type=float, default=-0.77,
                   help="Spectral index alpha, S ∝ nu^alpha (default -0.77).")
    p.add_argument("--fornax-logpoly", default=None,
                   help="Comma-separated log-polynomial coeffs c0,c1,c2,... in "
                        "log10(S)=Σ c_i (log10(nu/ref))^i, overrides flux/index.")

    # --- smooth solution model ---
    p.add_argument("--time-order", type=int, default=1, choices=[0, 1],
                   help="Polynomial order of k in time (0=constant, 1=linear; default 1).")
    p.add_argument("--freq-order", type=int, default=1, choices=[0, 1],
                   help="Polynomial order of k in frequency (0=constant, 1=linear; default 1).")
    p.add_argument("--fit-nsigma", type=float, default=5.0,
                   help="Sigma-clip threshold for the robust plane fit (default 5).")

    # --- contamination screening ---
    p.add_argument("--stat", default="maxabs", choices=["maxabs", "std"],
                   help="Which aperture statistic to use (default: maxabs).")
    p.add_argument("--sat-factor", type=float, default=1.5,
                   help="Flag a sample if any satellite aperture |flux| exceeds this × "
                        "the channel Fornax median (sidelobe/overlap risk). 0 disables.")
    p.add_argument("--spike-nsigma", type=float, default=5.0,
                   help="Flag Fornax samples above median + N·MAD per channel (default 5).")
    p.add_argument("--min-clean", type=int, default=10,
                   help="Min clean time samples for a channel to count as usable (default 10).")
    p.add_argument("--min-good-frac", type=float, default=0.5,
                   help="Abort as uncalibratable if fewer than this fraction of channels "
                        "are usable (default 0.5).")
    p.add_argument("--allow-uncalibratable", action="store_true", default=False,
                   help="Warn instead of aborting when the data looks uncalibratable.")

    # --- TLE proximity flagging ---
    p.add_argument("--tle-flag", action="store_true", default=False,
                   help="Flag times where a satellite is within --tle-radius-deg of "
                        "Fornax A (propagates TLEs; needs Space-Track + norad_ids).")
    p.add_argument("--tle-radius-deg", type=float, default=5.0,
                   help="Angular radius [deg] for TLE proximity flagging (default 5).")
    p.add_argument("-st", "--spacetrack-path", default=None,
                   help="Space-Track login YAML (default: config gif.spacetrack_path).")
    p.add_argument("-n", "--norad-ids", default=None,
                   help="Extra NORAD IDs (comma-separated) for TLE flagging.")

    # --- application ---
    p.add_argument("--apply", action="store_true", default=False,
                   help="Write a flux-scaled column back to the MS.")
    p.add_argument("--source-col", default=None,
                   help="MS column to scale when applying (default: data_type's column).")
    p.add_argument("--out-col", default=None,
                   help="Output column name (default: <SOURCE_COL>_FLUXCAL).")
    p.add_argument("--chunk", type=int, default=50000,
                   help="Row chunk size when applying (default 50000).")
    p.add_argument("-dr", "--dry-run", action="store_true", default=False,
                   help="Compute and report only; never modify the MS.")
    args = p.parse_args()

    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from riflow import load_config
    from riflow.config import prepend_suffix

    config  = load_config(args.config_path)
    cfg_dir = os.path.dirname(os.path.abspath(args.config_path))

    def _resolve(path):
        return path if os.path.isabs(path) else os.path.join(cfg_dir, path)

    image_path = _resolve(args.image_path or config["data"]["image_path"])
    ms_path    = _resolve(args.ms_path or config["data"]["ms_path"])

    tab_suffix = prepend_suffix(args.tab_suffix)
    img_suffix = prepend_suffix(args.img_suffix)
    gif_suffix = prepend_suffix(args.gif_suffix)
    tab_data_suffix = prepend_suffix(args.tab_data) if tab_suffix else ""
    gif_base   = f"{args.data_type}{tab_suffix}{img_suffix}{gif_suffix}{tab_data_suffix}"

    if args.lc_dir:
        lc_dir = _resolve(args.lc_dir)
    else:
        gifs_root = os.path.join(image_path, "gifs")
        tab_dir   = args.tab_suffix or ""
        save_path = os.path.join(gifs_root, tab_dir, args.data_type) if tab_dir \
                    else os.path.join(gifs_root, args.data_type)
        lc_dir    = os.path.join(save_path, "lc_data")
    save_path = os.path.dirname(lc_dir)

    # --- Fornax A model ---
    if args.fornax_logpoly:
        logpoly = [float(x) for x in args.fornax_logpoly.split(",")]
    else:
        logpoly = [np.log10(args.fornax_flux), args.fornax_index]
    print(f"Fornax A model: log10(S) = {logpoly} about nu_ref={args.fornax_freq} MHz")
    print("  NOTE: default model is APPROXIMATE — verify against a literature "
          "spectrum (e.g. McKinley et al. 2015/2020) for your band.")

    # --- gather the (n_chan, n_times) Fornax + satellite light-curve cube ---
    files = sorted(glob.glob(os.path.join(lc_dir, f"{gif_base}_ch*_light_curves.npz")))
    if not files:
        files = sorted(glob.glob(os.path.join(lc_dir, f"{gif_base}_*_light_curves.npz")))
    if not files:
        raise SystemExit(f"No light-curve npz found in {lc_dir} for base '{gif_base}'. "
                         f"Run `nufft-gif -mo perchan -d {args.data_type}` first.")

    stat_idx = {"std": 0, "maxabs": 1}[args.stat]

    def _is_satellite(name):
        n = name.lower()
        return not any(t in n for t in ("fornax", "centre", "center",
                                        "shifted", "pass", "phase"))

    FOR, SAT, freqs, ch_lo, ch_hi = [], [], [], [], []
    times_sec = None
    for f in files:
        d = np.load(f, allow_pickle=True)
        if "titles" not in d.files or "chan_freq_mhz" not in d.files:
            raise SystemExit(
                f"{os.path.basename(f)} lacks metadata (titles/chan_freq_mhz). "
                f"Re-run nufft-gif or backfill-lc-metadata first.")
        titles = [str(t) for t in d["titles"]]
        if "Fornax A" not in titles:
            raise SystemExit(f"{os.path.basename(f)}: no 'Fornax A' in titles {titles}.")
        lcs       = d["light_curves"][:, :, stat_idx]   # (n_sources, n_times)
        fornax_lc = lcs[titles.index("Fornax A")]
        sat_idx   = [i for i, t in enumerate(titles) if _is_satellite(t)]
        sat_lc    = (np.nanmax(np.abs(lcs[sat_idx]), axis=0) if sat_idx
                     else np.full_like(fornax_lc, np.nan))
        cf = np.atleast_1d(d["chan_freq_mhz"])
        ci = np.atleast_1d(d["chan_idx"])
        FOR.append(fornax_lc); SAT.append(sat_lc)
        freqs.append(float(np.mean(cf))); ch_lo.append(int(ci.min())); ch_hi.append(int(ci.max()))
        if times_sec is None:
            times_sec = np.asarray(d["times_sec"], float)

    order  = np.argsort(freqs)
    FOR    = np.array(FOR)[order]          # (n_chan, n_times)
    SAT    = np.array(SAT)[order]
    freqs  = np.array(freqs)[order]
    ch_lo  = np.array(ch_lo)[order]
    ch_hi  = np.array(ch_hi)[order]
    n_chan, n_times = FOR.shape
    model_jy = _fornax_model_jy(freqs, logpoly, args.fornax_freq)

    # --- optional TLE proximity time flag (geometry: same for all channels) ---
    tle_flag = np.zeros(n_times, bool)
    if args.tle_flag:
        spacetrack = args.spacetrack_path or config["gif"].get("spacetrack_path")
        if not spacetrack:
            raise SystemExit("--tle-flag needs a Space-Track login (gif.spacetrack_path "
                             "in config or --spacetrack-path).")
        norad = [int(x) for x in config["gif"].get("norad_ids", [])]
        if args.norad_ids:
            norad += [int(x) for x in args.norad_ids.split(",")]
        norad = list(dict.fromkeys(norad))
        print(f"\nTLE proximity flag: satellites within {args.tle_radius_deg}° of "
              f"Fornax A | NORAD={norad}")
        tflag, times_mjd, valid, min_sep = _tle_proximity_flag(
            ms_path, norad, _resolve(spacetrack), args.tle_radius_deg)
        if tflag.size != n_times:
            print(f"  WARNING: MS has {tflag.size} times but lc_data has {n_times}; "
                  f"skipping TLE flag (re-run nufft-gif so they match).")
        else:
            tle_flag = tflag
            print(f"  flagged {tle_flag.sum()}/{n_times} times "
                  f"(closest satellite approach: {np.nanmin(min_sep):.2f}°)")

    # --- per-channel screening -> clean (n_chan, n_times) mask ---
    clean      = np.zeros((n_chan, n_times), bool)
    fornax_med = np.full(n_chan, np.nan)
    sat_level  = np.zeros(n_chan)
    n_sat      = np.zeros(n_chan, int)
    n_spike    = np.zeros(n_chan, int)
    for i in range(n_chan):
        cl, sf, spf, fmed, slev = _channel_flags(
            FOR[i], SAT[i], args.sat_factor, args.spike_nsigma)
        clean[i] = cl & ~tle_flag
        fornax_med[i] = fmed
        sat_level[i]  = slev
        n_sat[i]      = int(sf.sum())
        n_spike[i]    = int((spf & ~sf).sum())
    n_clean = clean.sum(axis=1)
    good    = n_clean >= args.min_clean      # channels with enough clean samples

    # --- robust plane fit k(t, ν) over all clean samples ---
    with np.errstate(divide="ignore", invalid="ignore"):
        K = model_jy[:, None] / FOR          # (n_chan, n_times)
    fit_mask = clean & np.isfinite(K) & good[:, None]
    ci, ti   = np.where(fit_mask)
    if ci.size < 4:
        raise SystemExit(f"Only {ci.size} clean samples — cannot fit a calibration.")
    coef = _robust_plane(times_sec[ti], freqs[ci], K[ci, ti],
                         args.time_order, args.freq_order, args.fit_nsigma)

    # per-channel summary scale from the clean Fornax median
    with np.errstate(divide="ignore", invalid="ignore"):
        scale_ch = model_jy / fornax_med

    print(f"\nPer-channel summary ({good.sum()}/{n_chan} channels usable):")
    for i in range(n_chan):
        note = "" if good[i] else f"  SKIP (clean {n_clean[i]} < {args.min_clean})"
        fm = f"{fornax_med[i]:8.3g}" if np.isfinite(fornax_med[i]) else "     nan"
        print(f"  ch{ch_lo[i]:>3}-{ch_hi[i]:<3} {freqs[i]:8.3f} MHz | "
              f"Fornax_med={fm} | k≈{scale_ch[i]:7.4g} | clean={n_clean[i]:3d} "
              f"sat={n_sat[i]:3d} spike={n_spike[i]:3d}{note}")

    # fitted solution summary
    k_mid   = coef["c0"]
    dk_time = 2 * coef["ct"] / k_mid if k_mid else 0.0   # full-range fractional change
    dk_freq = 2 * coef["cnu"] / k_mid if k_mid else 0.0
    resid   = K[ci, ti] - _plane_predict(coef, times_sec[ti], freqs[ci])
    _, rsig = _mad_sigma(resid[coef["used"]])
    print(f"\nSmooth k(t, ν) fit  (time-order={args.time_order}, freq-order={args.freq_order}):")
    print(f"  k(centre)           = {k_mid:.4g}")
    print(f"  change over band    = {dk_freq*100:+.1f}%  ({freqs.min():.3f}–{freqs.max():.3f} MHz)")
    print(f"  change over track   = {dk_time*100:+.1f}%  (0–{times_sec.max():.0f} s)")
    print(f"  fit used {coef['n_used']}/{coef['n_pts']} clean samples "
          f"(σ-clipped {coef['n_pts']-coef['n_used']}); residual scatter "
          f"{rsig/k_mid*100:.1f}% of k")

    # uncalibratable: too few usable channels, or satellites dominate Fornax
    good_frac = good.sum() / max(n_chan, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        dominance = np.nanmedian((sat_level / fornax_med)[good]) if good.any() else np.inf
    print(f"\nUsable channels: {good.sum()}/{n_chan} ({good_frac:.0%}) | "
          f"median satellite/Fornax level on usable channels: {dominance:.2f}")
    if (good_frac < args.min_good_frac) or (np.isfinite(dominance) and dominance > 1.0):
        msg = (f"UNCALIBRATABLE: {good.sum()}/{n_chan} channels usable "
               f"({good_frac:.0%}; need ≥{args.min_good_frac:.0%}) and satellites sit "
               f"at ~{dominance:.1f}× Fornax — satellites likely dominate the band.")
        if args.allow_uncalibratable:
            print("WARNING: " + msg + "\n  --allow-uncalibratable set: continuing anyway.")
        else:
            raise SystemExit("\n" + msg + "\n  (use --allow-uncalibratable to force, or "
                             "loosen --min-good-frac / --sat-factor.)")

    # --- save solution + diagnostic plot ---
    os.makedirs(save_path, exist_ok=True)
    scale_npz = os.path.join(save_path, f"{gif_base}_flux_scale.npz")
    np.savez(scale_npz, freq_mhz=freqs, chan_lo=ch_lo, chan_hi=ch_hi,
             fornax_med=fornax_med, model_jy=model_jy, scale_ch=scale_ch, good=good,
             n_clean=n_clean, n_sat=n_sat, n_spike=n_spike, sat_level=sat_level,
             times_sec=times_sec, tle_flag=tle_flag,
             coef_c0=coef["c0"], coef_ct=coef["ct"], coef_cnu=coef["cnu"],
             coef_t0=coef["t0"], coef_ts=coef["ts"], coef_n0=coef["n0"], coef_ns=coef["ns"],
             time_order=args.time_order, freq_order=args.freq_order,
             ref_freq_mhz=args.fornax_freq, logpoly=np.array(logpoly))
    print(f"\nSolution saved: {scale_npz}")

    t_mid = float(np.median(times_sec))
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    # (0,0) apparent Fornax median vs freq + model
    a = ax[0, 0]
    a.plot(freqs[good], fornax_med[good], "o", label="Fornax median (clean)")
    a.plot(freqs[~good], fornax_med[~good], "rx", ms=8, label="unusable channel")
    a.set_yscale("log"); a.set_xlabel("Frequency [MHz]")
    a.set_ylabel("apparent flux [data units]"); a.legend(); a.grid(True, alpha=0.3)
    a.set_title("Fornax A apparent flux")
    # (0,1) k vs freq: per-channel + fitted plane at t_mid
    a = ax[0, 1]
    a.plot(freqs[good], scale_ch[good], "o", label="per-channel k")
    a.plot(freqs[~good], scale_ch[~good], "rx", ms=8, label="unusable")
    a.plot(freqs, _plane_predict(coef, t_mid, freqs), "-", color="C2",
           label="fit k(ν, t=mid)")
    a.set_xlabel("Frequency [MHz]"); a.set_ylabel("scale k"); a.legend(); a.grid(True, alpha=0.3)
    a.set_title("Calibration vs frequency")
    # (1,0) k vs time: median over channels + fitted plane at nu_mid
    a = ax[1, 0]
    with np.errstate(invalid="ignore"):
        Kt = np.where(clean & good[:, None], K, np.nan)
        kt_med = np.nanmedian(Kt, axis=0)
    a.plot(times_sec, kt_med, ".", ms=3, color="0.5", label="median k over chans")
    if tle_flag.any():
        a.plot(times_sec[tle_flag], kt_med[tle_flag], "x", color="orange",
               ms=6, label="TLE-flagged time")
    a.plot(times_sec, _plane_predict(coef, times_sec, float(np.median(freqs))), "-",
           color="C2", label="fit k(t, ν=mid)")
    a.set_xlabel("Time [s]"); a.set_ylabel("scale k"); a.legend(); a.grid(True, alpha=0.3)
    a.set_title("Calibration vs time")
    # (1,1) residuals of the fit
    a = ax[1, 1]
    a.hist((resid[coef["used"]] / k_mid) * 100, bins=40, color="C0")
    a.set_xlabel("fit residual [% of k]"); a.set_ylabel("clean samples")
    a.set_title(f"Fit residuals (σ={rsig/k_mid*100:.1f}%)"); a.grid(True, alpha=0.3)
    fig.suptitle(f"Fornax A flux calibration — {gif_base}", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plot_path = os.path.join(save_path, f"{gif_base}_flux_scale.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"Diagnostic plot saved: {plot_path}")

    if not args.apply:
        print("\n(--apply not set: MS not modified.)")
        return

    # --- apply: scale a chosen MS column into a new column using k(t, ν) ---
    source_col = args.source_col or config[args.data_type]["data_col"]
    out_col    = args.out_col or f"{source_col}_FLUXCAL"

    from casacore.tables import table, maketabdesc, makecoldesc

    with table(ms_path + "::SPECTRAL_WINDOW", ack=False) as spw:
        all_freq_mhz = spw.getcol("CHAN_FREQ")[0] / 1e6
    k_band = _plane_predict(coef, t_mid, all_freq_mhz)
    print(f"\nApply: {source_col} × k(t, ν) → {out_col} in {ms_path}")
    print(f"  band scale @t=mid: [{k_band.min():.4g}, {k_band.max():.4g}] over "
          f"{all_freq_mhz.size} channels; time order={args.time_order}")
    if (k_band <= 0).any():
        print("  WARNING: fitted k(ν) is non-positive in part of the band — "
              "check the model / contamination.")
    if args.dry_run:
        print("  [dry-run] MS not modified.")
        return

    with table(ms_path, readonly=False, ack=False) as t:
        if source_col not in t.colnames():
            raise SystemExit(f"Source column {source_col} not in MS.")
        if out_col not in t.colnames():
            t.addcols(maketabdesc(makecoldesc(out_col, t.getcoldesc(source_col))))
            print(f"  created column {out_col}")
        times_all = t.getcol("TIME")
        t_ref = float(times_all.min())
        nrows = t.nrows()
        for start in range(0, nrows, args.chunk):
            n = min(args.chunk, nrows - start)
            data = t.getcol(source_col, start, n)            # (n, n_chan, n_pol)
            t_sec = times_all[start:start + n] - t_ref
            k = _plane_predict(coef, t_sec[:, None], all_freq_mhz[None, :])  # (n, n_chan)
            data = data * k[:, :, None].astype(data.real.dtype)
            t.putcol(out_col, data, start, n)
        print(f"  wrote {nrows} rows to {out_col}")
    print("Done.")


if __name__ == "__main__":
    main()
