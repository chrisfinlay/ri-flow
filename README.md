# ri-flow

Workflow pipeline for radio interferometry with specific application to TABASCAL analysis.

---

## Installation

```bash
pip install -e ".[sat]"
```

Dependencies: `jax-finufft`, `tabsim`, `dask-ms`, `xarray`, `zarr<3.0.0`, `tqdm`, `matplotlib`, `astropy`, `numpy`, `regions` (sat extra).

---

## Scripts

### `nufft-gif`

Produces dirty-image GIFs, time-integrated image plots, light curve plots, and spectrograms directly from a Measurement Set using a JAX-based NUFFT. No WSClean, no intermediate FITS files.

**Usage**

```bash
nufft-gif -c <config.yaml> [options]
```

**Required**

| Flag | Description |
|------|-------------|
| `-c / --config_path` | Path to the imaging config YAML (see format below). |

**Data selection**

| Flag | Default | Description |
|------|---------|-------------|
| `-d / --data_types` | `tab_rfi` | Comma-separated data column keys from the config (e.g. `real`, `tab_rfi,tab_ast`). |
| `-ms / --ms_path` | from config | Override the MS path. |
| `-ch / --channels` | all | Numpy-style channel selection: `':'`, `'0:16'`, `'::2'`, `'0,4,8'`. |
| `-tr / --time-range` | all | Numpy-style time index selection for the integrated image: `'0:50'`, `'10:60'`. |

**Imaging**

| Flag | Default | Description |
|------|---------|-------------|
| `-mo / --mode` | `mfs` | `mfs` — all selected channels combined per timestep into one GIF; `perchan` — one GIF per channel, saved in a dedicated subdirectory. |
| `-ws / --wstack` | off | Enable w-stacking correction for wide-field imaging. |
| `-nw / --n_wplanes` | auto | Number of w-stacking planes (estimated from data if not set). |
| `-img / --img_suffix` | — | Appended to output filenames to distinguish imaging parameter sets. |

**Output control**

| Flag | Default | Description |
|------|---------|-------------|
| `-i / --image_path` | from config | Override output directory. Outputs go into `<image_path>/gifs/`. |
| `-g / --gif_suffix` | — | Extra string appended to all output filenames. |
| `-ng / --no-gif` | off | Skip GIF rendering; still compute dirty images for aperture statistics, light curves, spectrograms, and integrated image plots. |
| `-pcl / --per-chan-lc` | off | In `perchan` mode, save one light curve PNG per channel instead of the default single-grid figure. |
| `-rp / --replot` | off | Skip the MS read and imaging entirely; regenerate the light curve, spectrogram, and spectrum plots from the previously saved `lc_data/*.npz`. Fast iteration on plot styling. Use the same `-d`/`-mo` and suffix flags as the original run so the right files are found. |
| `-nbl / --noise-band-level` | `0.95` | Central level for the per-timestep noise bands drawn on the light curves / grid / spectrum (see [Noise nulls](#noise-nulls-on-the-light-curves) below). `0.95` → 2.5–97.5 percentile band. |

**Satellite tracking**

| Flag | Default | Description |
|------|---------|-------------|
| `-n / --norad_ids` | from config | Extra NORAD IDs to track (comma-separated), appended to any IDs in the config. |
| `-st / --spacetrack_path` | from config | Path to Space-Track login credentials YAML. |
| `-r / --radius_deg` | from config | Aperture radius in degrees for light curve extraction and image overlays. |
| `-fs / --frame-shift` | — | Add a time-shifted copy of each satellite track as a sky-background reference. Value in `(-1, 1)` is a fraction of the total track length (e.g. `0.1` shifts by 10 % of timesteps). The extracted light curve is rolled back into alignment after extraction. |

**TABASCAL integration**

| Flag | Default | Description |
|------|---------|-------------|
| `-tsx / --tab_suffix` | — | TABASCAL solution suffix (e.g. `sgp4_bstar_var`). |
| `-td / --tab_data` | `map` | Zarr result type: `map` or `init`. |
| `-mn / --model_name` | `Custom` | TABASCAL model name encoded in the zarr filename. |
| `-rc / --recopy` | off | Re-copy TABASCAL results from the zarr into the MS before imaging. Requires `-tsx`. |
| `-dc / --data-col` | from config | Override the reference data column used by `-rc` when forming residuals from the zarr. **Must match the scale the run was fit on** — pass `REAL_DATA_FLUXCAL` when recopying a flux-calibrated zarr, otherwise the residual is `data − k·models` (the k-scaled models appear as if unsubtracted). |

**Config YAML format**

Each top-level key (other than `image` and `gif`) defines a named data column:

```yaml
real:
  data_col: REAL_DATA
  flag:
    type: null      # null | "threshold"
    thresh: 0

tab_rfi:
  data_col: TAB_RFI_DATA
  flag:
    type: null
    thresh: 0

image:
  params:
    size: 256 256
    scale: 30amin   # pixel scale: NNamin / NNasec / NNadeg
    niter: 0
    pol: xx
    weight: briggs -0.5
    intervals-out: 77

gif:
  spacetrack_path: ../../spacetrack_login.yaml
  norad_ids: [60441]
```

**Output files** (written to `<image_path>/gifs/`)

| File | Mode | Description |
|------|------|-------------|
| `<base>_mfs-chan_X-Y_tidx_X-Y_image.png` | MFS | Time-integrated dirty image with Fornax aperture circle (cyan) and satellite track dots (red). |
| `<base>_mfs-chan_X-Y.gif` | MFS | Dirty-image GIF, one frame per timestep. |
| `<base>_mfs-chan_X-Y.png` | MFS | Light curve plot (std and max\|flux\|). |
| `<base>_perchan_image_grid_chan_X-Y_tidx_X-Y.png` | perchan | Grid of time-integrated images, one per channel, with source overlays. |
| `<base>_chan_X-Y_tidx_X-Y_gifs/<base>_chN.gif` | perchan | Per-channel GIFs in a dedicated subdirectory. |
| `<base>_perchan_lc_grid_{std,maxabs}_chan_X-Y.png` | perchan | Grid of light curve subplots, one per channel, per-row y-scale (default). |
| `<base>_spectrogram_{std,maxabs}_chan_X-Y.png` | perchan | Time–frequency spectrogram (std and max\|flux\| statistics); colorbar marks `E[noise]` (cyan dashed) and the upper band (white dotted). |
| `<base>_spectrum_{std,maxabs}_chan_X-Y.png` | perchan | Frequency spectrum (mean and max over time, NaN-safe) with the `E[noise]` line + shaded band vs frequency. |
| `lc_data/<base>_light_curves.npz` | both | NumPy archive: `light_curves (n_sources, n_times, 2)` (columns `[std, max\|flux\|]`), `noise_level`, the per-timestep noise nulls `sigma_t`, `exp_max_t`, `max_lo_t`, `max_hi_t`, `std_exp_t`, `std_lo_t`, `std_hi_t`, the `gumbel_loc`/`gumbel_scale` and `band_level`, plus `times_sec`, `titles`, `chan_freq_mhz`, `chan_idx` so the file is self-contained for `--replot` and `flux-calibrate`. |

#### Noise nulls on the light curves

The light curves measure, in each source aperture, the `std` and the
`max|flux|` per timestep. Even on a pure-noise residual those are non-zero — the
`max|flux|` over a multi-beam aperture follows extreme-value statistics — so to
judge whether a satellite is subtracted to the noise floor each plot overlays a
**per-timestep noise null** computed from the image itself:

- **`σ(t)`** — the background per-pixel noise, estimated each timestep from
  off-source pixels inside the horizon (excluding every source aperture and any
  `gif.avoid_sources`). Compare it against the `std` curve.
- **`E[max|flux|]` + band** — control apertures of the same radius are scattered
  over the off-source background; their `max|flux|`/`σ` is pooled across all
  timesteps and fitted with a single Gumbel (an extreme-value distribution). The
  expected max and a central `--noise-band-level` band (default 95 %) then scale
  per timestep as `σ(t) × (dimensionless Gumbel statistic)`. The `std` curve gets
  an analogous band from the empirical `std`/`σ` quantiles.

**Reading it.** A `max|flux|` (or `std`) curve that stays inside its band is
consistent with noise — the source is subtracted to the floor, and the
suppression can only be quoted as a *lower limit* `S_before / E[max]`. Excursions
above the band are candidate residual signal at that flux. The same nulls drive
the perchan grid (per-channel band + `E[noise]` line), the spectrogram (colorbar
markers) and the spectrum (band vs frequency; note its *max-over-time* line
naturally sits above a per-timestep band, so compare the *mean-over-time* line).
These pair naturally with flux-calibrated columns so the floor is in Jy — image
a `*_FLUXCAL`-fit residual via `-rc -dc REAL_DATA_FLUXCAL`.

---

### `vis-analysis`

Gridded UV-plane diagnostics straight from a Measurement Set — no imaging. For each
data column it grids the complex visibilities onto the UV plane (`|mean(vis)|` and
`std(vis)` per cell, conjugate points included) and produces three plot types: one
panel per channel, one panel per time chunk, and a single full-coverage two-panel
figure.

**Usage**

```bash
vis-analysis -c <config.yaml> -d real -ch 0:27 --uv-chunks 25
```

**Required**

| Flag | Description |
|------|-------------|
| `-c / --config_path` | Imaging config YAML (same format as `nufft-gif`; named data columns + `image`). |

**Options**

| Flag | Default | Description |
|------|---------|-------------|
| `-d / --data_types` | `real` | Comma-separated data column keys from the config (e.g. `real,tab_rfi`). |
| `-ms / --ms_path` | from config | Override the MS path. |
| `-i / --image_path` | from config | Override output directory. Outputs go into `<image_path>/vis_analysis/`. |
| `-ch / --channels` | all | Numpy-style channel selection: `':'`, `'0:16'`, `'::2'`, `'0,4,8'`. |
| `-tr / --time-range` | all | Numpy-style time index selection. |
| `-uvc / --uv-chunks` | `25` | Number of time chunks for the per-chunk UV grid. |
| `-ng / --n-grid` | `256` | UV grid resolution along each axis. |
| `-sx / --suffix` | — | Extra suffix appended to all output filenames. |
| `-tsx / --tab_suffix` | — | TABASCAL solution suffix (e.g. `sgp4_bstar_var`). |
| `-td / --tab_data` | `map` | Zarr result type: `map` or `init`. |
| `-mn / --model_name` | `Custom` | TABASCAL model name encoded in the zarr filename. |
| `-rc / --recopy` | off | Re-copy TABASCAL results from the zarr into the MS first. Requires `-tsx`. |

**Output files** (written to `<image_path>/vis_analysis/<data_type>/`)

| File | Description |
|------|-------------|
| `<base>_uv_perchan_{abs,std}_chan_X-Y_tidx_X-Y.png` | One panel per channel, all times: `|mean(vis)|` (`abs`) and `std(vis)` (`std`). |
| `<base>_uv_timechunksN_{abs,std}_chan_X-Y_tidx_X-Y.png` | One panel per time chunk (all channels combined). |
| `<base>_uv_full_chan_X-Y_tidx_X-Y.png` | Single two-panel figure (`|mean(vis)|` and `std(vis)`) over all selected times and channels. |

---

### `uv-redundancy`

UV-plane **redundancy / imageability** diagnostic. A sky image can predict only one
visibility per UV cell, so the best it can do in a cell is the mean of the samples
there; the scatter *about that mean*, normalised by the thermal noise, is a reduced
χ² that is ≈1 when the cell is redundant and noise-limited (fully imageable and
calibrated) and >1 when there is non-redundant per-baseline / per-antenna structure
no image can fit (miscalibration, mutual coupling, differing element beams). Binning
is done **within each (time, channel) snapshot** then accumulated, so earth rotation
does not masquerade as non-redundancy. The UV cell is sized to the imaging resolution
`1/(2 sin(fov/2))` so the χ² means "imageability at that field of view".

With `--geom-floor` it also grids each channel to a dirty image and degrids it
(`vis2dirty → dirty2vis`, an image-consistent round-trip whose within-cell χ² is the
pure sub-cell sky-gradient floor), and writes a **non-redundancy map**
`χ²(data) − χ²(round-trip) − noise_floor` — the calibratable excess, which calibration
should drive toward 0. Re-run it on the corrected/residual column written by
successive TABASCAL runs to track that.

**Usage**

```bash
# explicit MS columns (e.g. compare before/after a calibration run)
uv-redundancy -ms path/to.ms --cols REAL_DATA,CORRECTED_DATA -ch 0:27 --geom-floor

# or via a riflow config (column names resolved like vis-analysis)
uv-redundancy -c config.yaml -d real -ch 0:27 --geom-floor
```

**Options**

| Flag | Default | Description |
|------|---------|-------------|
| `-ms / --ms_path` | from config | Measurement Set path (overrides config). |
| `--cols` | — | Comma-separated MS columns to analyse (e.g. `REAL_DATA,CORRECTED_DATA`). Takes precedence over `-d`. |
| `-c / --config_path` | — | Optional riflow config YAML (for `ms_path`/`image_path` and column names). |
| `-d / --data_types` | `real` | Config data-type keys (used when `--cols` is absent). |
| `-i / --image_path` | `<ms dir>` | Output directory. Outputs go into `<image_path>/uv_redundancy/`. |
| `-ch / --channels` | all | Numpy-style channel selection. |
| `--fov-deg` | `180` | Field of view setting the UV cell `= 1/(2 sin(fov/2))` λ. |
| `-ng / --n-grid` | from `--fov-deg` | Explicit UV grid resolution (overrides the fov-derived cell). |
| `--sigma` | median MS `SIGMA` | Per-visibility noise (real/imag std) for the χ² normalisation. |
| `--geom-floor` | off | Also compute the geometric (sub-cell gradient) floor and the non-redundancy map. |
| `--noise-floor` | `1.0` | Within-cell χ² attributable to thermal noise, subtracted in the non-redundancy map. Use `0` for a **noiseless model column** (e.g. `TAB_AST_DATA`/`vis_ast`). |
| `--n-pix` | `180` | Image size for the `--geom-floor` round-trip. |
| `--eps` | `1e-6` | Gridder accuracy for the `--geom-floor` round-trip. |
| `--vmax` | `3.0` | Colour-scale max for the χ² maps. |
| `-sx / --suffix` | — | Extra suffix appended to output filenames (use to avoid clobbering across runs). |

**Output files** (written to `<image_path>/uv_redundancy/`, per column `<base>`)

| File | Mode | Description |
|------|------|-------------|
| `<base>_chi2_combined.png` | always | Within-UV-cell redundancy reduced χ², all channels (white at 1 = redundant, red >1 = non-redundant). Title shows global χ² and imageable power fraction. |
| `<base>_chi2_perchan.png` | always (>1 chan) | Same χ², one panel per channel. |
| `<base>_geomfloor_combined.png` | `--geom-floor` | Geometric (sub-cell gradient) floor χ² from the `vis2dirty → dirty2vis` round-trip. |
| `<base>_nonredund_combined.png` | `--geom-floor` | Calibratable non-redundancy map `χ²(data) − χ²(geom) − noise_floor` (centred at 0: white = calibrated, red = excess). |

Console output prints, per column, the global redundancy χ², imageable power
fraction, and (with `--geom-floor`) the geometric-floor χ² and the non-redundancy χ².

---

### `flux-calibrate`

Simple array-wide flux calibration from **Fornax A** — the one source common to all
EDA2 fields — intended as an initial flux scale before TABASCAL. It derives a smooth,
direction-independent amplitude solution `k(t, ν) = S_model_FornaxA(ν) / S_obs(t, ν)`
from the per-channel Fornax light curves and (with `--apply`) writes a flux-scaled
column back to the MS. It does **not** recover per-antenna gains/phases (the rest of
the field is unmodelled); it only fixes the overall flux scale vs time and frequency.

The solution is modelled as **at most linear in time and frequency** and fit robustly
(iterative σ-clipping), since the underlying gain is expected to be smooth — rapid
variation is treated as noise/artefacts. Fornax samples are screened for satellite
contamination first (out-of-FOV, bright-satellite aperture, Fornax spikes, optional
TLE proximity). If too little survives, or satellites dominate the band, the data is
reported as **uncalibratable**.

**Prerequisite:** run `nufft-gif -mo perchan -d <type>` first so the per-channel
`lc_data/*.npz` (with metadata) exist. The calibration is derived from those, not from
the MS visibilities directly; the MS is read only for channel frequencies, per-row
`TIME`, and the source column at apply time.

**Usage**

```bash
# derive + diagnose only (no MS write); uses the placeholder Fornax model
flux-calibrate -c <config.yaml> -d real

# supply a real Fornax A spectrum and flag times where a satellite nears Fornax
flux-calibrate -c <config.yaml> -d real \
    --fornax-flux 750 --fornax-freq 154 --fornax-index -0.77 --tle-flag

# write REAL_DATA × k(t,ν) -> REAL_DATA_FLUXCAL
flux-calibrate -c <config.yaml> -d real --apply
```

**Fornax A model** (log-polynomial `log10 S = Σ c_i (log10(ν/ν_ref))^i`)

| Flag | Default | Description |
|------|---------|-------------|
| `--fornax-flux` | `750` | Fornax A flux [Jy] at `--fornax-freq`. **Approximate placeholder — replace with a literature spectrum** (e.g. McKinley et al. 2015/2020). |
| `--fornax-freq` | `154` | Reference frequency [MHz] for `--fornax-flux`. |
| `--fornax-index` | `-0.77` | Spectral index α, `S ∝ ν^α`. |
| `--fornax-logpoly` | — | Explicit comma-separated coeffs `c0,c1,c2,...`; overrides flux/index. |

**Smooth solution + screening**

| Flag | Default | Description |
|------|---------|-------------|
| `-ch / --channels` | all | Numpy-style channel selection for the **fit only** (e.g. `3:30`, `0,4,8`). Use to exclude band-edge roll-off the linear model can't capture. The smooth solution is still applied (extrapolated) to **all** channels on `--apply`, so edges get an approximate scale for a later bandpass step or exclusion. |
| `--time-order` | `1` | Order of `k` in time (`0`=constant, `1`=linear). Use `0` if the Fornax level shows a strong (beam) trend in time. |
| `--freq-order` | `1` | Order of `k` in frequency (`0`=constant, `1`=linear). |
| `--fit-nsigma` | `5` | σ-clip threshold for the robust plane fit. |
| `--stat` | `maxabs` | Aperture statistic to use (`maxabs` or `std`). |
| `--sat-factor` | `1.5` | Flag a sample if any satellite aperture \|flux\| exceeds this × the channel Fornax median. `0` disables. |
| `--spike-nsigma` | `5` | Flag Fornax samples above median + N·MAD per channel. |
| `--min-clean` | `10` | Min clean time samples for a channel to count as usable. |
| `--min-good-frac` | `0.5` | Abort as uncalibratable if fewer than this fraction of channels are usable. |
| `--allow-uncalibratable` | off | Warn instead of aborting when the data looks uncalibratable. |

**TLE proximity flagging**

| Flag | Default | Description |
|------|---------|-------------|
| `--tle-flag` | off | Flag times where a satellite is within `--tle-radius-deg` of Fornax A (propagates TLEs; needs Space-Track + NORAD IDs). |
| `--tle-radius-deg` | `5` | Angular radius [deg] for TLE proximity flagging. |
| `-st / --spacetrack-path` | from config | Space-Track login YAML. |
| `-n / --norad-ids` | from config | Extra NORAD IDs (comma-separated). |

**Application**

| Flag | Default | Description |
|------|---------|-------------|
| `--apply` | off | Write the flux-scaled column back to the MS. |
| `--source-col` | data-type's column | MS column to scale (e.g. `REAL_DATA`). |
| `--out-col` | `<SOURCE_COL>_FLUXCAL` | Output column name (created if missing; source column untouched). |
| `--chunk` | `50000` | Row chunk size when applying. |
| `-dr / --dry-run` | off | With `--apply`, report what would be written without modifying the MS. |

Common selection flags (`-d`, `-ms`, `-i`, `--lc-dir`, `-tsx`, `-img`, `-g`, `-td`)
match `nufft-gif` so the right `lc_data` is located.

**Output files** (written next to `lc_data`, i.e. `<image_path>/gifs/[<tab>/]<type>/`)

| File | Description |
|------|-------------|
| `<base>_flux_scale.npz` | Solution archive: `freq_mhz`, `model_jy`, per-channel `fornax_med`/`scale_ch`/`good`/`selected`/`used`/`n_clean`/`n_sat`/`n_spike`, `times_sec`, `tle_flag`, and the plane coefficients (`coef_*`, `time_order`, `freq_order`, `ref_freq_mhz`, `logpoly`). |
| `<base>_flux_scale.png` | 4-panel diagnostic: apparent flux vs ν, `k` vs ν with the fit, `k` vs time with the fit (TLE-flagged times marked), and the fit-residual histogram. |
