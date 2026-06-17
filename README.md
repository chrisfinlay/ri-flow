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
| `<base>_perchan_lc_grid_chan_X-Y.png` | perchan | Grid of light curve subplots, one per channel (default). |
| `<base>_spectrogram_std_chan_X-Y.png` | perchan | Time–frequency spectrogram (std statistic) with noise indicator. |
| `<base>_spectrogram_maxabs_chan_X-Y.png` | perchan | Time–frequency spectrogram (max\|flux\| statistic) with noise indicator. |
| `<base>_spectrum_chan_X-Y.png` | perchan | Frequency spectrum (mean and max over time) with noise indicator. |
| `lc_data/<base>_light_curves.npz` | both | NumPy archive: `light_curves (n_sources, n_times, 2)` — columns are `[std, max\|flux\|]` — and scalar `noise_level`. |

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
