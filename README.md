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
