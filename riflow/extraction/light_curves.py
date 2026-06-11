import numpy as np
from numpy.typing import NDArray

import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS, FITSFixedWarning
from astropy.wcs.utils import proj_plane_pixel_scales

from riflow.io.ms import read_times
from riflow.io.fits import get_centre_radec
from riflow.coords import get_fornax_radec, get_sat_radec_from_ms

from typing import Callable, Union, Optional

import warnings

warnings.filterwarnings("ignore", category=FITSFixedWarning)


def extract_light_curve(fits_files, ra, dec, aperture_radius_deg=1.0):
    """
    Extract light curve from FITS images using manual circular aperture photometry.

    Parameters:
    -----------
    fits_files: list[str]
        List of FITS files at each time step.
    ra/dec: NDArray
        RA and Dec in degrees at each time step.
    aperture_radius_deg: float
        aperture radius in degrees.

    Returns:
    --------
    light_curve: NDArray
        Fluxes at each time step.
    """

    stat_funcs = [np.min, np.mean, np.max]

    n_files = len(fits_files)

    light_curve = np.zeros((n_files, 3))

    for i, fits_file in enumerate(fits_files):
        with fits.open(fits_file) as hdul:
            data = hdul[0].data[0, 0]  # type: ignore
            hdr = hdul[0].header  # type: ignore
            wcs = WCS(hdr).celestial

            radec = np.array([ra[i], dec[i]])

            light_curve[i] = get_region_stats(  # type: ignore
                data,
                wcs,
                radec,
                stat_funcs,
                aperture_radius_deg,
            )

    return light_curve


def get_region_stats(
    image: NDArray,
    wcs: WCS,
    radec: NDArray,
    stat_funcs: list[Callable] = [np.min, np.mean, np.max],
    aperture_radius_deg: float = 1.0,
    draw_circle_with_label: bool = False,
    label: str = "",
    label_offset: float = 0,
) -> Union[NDArray, tuple[NDArray, Circle, dict]]:

    # Get pixel scale (deg/pix) and convert aperture radius
    pixel_scales = proj_plane_pixel_scales(wcs)[:2]
    aperture_radius_pix = aperture_radius_deg / np.mean(pixel_scales)

    # Convert RA/Dec to pixel coordinates
    skycoord = SkyCoord(ra=radec[0], dec=radec[1], unit="deg")
    x_centre, y_centre = wcs.world_to_pixel(skycoord)

    x, y = np.arange(image.shape[0]), np.arange(image.shape[1])
    xx, yy = np.meshgrid(x, y)

    idx = np.where(
        aperture_radius_pix >= np.sqrt((xx - x_centre) ** 2 + (yy - y_centre) ** 2)
    )

    if len(idx[0]) > 0:
        stats = np.array([stat(image[idx]) for stat in stat_funcs])
    else:
        stats = np.empty(len(stat_funcs))

    if draw_circle_with_label:
        circ = Circle(
            (x_centre, y_centre),
            aperture_radius_pix,
            edgecolor="red",
            facecolor="none",
            linewidth=1.5,
        )
        label_kwargs = {
            "x": x_centre + 1 * aperture_radius_pix + label_offset,
            "y": y_centre
            + 1 * aperture_radius_pix
            + label_offset,  # Offset to avoid overlapping
            "s": label,  # You can customize this label
            "color": "red",  # Choose contrasting color
            "fontsize": 9,
            "ha": "left",
            "va": "bottom",  # Horizontal and vertical alignment
            # "bbox": {
            # "facecolor": "black", "alpha": 0.5, "pad": 1, "edgecolor": "none"
            # },
        }
        return stats, circ, label_kwargs
    else:
        return stats


def plot_spectrogram(
    times: np.ndarray,
    chan_freqs_mhz: np.ndarray,
    lc_per_chan: np.ndarray,
    titles: list,
    save_name: str,
    stat_idx: int = 1,
    noise_level: float | None = None,
) -> None:
    """Plot a time-frequency spectrogram from per-channel light curves.

    Parameters
    ----------
    times          : (n_times,) time in seconds from observation start
    chan_freqs_mhz : (n_chan,) channel centre frequencies in MHz
    lc_per_chan    : (n_chan, n_sources, n_times, 2) — axis -1 is [std, max_abs]
    titles         : source names, length n_sources
    save_name      : output PNG path
    stat_idx       : 0 → std, 1 → max_abs
    noise_level    : if given, drawn as a horizontal line on each colorbar
    """
    stat_labels = {0: ("std", "Std [Jy/bm]"), 1: ("max|flux|", "max|flux| [Jy/bm]")}
    stat_name, cbar_label = stat_labels[stat_idx]

    n_sources = len(titles)
    data = lc_per_chan[:, :, :, stat_idx]  # (n_chan, n_sources, n_times)

    fig, axes = plt.subplots(n_sources, 1, figsize=(10, 4 * n_sources), squeeze=False)

    for s_idx, ax in enumerate(axes[:, 0]):
        z = data[:, s_idx, :]  # (n_chan, n_times)
        pcm = ax.pcolormesh(times, chan_freqs_mhz, z, shading="nearest", cmap="inferno")
        cb = fig.colorbar(pcm, ax=ax, label=cbar_label)
        if noise_level is not None:
            cb.ax.axhline(noise_level, color="cyan", linewidth=1.5, linestyle="--")
        ax.set_title(f"{titles[s_idx]} — {stat_name}")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Frequency [MHz]")

    plt.tight_layout()
    plt.savefig(save_name, dpi=200, format="png", bbox_inches="tight")
    plt.close(fig)


def plot_spectrum(
    chan_freqs_mhz: np.ndarray,
    lc_per_chan: np.ndarray,
    titles: list,
    save_name: str,
    noise_level: float | None = None,
    stat_idx: int = 1,
) -> None:
    """Plot flux spectrum (frequency vs flux) from per-channel light curves.

    Parameters
    ----------
    chan_freqs_mhz : (n_chan,) channel centre frequencies in MHz
    lc_per_chan    : (n_chan, n_sources, n_times, 2) — axis -1 is [std, max_abs]
    titles         : source names, length n_sources
    save_name      : output PNG path
    noise_level    : if given, drawn as a horizontal line on each subplot
    stat_idx       : 0 → std, 1 → max|flux|
    """
    stat_labels = {0: "Std [Jy/bm]", 1: "max|flux| [Jy/bm]"}
    y_label = stat_labels[stat_idx]

    n_sources = len(titles)
    stat_vals = lc_per_chan[:, :, :, stat_idx]  # (n_chan, n_sources, n_times)

    mean_over_time = stat_vals.mean(axis=2)     # (n_chan, n_sources)
    max_over_time  = stat_vals.max(axis=2)      # (n_chan, n_sources)

    fig, axes = plt.subplots(n_sources, 1, figsize=(8, 4 * n_sources), squeeze=False)

    for s_idx, ax in enumerate(axes[:, 0]):
        ax.plot(chan_freqs_mhz, mean_over_time[:, s_idx], label="mean over time")
        ax.plot(chan_freqs_mhz, max_over_time[:, s_idx],  label="max over time")
        if noise_level is not None:
            ax.axhline(noise_level, color="gray", linewidth=1,
                       linestyle="--", alpha=0.7, label="noise")
        ax.set_title(titles[s_idx])
        ax.set_xlabel("Frequency [MHz]")
        ax.set_ylabel(y_label)
        ax.legend()
        ax.grid()

    plt.tight_layout()
    plt.savefig(save_name, dpi=200, format="png", bbox_inches="tight")
    plt.close(fig)


def plot_perchan_lc_grid(
    times: np.ndarray,
    perchan_lcs: list,
    chan_freqs_mhz: np.ndarray,
    titles: list,
    save_name: str,
    noise_levels: list | None = None,
    stat_idx: int = 1,
) -> None:
    """Grid of light curve subplots, one per frequency channel.

    Layout is roughly square in the number of channels.

    Parameters
    ----------
    stat_idx : 0 → std, 1 → max|flux|
    """
    stat_labels = {0: "Std [Jy/bm]", 1: "max|flux| [Jy/bm]"}
    y_label = stat_labels[stat_idx]

    n_chan = len(perchan_lcs)
    n_cols = int(np.ceil(np.sqrt(n_chan)))
    n_rows = int(np.ceil(n_chan / n_cols))

    prop_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(3.5 * n_cols, 2.5 * n_rows),
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    axes_flat = axes.ravel()

    for i, (lc, freq_mhz) in enumerate(zip(perchan_lcs, chan_freqs_mhz)):
        ax = axes_flat[i]
        for s_idx, title in enumerate(titles):
            color = prop_cycle[s_idx % len(prop_cycle)]
            ax.plot(times, lc[s_idx, :, stat_idx], color=color, linewidth=0.9, label=title)
        if noise_levels is not None:
            lbl = "noise" if i == 0 else None
            ax.axhline(noise_levels[i], color="gray", linewidth=0.7,
                       linestyle="--", alpha=0.7, label=lbl)
        ax.set_title(f"{freq_mhz:.2f} MHz", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=6)

    for j in range(n_chan, len(axes_flat)):
        axes_flat[j].set_visible(False)

    axes_flat[0].legend(fontsize=6, loc="upper right")

    fig.text(0.5, 0.01, "Time [s]", ha="center", fontsize=10)
    fig.text(0.01, 0.5, y_label, va="center", rotation="vertical", fontsize=10)

    plt.tight_layout(rect=[0.03, 0.03, 1, 1])
    plt.savefig(save_name, dpi=150, format="png", bbox_inches="tight")
    plt.close(fig)


def plot_light_curves(
    times: np.ndarray,
    light_curves: np.ndarray,
    titles: list,
    save_name: str,
    noise_level: float | None = None,
):
    plt.rcParams["font.size"] = 16
    n_curves = len(titles)

    # squeeze=False keeps `ax` a 2-D array even when n_curves == 1, so a single
    # source (e.g. Fornax A only) doesn't break subscripting.
    fig, ax = plt.subplots(n_curves, 1, figsize=(8, 6.5 * n_curves),
                           squeeze=False)

    for i in range(n_curves):
        ax[i, 0].plot(times, light_curves[i, :, 0], label="std")
        ax[i, 0].plot(times, light_curves[i, :, 1], label="max|flux|")
        if noise_level is not None:
            ax[i, 0].axhline(noise_level, color="gray", linewidth=1,
                             linestyle="--", alpha=0.7, label="noise")
        ax[i, 0].legend()
        ax[i, 0].set_title(titles[i])
        ax[i, 0].set_ylabel("Flux [Jy/bm]")
        ax[i, 0].set_xlabel("Time [s]")
        ax[i, 0].grid()

    plt.savefig(save_name, dpi=200, format="png", bbox_inches="tight")
    plt.close(fig)


def get_all_radecs(
    ms_path: str,
    fits_fps: list[str],
    spacetrack_path: str,
    norad_ids: list[int],
    include_centre: bool = True,
    include_fornax: bool = True,
    frame_shift: Optional[float] = None,
    sat_pass: Optional[float] = None,
    additonal_locs: Optional[dict] = None,
) -> tuple[NDArray, NDArray, list[str], int]:

    times_mjd = read_times(ms_path)
    times = (times_mjd - times_mjd[0]) * (24 * 3600)
    n_time = len(times_mjd)
    shift = 0

    assert n_time == len(fits_fps), f"Number of time steps ({n_time}) does not equal the number of FITS files ({len(fits_fps)})." 

    if len(norad_ids) > 0:
        radec_sats, norad_ids = get_sat_radec_from_ms(
            ms_path, spacetrack_path, norad_ids
        )
        sat_labels = [str(id) for id in norad_ids]
        all_radecs = [radec_sats]
        titles = sat_labels.copy()

        if frame_shift:
            assert -1 < frame_shift < 1, "Frame shift must be between -1 and 1"
            shift = int(n_time * frame_shift)
            all_radecs.append(np.roll(radec_sats, shift, axis=-1))
            titles += [f"{label} Shifted" for label in sat_labels]

        if sat_pass:
            assert 0 < sat_pass < 1, "Satellite pass must be between 0 and 1"
            pass_idx = int(n_time * sat_pass)
            all_radecs.append(
                radec_sats[:, :, pass_idx, None] * np.ones((1, 2, n_time))
            )
            titles += [f"{label} Pass" for label in sat_labels]

    else:
        all_radecs = []
        titles = []

    if include_centre:
        all_radecs.append(get_centre_radec(fits_fps))
        titles.append("Phase Centre")

    if include_fornax:
        all_radecs.append(get_fornax_radec(n_time))
        titles.append("Fornax A")

    if additonal_locs:
        for label, coord in additonal_locs.items():
            all_radecs.append(np.array(coord)[None, :, None] * np.ones((1, 2, n_time)))
            titles += label

    radec = np.concatenate(all_radecs)

    return times, radec, titles, shift


def shift_light_curves(light_curves: NDArray, titles: list[str], shift: int) -> NDArray:

    assert len(light_curves) == len(
        titles
    ), "Number of light curves and titles do not match."

    for s_idx, title in enumerate(titles):
        if "shifted" in title.lower():
            light_curves[s_idx] = np.roll(light_curves[s_idx], shift, axis=-2)

    return light_curves


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="Modify a MS by adding real data.")
    parser.add_argument(
        "-st",
        "--spacetrack_path",
        required=True,
        help="Path to space-track login details YAML file.",
    )
    parser.add_argument(
        "-f",
        "--fits_path",
        required=True,
        help="Search path with wildcard.",
    )
    parser.add_argument(
        "-s",
        "--save_name",
        required=True,
        help="Save name for the plot.",
    )
    parser.add_argument(
        "-ms",
        "--ms_path",
        required=True,
        help="MS Path of data used to image.",
    )
    parser.add_argument(
        "-r",
        "--radius_deg",
        default=3.0,
        help="Aperture radius in degees.",
    )
    args = parser.parse_args()

    # plot_extracted_light_curves(
    #     args.uvfits_path,
    #     args.fits_path,
    #     args.spacetrack_path,
    #     args.save_name,
    #     args.radius_deg,
    # )
