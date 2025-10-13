import numpy as np
from numpy.typing import NDArray

import matplotlib.pyplot as plt

from PIL import Image

from astropy.wcs import WCS, FITSFixedWarning
from astropy.io import fits

import warnings
import os

from riflow.extraction.light_curves import get_region_stats


def make_frame(image: NDArray, wcs: WCS, vmin: float, vmax: float):

    # Create figure with WCS projection
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection=wcs)
    im = ax.imshow(image, cmap="gray", vmin=vmin, vmax=vmax, origin="lower")

    # Sky grid
    ax.coords.grid(True, color="blue", ls="dotted")  # type: ignore
    ax.coords[0].set_axislabel("Right Ascension")  # type: ignore
    ax.coords[1].set_axislabel("Declination")  # type: ignore

    # Inset colorbar
    cax = ax.inset_axes((0.87, 0.05, 0.03, 0.4))

    # Create colorbar
    cb = ax.figure.colorbar(im, cax=cax, orientation="vertical")
    cb.ax.tick_params(labelsize=6)

    return fig, ax


def make_gif(
    fits_files: list[str],
    output_gif: str,
    radec: NDArray,
    labels: list[str],
    aperture_radius_deg: float,
    label_offset: float = 0,
):
    # List to store image frames
    frames = []

    # Normalize the data for display
    # data = [fits.getdata(fp)[0, 0] for fp in fits_files]
    data = []
    for fp in fits_files:
        with fits.open(fp) as hdul:
            data.append(hdul[0].data[0, 0])  # type: ignore
    vmin, vmax = np.percentile(data, [0.5, 99.5])

    n_src, _, n_fits = radec.shape
    light_curves = np.empty((n_src, n_fits, 3))

    for f_idx, fits_file in enumerate(fits_files):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FITSFixedWarning)
            with fits.open(fits_file) as hdul:
                hdr = hdul[0].header  # type: ignore
                wcs = WCS(hdr).celestial

        fig, ax = make_frame(data[f_idx], wcs, vmin, vmax)

        # Draw circles and get stats
        for s_idx, coord in enumerate(radec[:, :, f_idx]):
            light_curves[s_idx, f_idx], circ, label_kwargs = get_region_stats(
                data[f_idx],
                wcs,
                coord,
                aperture_radius_deg=aperture_radius_deg,
                draw_circle_with_label=True,
                label=labels[s_idx],
                label_offset=label_offset,
            )

            ax.add_patch(circ)
            ax.text(**label_kwargs)

        # Save to temp image
        temp_path = f"temp_frame_{f_idx:03d}.png"
        plt.savefig(temp_path, bbox_inches="tight", pad_inches=0)
        plt.close(fig)

        frame = Image.open(temp_path).convert("P")
        frames.append(frame)
        os.remove(temp_path)

    # Save as animated GIF
    frames[0].save(
        output_gif,
        save_all=True,
        append_images=frames[1:],
        duration=20,  # milliseconds per frame
        loop=0,  # infinite loop
    )

    print(f"GIF saved as {output_gif}")

    return light_curves
