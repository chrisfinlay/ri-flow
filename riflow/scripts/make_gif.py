import numpy as np

from riflow.imaging.gif import make_gif
from riflow.extraction.light_curves import (
    get_all_radecs,
    shift_light_curves,
    plot_light_curves,
)
from riflow import load_config
from riflow.config import sanitise_gif_config, prepend_suffix

from glob import glob
import os

import argparse


def main():

    parser = argparse.ArgumentParser(description="Create a GIF from many FITS files.")
    parser.add_argument(
        "-c",
        "--config_path",
        required=True,
        help="Path to config file.",
    )
    parser.add_argument(
        "-ms",
        "--ms_path",
        help="MS file path.",
    )
    parser.add_argument(
        "-i",
        "--image_path",
        help="Path to directory containing input FITS files.",
    )
    parser.add_argument(
        "-d", "--data_types", default="tab_rfi_res", help="Data type to image."
    )
    parser.add_argument(
        "-r",
        "--radius_deg",
        default=3.0,
        type=float,
        help="Circle radius for overplots. Default is 3 degrees.",
    )
    parser.add_argument(
        "-n",
        "--norad_ids",
        help="Norad IDs",
    )
    parser.add_argument(
        "-st",
        "--spacetrack_path",
        help="Path to space-track login details YAML file.",
    )
    parser.add_argument(
        "-f",
        "--freq_chan",
        help="Frequency channel number. Default is None. If None, this is associated with imaging where channels-out was not given. This is equivalent to MFS when channels-out was defined.",
    )
    parser.add_argument(
        "--diff",
        default=False,
        help="Whether to time difference the images when making GIF.",
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument("-tsx", "--tab_suffix", help="TABASCAL solution suffix.")
    parser.add_argument("-isx", "--im_suffix", help="Image suffix.")
    parser.add_argument("-g", "--gif_suffix", help="Suffix for GIF.")
    args = parser.parse_args()
    config = load_config(args.config_path)
    config = sanitise_gif_config(config, args)

    tab_suffix = prepend_suffix(args.tab_suffix)
    im_suffix = prepend_suffix(args.im_suffix)
    gif_suffix = prepend_suffix(args.gif_suffix)
    diff_str = "_diff" if args.diff else ""

    types = args.data_types.split(",")

    save_path = os.path.join(config["data"]["image_path"], "gifs")
    os.makedirs(save_path, exist_ok=True)

    names = {
        key: config[key]["data_col"]
        for key in config.keys()
        if key not in ["data", "image", "extract", "gif"]
    }

    for data_type in types:
        for channel in config["gif"]["channels"]:

            chan_name = f"_chan_{channel[1:-1]}" if channel else ""
            # chan_name = "_chan_MFS"

            assert (
                data_type in names.keys()
            ), f"{data_type} not defined in config file with path {args.config_path}"

            # Get sorted list of FITS files
            fits_search = os.path.join(
                config["data"]["image_path"],
                f"{names[data_type]}_0.0sigma{tab_suffix}-t*{channel}dirty.fits",
                # f"{names[data_type]}_0.0sigma{im_suffix}{tab_suffix}-t*-MFS-dirty.fits",
            )
            # thresh = config[data_type]["flag"]["thresh"]
            # name = f"{thresh:.1f}sigma{im_suffix}{tab_suffix}"
            fits_fps = sorted(glob(fits_search))
            if len(fits_fps) == 0:
                raise IOError(f"No fits files found with search path: {fits_search}")

            gif_name = (
                f"{data_type}{diff_str}{im_suffix}{tab_suffix}{chan_name}{gif_suffix}"
            )
            save_name = os.path.join(save_path, gif_name)

            times, radec, titles, shift = get_all_radecs(
                config["data"]["ms_path"],
                fits_fps,
                config["gif"]["spacetrack_path"],
                config["gif"]["norad_ids"],
                # frame_shift=0.5,
            )

            light_curves = make_gif(
                fits_fps,
                f"{save_name}.gif",
                radec,
                titles,
                config["gif"]["marker_radius_deg"],
                diff=args.diff,
            )
            if args.diff:
                times = times[1:]
            light_curves = shift_light_curves(light_curves, titles, shift)

            np.save(
                os.path.join(save_path, f"{gif_name}_light_curves.npy"), light_curves
            )

            plot_light_curves(times, light_curves, titles, f"{save_name}.png")


if __name__ == "__main__":

    main()
