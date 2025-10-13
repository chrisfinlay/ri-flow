import numpy as np

import os

from argparse import Namespace

from typing import Union


def replace(config_value: str, cli_value: str) -> str:

    if cli_value:
        return cli_value
    else:
        return config_value


def replace_float(
    config_value: Union[None, str, float],
    cli_value: Union[None, str, float],
    error_msg: str = "None value given in both config and at runtime.",
) -> float:

    if cli_value:
        return float(cli_value)
    elif config_value:
        return float(config_value)
    else:
        raise IOError(error_msg)


def replace_path(
    config_value: Union[None, str],
    cli_value: Union[None, str],
    required: bool = False,
    error_msg: str = "A path must be given in either the config file or at runtime.",
) -> Union[None, str]:

    if cli_value:
        return os.path.abspath(cli_value)
    elif config_value:
        return os.path.abspath(config_value)
    elif required:
        raise IOError(error_msg)
    else:
        return None


def sanitise_norad_ids(config_nids: list, cli_nids: str) -> list[int]:

    if cli_nids:
        norad_ids = [int(x) for x in cli_nids.split(",")]
        config_nids.append(norad_ids)

    config_nids = [int(x) for x in np.unique(config_nids).astype(int)]

    return config_nids


def process_freq_chan(freq_chan: str) -> list[str]:

    if freq_chan:
        if freq_chan.upper() == "MFS":
            channels = ["-MFS-"]
        else:
            freq_chans = [int(x) for x in freq_chan.split(" ")]

            if len(freq_chans) == 1:
                channels = [f"-{freq_chans[0]:04}-"]

            elif len(freq_chans) == 2:
                freq_chans[1] += 1
                channels = [f"-{int(chan):04}-" for chan in range(*freq_chans)]
            else:
                raise IOError(
                    f"Too many frequency channel indices given for freq_chan. Expected 1, 2 or MFS but got {len(freq_chan)} indices."
                )
    else:
        channels = [""]

    return channels


def prepend_suffix(suffix: Union[None, str], prepend: str = "_") -> str:

    new_suffix = "_" + suffix if suffix else ""

    return new_suffix


def sanitise_gif_config(config: dict, cli_args: Namespace) -> dict:

    # Fix Image directory path
    config["data"]["image_path"] = replace_path(
        config["data"]["image_path"], cli_args.image_path, required=True
    )
    # Fix MS path
    config["data"]["ms_path"] = replace_path(
        config["data"]["ms_path"], cli_args.ms_path, required=True
    )

    # Fix SpaceTrack path
    config["gif"]["spacetrack_path"] = replace_path(
        config["gif"]["spacetrack_path"], cli_args.spacetrack_path
    )

    # Fix NORAD IDs
    config["gif"]["norad_ids"] = sanitise_norad_ids(
        config["gif"]["norad_ids"], cli_args.norad_ids
    )

    # Fix marker radius
    config["gif"]["marker_radius_deg"] = replace_float(
        config["gif"]["marker_radius_deg"], cli_args.radius_deg
    )

    # Fix frequency channel selection
    config["gif"]["freq_chan"] = replace(config["gif"]["freq_chan"], cli_args.freq_chan)
    config["gif"]["channels"] = process_freq_chan(config["gif"]["freq_chan"])

    return config
