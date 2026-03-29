import argparse
import shutil
import yaml
import sys
import os
import subprocess
from datetime import datetime

# from riflow.extraction.source_extraction import extract
from riflow.flagging.flag_data import write_perfect_flags, run_aoflagger, flag_zeros
from riflow import Tee, load_config


def main():
    parser = argparse.ArgumentParser(
        description="Process a simulation file that has potentially had tabascal run on it."
    )
    parser.add_argument(
        "-c",
        "--config_path",
        required=True,
        help="File path to the source extraction config file.",
    )
    parser.add_argument(
        "-s", "--sim_dir", help="Path to the directory of the simulation."
    )
    parser.add_argument(
        "-t",
        "--tab_data",
        default="map",
        help="Type of tabascal data to copy over. Default is map. Options are {'map', 'init'}",
    )
    parser.add_argument(
        "-d",
        "--data",
        default="perfect,ideal,tab,flag1,flag2",
        help="The data types to analyse. {'perfect', 'ideal', 'tab', 'flag1', 'flag2'}",
    )
    parser.add_argument(
        "-p",
        "--processes",
        default="image",
        help="The types of processing to do. Default is 'image'. Options are {'image', 'extract', 'pow_spec'}",
    )
    parser.add_argument(
        "-b",
        "--bash_exec",
        default="/bin/bash",
        help="Path to the bash exectuable used to run docker. Default is /bin/bash.",
    )
    parser.add_argument(
        "-sp",
        "--sif_path",
        default=None,
        help="Singularity image path if using singularity.",
    )
    parser.add_argument("-isx", "--im_suffix", default=None, help="Image name suffix.")
    parser.add_argument("-tsx", "--tab_suffix", default=None, help="Image name suffix.")
    parser.add_argument(
        "-r",
        "--recopy",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Recopy data from tab zarr files to MS file before imaging.",
    )
    args = parser.parse_args()
    bash = args.bash_exec
    sim_dir = args.sim_dir
    sif_path = args.sif_path
    im_suffix = args.im_suffix
    tab_suffix = args.tab_suffix
    recopy = args.recopy

    im_suffix, tab_suffix = [
        "_" + suffix if suffix else "" for suffix in [im_suffix, tab_suffix]
    ]

    run_id = datetime.now().strftime("%m-%d-%YT%H:%M:%S")

    log_path = f"log_extract_{im_suffix}{tab_suffix}{run_id}.txt"
    log = open(log_path, "w")
    backup = sys.stdout
    sys.stdout = Tee(sys.stdout, log)

    config = load_config(args.config_path)
    model_name = "Custom"

    if sim_dir is not None:
        sim_dir = os.path.abspath(sim_dir)
        config["data"]["sim_dir"] = sim_dir
    elif config["data"]["sim_dir"] is not None:
        sim_dir = os.path.abspath(config["data"]["sim_dir"])
        config["data"]["sim_dir"] = sim_dir
    else:
        raise KeyError(
            "'sim_dir' must be specified in either the config file or as a command line argument."
        )

    tab_path = os.path.join(
        sim_dir, f"results/{args.tab_data}_pred_{model_name}{tab_suffix}.zarr"
    )

    if sif_path is not None:
        sif_path = os.path.abspath(sif_path)
        config["image"]["sif_path"] = sif_path
    elif config["image"]["sif_path"] is not None:
        sif_path = config["image"]["sif_path"]

    if sim_dir[-1] == "/":
        sim_dir = sim_dir[:-1]

    f_name = os.path.split(sim_dir)[1]
    zarr_path = os.path.join(sim_dir, f"{f_name}.zarr")
    ms_path = os.path.join(sim_dir, f"{f_name}.ms")
    img_dir = os.path.join(sim_dir, "images")

    os.makedirs(img_dir, exist_ok=True)

    # print(config)
    # sys.exit(0)

    print()
    print(f"Working on {ms_path}")

    data = args.data.lower().split(",")
    procs = args.processes.lower().split(",")

    if sif_path is not None:
        singularity = f" -s {sif_path}"
    else:
        singularity = ""

    for key in data:

        data_col = config[key]["data_col"]
        flag_type = config[key]["flag"]["type"]

        if "image" in procs:
            print(
                "\n\n================================================================================"
            )
            print()
            print(f"Flagging {data_col} column of the MS file.")

            if flag_type in ["perfect", "zeros", None]:
                thresh = config[key]["flag"]["thresh"]
                name = f"{thresh:.1f}sigma{im_suffix}{tab_suffix}"
                if flag_type == "perfect":
                    write_perfect_flags(ms_path, thresh)
                elif flag_type == "zeros":
                    flag_zeros(ms_path, config["data"]["data_col"])
                else:
                    print("No flagging done.")

                if "tab" in key and recopy:

                    subprocess.run(
                        f"tab2MS -m {ms_path} -z {tab_path} -d {config['data']['data_col']}",
                        shell=True,
                        executable="/bin/bash",
                    )
                    # write_results(ms_path, tab_path, config["data"]["data_col"])
                # else:
                #     name = f"{thresh:.1f}sigma{im_suffix}"

                reflagged = True

            elif flag_type == "aoflagger":
                rerun_aoflagger = False
                flags, reflagged = run_aoflagger(
                    ms_path,
                    data_col,
                    config[key]["flag"]["strategies"],
                    config[key]["flag"]["sif_path"],
                    rerun_aoflagger=rerun_aoflagger,
                )
                name = f"aoflagger{im_suffix}"
            else:
                raise ValueError(
                    "Incorrect flagging type chosen. Must be one of {perfect, aoflagger}."
                )

            if reflagged:
                wsclean_opts = "".join(
                    [f" -{k} {v}" for k, v in config["image"]["params"].items()]
                )
                img_cmd = f"image{singularity} -m {ms_path} -d {data_col} -n {name} -w '{wsclean_opts}'"
                print()
                print(f"Imaging {data_col} column of the MS file.\nUsing {img_cmd}")
                subprocess.run(img_cmd, shell=True, executable=bash)

        else:
            reflagged = False

        # if "extract" in procs and reflagged:

        #     if flag_type == "aoflagger":
        #         name = f"aoflagger{im_suffix}"
        #     elif flag_type == "perfect":
        #         if key == "tab":
        #             name = f"{thresh:.1f}sigma{im_suffix}{tab_suffix}"
        #         else:
        #             name = f"{thresh:.1f}sigma{im_suffix}"
        #         thresh = config[key]["flag"]["thresh"]
        #     else:
        #         print(
        #             "Incorrect flagging type chosen. Must be one of {perfect, aoflagger}."
        #         )

        #     img_path = os.path.join(img_dir, f"{data_col}_{name}-image.fits")
        #     print()
        #     print(f"Extracting sources from {img_path}")
        #     extract(
        #         img_path,
        #         zarr_path,
        #         config["extract"]["sigma_cut"],
        #         config["extract"]["beam_cut"],
        #         config["extract"]["thresh_isl"],
        #         config["extract"]["thresh_pix"],
        #     )

    log.close()
    shutil.copy(log_path, sim_dir)
    os.remove(log_path)
    sys.stdout = backup

    with open(
        os.path.join(img_dir, f"extract_config{im_suffix}{tab_suffix}.yaml"), "w"
    ) as fp:
        yaml.dump(config, fp)


if __name__ == "__main__":
    main()
