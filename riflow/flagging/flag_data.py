from daskms import xds_from_ms, xds_to_table

import xarray as xr
import dask.array as da
import dask
import os

import argparse
import subprocess
import shutil

from typing import Optional

# aoflagger -strategy ../yaml_configs/target/firstpass.rfis tab1_obs_16A_450T-0440-1338_256I_001F-1.227e+09-1.227e+09_100PAST_000GAST_000EAST_1SAT_0GRD_1.0e+00RFI/tab1_obs_16A_450T-0440-1338_256I_001F-1.227e+09-1.227e+09_100PAST_000GAST_000EAST_1SAT_0GRD_1.0e+00RFI.ms/


def flag_zeros(ms_path: str, data_col: str):

    xds = xds_from_ms(ms_path)[0]  # type: ignore

    flags = xds[data_col] == 0.0  # type: ignore

    xds = xds.assign(FLAG=flags)  # type: ignore
    cols = ["FLAG"]

    dask.compute(xds_to_table([xds], ms_path, cols))  # type: ignore

    print()
    print(f"Flag Rate      : {100*flags.data.mean().compute(): .1f} %")

    return flags


def flag_ants(ms_path: str, bad_ants):

    xds = xds_from_ms(ms_path)[0]  # type: ignore

    a1 = xds["ANTENNA1"].data  # type: ignore
    a2 = xds["ANTENNA2"].data  # type: ignore

    flags = da.isin(a1, bad_ants) | da.isin(a2, bad_ants)  # type: ignore

    xds = xds.assign(FLAG=flags)  # type: ignore
    cols = ["FLAG"]

    dask.compute(xds_to_table([xds], ms_path, cols))  # type: ignore

    print()
    print(f"Flag Rate      : {100*flags.data.mean().compute(): .1f} %")

    return flags


def write_perfect_flags(ms_path: str, n_sigma: float = 3.0):

    xds = xds_from_ms(ms_path)[0]  # type: ignore

    if n_sigma > 0:
        flags = da.abs(xds.CAL_DATA - xds.AST_MODEL_DATA) > (  # type: ignore
            n_sigma * xds.SIGMA * da.sqrt(2)  # type: ignore
        )
    else:
        flags = xr.zeros_like(xds.DATA).astype(bool)  # type: ignore

    xds = xds.assign(FLAG=flags)  # type: ignore
    cols = ["FLAG"]

    dask.compute(xds_to_table([xds], ms_path, cols))  # type: ignore

    print()
    print(f"Flag Threshold : {n_sigma: .1f} sigma")
    print(f"Flag Rate      : {100*flags.data.mean().compute(): .1f} %")

    return flags


def write_to_aoflags(ms_path: str):

    xds_ms = xds_from_ms(ms_path)[0]  # type: ignore

    xds_ms = xds_ms.assign({"AO_FLAGS": xds_ms["FLAG"]})  # type: ignore
    cols = ["AO_FLAGS"]

    print("Writing AOFlagger flags to 'AO_FLAGS' column in MS file.")

    dask.compute(xds_to_table([xds_ms], ms_path, cols))  # type: ignore


def write_aoflags_to_flag(ms_path: str):

    xds_ms = xds_from_ms(ms_path)[0]  # type: ignore

    xds_ms = xds_ms.assign(FLAG=xds_ms["AO_FLAGS"])  # type: ignore
    cols = ["FLAG"]

    print("Writing AOFlagger flags to 'FLAG' column in MS file.")

    dask.compute(xds_to_table([xds_ms], ms_path, cols))  # type: ignore


def run_aoflagger(
    ms_path: str,
    data_column: str = "CAL_DATA",
    strategy_paths: Optional[list] = None,
    sif_path: Optional[str] = None,
    bash_exec: str = "/bin/bash",
    rerun_aoflagger: bool = False,
):

    reflagged = False

    if ms_path[-1] == "/":
        ms_path = ms_path[:-1]

    if not rerun_aoflagger:
        xds_ms = xds_from_ms(ms_path)
        try:
            aoflags = xds_ms["AO_FLAGS"]  # type: ignore
            write_aoflags_to_flag(ms_path)
            print("Using previous AOFlagger run.")
        except:
            aoflags = None
    else:
        aoflags = None

    if not aoflags or rerun_aoflagger:

        data_dir, ms_file = os.path.split(os.path.abspath(ms_path))

        docker_opts = "--rm -v /etc/group:/etc/group -v /etc/passwd:/etc/passwd -v /etc/shadow:/etc/shadow -v/etc/sudoers.d:/etc/sudoers.d -e HOME=${HOME} --user=`id -ur`"
        container_cmd = f"docker run {docker_opts} -v {data_dir}:/data --workdir /data stimela/aoflagger:latest"

        if sif_path is not None:
            sif_path = os.path.abspath(sif_path)
            container_cmd = (
                f"singularity exec --bind {data_dir}:/data --pwd /data {sif_path}"
            )

        if strategy_paths is not None:
            write_perfect_flags(ms_path, 0)
            for strategy_path in strategy_paths:

                strategy_path = os.path.abspath(strategy_path)
                shutil.copy(strategy_path, data_dir)
                strategy_file = os.path.split(strategy_path)[1]
                strategy = f"-strategy /data/{strategy_file}"

                aoflag_cmd = f"{container_cmd} aoflagger -column {data_column} {strategy} /data/{ms_file}"
                subprocess.run(aoflag_cmd, shell=True, executable=bash_exec)

            print()
            print(
                f"Strategies : {[os.path.split(strategy)[1] for strategy in strategy_paths]}"
            )

        else:
            aoflag_cmd = f"{container_cmd} aoflagger /data/{ms_file}"
            subprocess.run(aoflag_cmd, shell=True, executable=bash_exec)
            print()

        write_to_aoflags(ms_path)
        reflagged = True

    flags = xds_from_ms(ms_path)[0].FLAG.data  # type: ignore
    print(f"Flag Rate      : {100*flags.mean().compute(): .1f} %")

    return flags, reflagged


def main():

    parser = argparse.ArgumentParser(
        description="Flag CAL_DATA lying a certain threshold away from the AST_MODEL_DATA. Requires tabascal simulated MS files."
    )
    parser.add_argument(
        "-m", "--ms_path", required=True, help="File path to the Measurement Set."
    )
    parser.add_argument(
        "-s",
        "--n_sigma",
        default=3.0,
        type=float,
        help="Threshold in number of std of noise given by SIGMA column. 0 unflags everything. Default is 3.0.",
    )
    parser.add_argument(
        "-d",
        "--data_col",
        default="DATA",
        help="Data column to run AOFlagger on. Default is 'DATA'.",
    )
    parser.add_argument(
        "-ao",
        "--aoflagger",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Whether to use AOFlagger to flag the data. Default is False.",
    )
    parser.add_argument(
        "-rao",
        "--rerun_aoflagger",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Whether to rerun AOFlagger to flag the data. Default is False.",
    )
    parser.add_argument(
        "-st",
        "--strategy_paths",
        default=None,
        help="list of paths to AOFlagger strategies.",
    )
    parser.add_argument(
        "-sp",
        "--sif_path",
        default=None,
        help="Paths to AOFlagger singularity image.",
    )
    parser.add_argument(
        "-z",
        "--zeros",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Whether to flag where visibilities are zeros. Default is False.",
    )
    args = parser.parse_args()
    ms_path = args.ms_path
    n_sigma = args.n_sigma
    strategy_paths = args.strategy_paths
    data_col = args.data_col

    if args.aoflagger:
        if strategy_paths is not None:
            strategy_paths = strategy_paths.split(",")
        flags = run_aoflagger(
            ms_path, data_col, strategy_paths, args.sif_path, args.rerun_aoflagger
        )
    elif args.zeros:
        flags = flag_zeros(ms_path, data_col)
    else:
        flags = write_perfect_flags(ms_path, n_sigma)


if __name__ == "__main__":
    main()
