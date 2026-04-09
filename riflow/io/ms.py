import os
import numpy as np
from numpy.typing import NDArray
from daskms import xds_from_ms, xds_from_table


def recopy_tab_results(
    ms_path: str,
    tab_suffix: str,
    data_col: str,
    sim_dir: str | None = None,
    tab_data: str = "map",
    model_name: str = "Custom",
) -> None:
    """Copy TABASCAL results from the matching zarr file into the MS.

    Calls ``tabascal.write.write_results_ms`` directly, which writes
    TAB_AST_DATA, TAB_RFI_DATA, TAB_AST_RES, TAB_RFI_RES, TAB_RES_DATA
    columns into the MS.

    The zarr file is expected at::

        <sim_dir>/results/<tab_data>_pred_<model_name><tab_suffix>.zarr

    Parameters
    ----------
    ms_path    : absolute path to the Measurement Set
    tab_suffix : TABASCAL run suffix including leading underscore
                 (e.g. ``'_sgp4_bstar_var'``).  Must be non-empty.
    data_col   : observed data column used as reference by write_results_ms
                 (e.g. ``'REAL_DATA'``).
    sim_dir    : simulation directory.  Inferred from ms_path parent if None.
    tab_data   : zarr result type — ``'map'`` (default) or ``'init'``.
    model_name : model name in the zarr filename, default ``'Custom'``.

    Raises
    ------
    ValueError        : if tab_suffix is empty.
    FileNotFoundError : if the zarr path does not exist.
    """
    from tabascal.write import write_results_ms

    if not tab_suffix:
        raise ValueError(
            "--recopy requires --tab_suffix (-tsx) so the correct zarr file "
            "can be identified."
        )

    if sim_dir is None:
        sim_dir = os.path.dirname(ms_path.rstrip("/"))

    zarr_path = os.path.join(
        os.path.abspath(sim_dir),
        f"results/{tab_data}_pred_{model_name}{tab_suffix}.zarr",
    )

    if not os.path.exists(zarr_path):
        raise FileNotFoundError(
            f"TABASCAL zarr not found: {zarr_path}\n"
            f"Check tab_suffix ('{tab_suffix}'), tab_data ('{tab_data}'), "
            f"and model_name ('{model_name}') are correct."
        )

    print(f"  Copying zarr → MS columns: {zarr_path}")
    write_results_ms(ms_path=ms_path, results_zarr_path=zarr_path, data_col=data_col)


def read_ants_itrf(ms_path: str) -> NDArray:

    xds_ant = xds_from_table(ms_path + "::ANTENNA")[0]  # type: ignore

    itrf = xds_ant.POSITION.data.compute()  # type: ignore

    return itrf


def read_times(ms_path: str) -> NDArray:

    xds_ms = xds_from_ms(ms_path)[0]  # type: ignore

    times_mjd = np.unique(xds_ms.TIME.data.compute()) / (24 * 3600)  # type: ignore

    return times_mjd
