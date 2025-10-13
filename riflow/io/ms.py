import numpy as np
from numpy.typing import NDArray
from daskms import xds_from_ms, xds_from_table


def read_ants_itrf(ms_path: str) -> NDArray:

    xds_ant = xds_from_table(ms_path + "::ANTENNA")[0]  # type: ignore

    itrf = xds_ant.POSITION.data.compute()  # type: ignore

    return itrf


def read_times(ms_path: str) -> NDArray:

    xds_ms = xds_from_ms(ms_path)[0]  # type: ignore

    times_mjd = np.unique(xds_ms.TIME.data.compute()) / (24 * 3600)  # type: ignore

    return times_mjd
