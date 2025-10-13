import numpy as np
from numpy.typing import NDArray
from astropy.coordinates import EarthLocation
from skyfield.api import EarthSatellite, wgs84, load

from tabsim.config import yaml_load
from tabsim.tle import get_tles_by_id
from tabsim.jax.coordinates import mjd_to_jd

from riflow.io.ms import read_ants_itrf, read_times


def get_fornax_radec(n_times: int) -> NDArray:

    radec_fornax = np.array([51.75, -37.2])[None, :, None] * np.ones((1, 1, n_times))

    return radec_fornax


def sat_radec(tle: list[str], times_mjd: np.ndarray, obs_xyz: np.ndarray) -> np.ndarray:

    ts = load.timescale()
    t = ts.ut1_jd(mjd_to_jd(times_mjd))

    satellite = EarthSatellite(tle[0], tle[1], ts=ts)
    location = EarthLocation(x=obs_xyz[0], y=obs_xyz[1], z=obs_xyz[2], unit="m")
    observer = wgs84.latlon(
        location.lat.degree, location.lon.degree, location.height.value
    )
    radec = (satellite - observer).at(t).radec()

    return np.stack([radec[0]._degrees, radec[1].degrees], axis=0)  # type: ignore


def get_tles(
    spacetrack_path: str, norad_ids: list[int], epoch_mjd: float
) -> tuple[NDArray, list[int]]:

    st_login = yaml_load(spacetrack_path)

    tles_df = get_tles_by_id(
        st_login["username"],
        st_login["password"],
        norad_ids,
        mjd_to_jd(epoch_mjd),
        # tle_dir="./tles",
    )
    tles = np.atleast_2d(tles_df[["TLE_LINE1", "TLE_LINE2"]].values)
    norad_ids = [int(nid) for nid in tles_df.NORAD_CAT_ID.values]

    return tles, norad_ids


def get_sat_radec_from_ms(
    ms_path: str, spacetrack_path: str, norad_ids: list[int]
) -> tuple[NDArray, list[int]]:

    ants_itrf = read_ants_itrf(ms_path)
    times_mjd = read_times(ms_path)

    tles, norad_ids = get_tles(spacetrack_path, norad_ids, float(np.mean(times_mjd)))

    # radec is shape (len(norad_ids), 2, len(times_mjd))
    radec = np.stack(
        [sat_radec(tle, times_mjd, np.mean(ants_itrf, axis=0)) for tle in tles], axis=0
    )

    return radec, norad_ids
