import numpy as np
from numpy.typing import NDArray

from astropy.io import fits


def get_centre_radec(fits_fps: list[str]) -> NDArray:

    ra = np.array([fits.getheader(fp)["CRVAL1"] for fp in fits_fps])
    dec = np.array([fits.getheader(fp)["CRVAL2"] for fp in fits_fps])

    # radec is shape (1, 2, len(fits_fps))
    radec = np.stack([ra, dec], axis=0)[None, :, :]

    return radec
