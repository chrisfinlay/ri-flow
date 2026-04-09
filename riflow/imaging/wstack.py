"""
wstack.py - W-stacking correction for wide-field NUFFT imaging.

The standard 2-D dirty image assumes a flat sky (w ≈ 0).  For wide-field
observations the full measurement equation includes a w-term:

    V(u,v,w) = ∫∫ I(l,m)/n · exp(-2πi(ul + vm + w(n-1))) dl dm

where n = sqrt(1 - l² - m²) and (l,m) are direction cosines.

W-stacking (Offringa et al. 2014, MNRAS 444) removes this error by:
  1. Grouping visibilities into N_w bins by their w-coordinate.
  2. Computing a 2-D NUFFT dirty image per bin.
  3. Multiplying each bin image by the w-phase correction exp(2πi w_k (n-1))
     in image space.
  4. Summing the corrected images.

The computational cost grows linearly with N_w.  A sensible default can be
estimated from the data with `estimate_n_wplanes`.
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp
from jax_finufft import nufft1


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def estimate_n_wplanes(
    w_wl: np.ndarray,
    n_pix: int,
    pixel_rad: float,
    phase_accuracy: float = 1.0,
) -> int:
    """Estimate the number of w-planes needed for a given phase accuracy.

    Within a bin of width Δw the representative w_k differs from the true
    w by at most Δw/2, giving a worst-case phase error of:

        error = 2π · (Δw/2) · |n - 1|_max = π · Δw · |n - 1|_max

    Requiring error < phase_accuracy gives:

        Δw < phase_accuracy / (π · |n - 1|_max)
        N_w = ceil( w_range / Δw )

    |n - 1|_max is the exact maximum of |sqrt(1 - l² - m²) - 1| over the
    valid (above-horizon) region of the image.  For narrow fields this
    reduces to the familiar small-angle approximation ≈ θ²; for ultra-wide
    fields (EDA2, MWA all-sky mode) the image extends past the horizon and
    |n - 1|_max → 1.

    EDA2 example (0.5° pixels, 256 px, 35 m baselines, 160 MHz):
      half-FoV  = 128 × 0.00873 = 1.117 rad  → image corner is below horizon
      |n-1|_max = 1  (reaches horizon within the image)
      w_range   ≈ 37.4 λ  →  N_w ≈ 118 planes

    Parameters
    ----------
    w_wl : array of w-coordinates in wavelengths
    n_pix : image size (pixels along one side)
    pixel_rad : pixel scale in radians
    phase_accuracy : maximum allowed phase error in radians (default 1.0)

    Returns
    -------
    n_wplanes : int  (at least 1)
    """
    # Worst-case |n-1| at the image corner, clipped to the valid sky (lm² < 1)
    half_fov = (n_pix / 2.0) * pixel_rad        # half image width in l/m units
    lm2_corner = 2.0 * half_fov ** 2            # l=m=half_fov at corner
    lm2_max    = min(lm2_corner, 1.0 - 1e-12)   # can't exceed horizon
    n_min      = np.sqrt(1.0 - lm2_max)
    max_n_minus_1 = 1.0 - n_min                 # always in (0, 1]

    delta_w_max = phase_accuracy / (np.pi * max_n_minus_1)
    w_range = float(np.max(w_wl) - np.min(w_wl))
    n = max(1, int(np.ceil(w_range / delta_w_max)))
    return n


def lm_grid(n_pix: int, pixel_rad: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (l, m) direction-cosine grids matching the NUFFT image layout.

    Convention (matches the existing nufft_gif.py WCS):
      rows  → Dec / m:  m[i, j] = (i - n_pix//2) · pixel_rad
      cols  → RA  / l:  l[i, j] = -(j - n_pix//2) · pixel_rad  (RA decreases left→right)

    Returns arrays of shape (n_pix, n_pix).
    """
    centre = n_pix // 2
    idx = np.arange(n_pix, dtype=np.float64) - centre
    m = idx[:, None] * pixel_rad          # (n_pix, 1) broadcast → (n_pix, n_pix)
    l = -(idx[None, :] * pixel_rad)       # (1, n_pix) broadcast
    m = np.broadcast_to(m, (n_pix, n_pix)).copy()
    l = np.broadcast_to(l, (n_pix, n_pix)).copy()
    return l, m


def w_correction_image(
    w_plane: float,
    n_pix: int,
    pixel_rad: float,
    l: np.ndarray | None = None,
    m: np.ndarray | None = None,
) -> np.ndarray:
    """Complex w-correction kernel in image space.

    Returns exp(2πi · w_plane · (n - 1)) of shape (n_pix, n_pix),
    where n = sqrt(1 - l² - m²).

    Pixels outside the unit circle (l² + m² > 1) are set to zero.

    Parameters
    ----------
    w_plane : representative w value for this stack (wavelengths)
    n_pix   : image size
    pixel_rad : pixel scale in radians
    l, m    : pre-computed direction-cosine grids (optional, recomputed if None)
    """
    if l is None or m is None:
        l, m = lm_grid(n_pix, pixel_rad)

    lm2 = l ** 2 + m ** 2
    valid = lm2 < 1.0
    n_img = np.where(valid, np.sqrt(np.where(valid, 1.0 - lm2, 0.0)), 0.0)
    phase = np.where(valid, 2.0 * np.pi * w_plane * (n_img - 1.0), 0.0)
    return np.exp(1j * phase).astype(np.complex64)


def dirty_image_wstack(
    u_wl: np.ndarray,
    v_wl: np.ndarray,
    w_wl: np.ndarray,
    vis: np.ndarray,
    n_pix: int,
    pixel_rad: float,
    weights: np.ndarray | None = None,
    n_wplanes: int | None = None,
    phase_accuracy: float = 1.0,
) -> np.ndarray:
    """W-stacked dirty image using NUFFT type-1.

    Parameters
    ----------
    u_wl, v_wl, w_wl : baseline coordinates in wavelengths, shape (N,)
    vis               : visibilities, complex, shape (N,)
    n_pix             : image size in pixels
    pixel_rad         : pixel scale in radians
    weights           : imaging weights, shape (N,).  None → natural (all ones).
    n_wplanes         : number of w-stacking planes.  None → auto-estimated.
    phase_accuracy    : maximum phase error (rad) used for auto-estimation.

    Returns
    -------
    image : float32 array of shape (n_pix, n_pix)
    """
    N = len(u_wl)
    if weights is None:
        weights = np.ones(N, dtype=np.float32)

    if n_wplanes is None:
        n_wplanes = estimate_n_wplanes(w_wl, n_pix, pixel_rad, phase_accuracy)

    # Pre-compute direction-cosine grids once (shared across all w-planes)
    l, m = lm_grid(n_pix, pixel_rad)

    # Bin boundaries and representative w-values
    w_min, w_max = float(w_wl.min()), float(w_wl.max())
    if w_max == w_min:
        # Degenerate: single w-plane, fall back to standard 2-D imaging
        w_planes = [w_min]
        bins = [np.ones(N, dtype=bool)]
    else:
        edges = np.linspace(w_min, w_max, n_wplanes + 1)
        w_planes = 0.5 * (edges[:-1] + edges[1:])
        bins = []
        for k in range(n_wplanes):
            if k < n_wplanes - 1:
                mask = (w_wl >= edges[k]) & (w_wl < edges[k + 1])
            else:
                mask = (w_wl >= edges[k]) & (w_wl <= edges[k + 1])
            bins.append(mask)

    image_sum = np.zeros((n_pix, n_pix), dtype=np.float64)
    weight_sum = 0.0

    wvis = (vis * weights).astype(np.complex64)

    for w_k, mask in zip(w_planes, bins):
        n_sel = int(mask.sum())
        if n_sel == 0:
            continue

        u_k = u_wl[mask].astype(np.float32)
        v_k = v_wl[mask].astype(np.float32)
        vis_k = wvis[mask]

        # Include Hermitian conjugates so the dirty image is real
        u_all = np.concatenate([u_k, -u_k])
        v_all = np.concatenate([v_k, -v_k])
        vis_all = np.concatenate([vis_k, vis_k.conj()])

        # Scale to NUFFT coordinates in [-π, π]
        u_nufft = jnp.array(2.0 * np.pi * pixel_rad * u_all)
        v_nufft = jnp.array(2.0 * np.pi * pixel_rad * (-v_all))
        vis_jax  = jnp.array(vis_all)

        # 2-D NUFFT for this w-plane
        im_k = np.array(nufft1((n_pix, n_pix), vis_jax, v_nufft, u_nufft))

        # Apply image-domain w-correction: exp(2πi w_k (n-1))
        corr = w_correction_image(w_k, n_pix, pixel_rad, l=l, m=m)
        image_sum += np.real(im_k * corr)

        weight_sum += float(weights[mask].sum())

    if weight_sum == 0.0:
        return np.zeros((n_pix, n_pix), dtype=np.float32)

    # Normalise by total weight × 2 (conjugates), same convention as _dirty_image
    return (image_sum / (2.0 * weight_sum)).astype(np.float32)
