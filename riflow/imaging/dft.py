"""
dft.py - Exact discrete Fourier transform dirty imager with optional per-baseline
w-correction.

Unlike the NUFFT imager, no interpolation kernel is involved — each visibility
contributes its exact phase to every pixel.  This is mathematically equivalent
to w-projection with perfect kernels, at the cost of O(N_pix² × N_vis) work
per image rather than O(N_pix² log N_pix).

For EDA2 (256×256 image, ~few-thousand baselines per snapshot) this is
typically 2–20× slower than NUFFT per timestep but produces reference-quality
images that are useful for validation and for cases where the NUFFT
interpolation artefacts are unacceptable.

W-correction
------------
When w_wl is supplied, the full measurement phase is applied per baseline:

    φ_j(l, m) = 2π [ u_j · l  +  v_j · m  +  w_j · (n(l,m) - 1) ]

where n(l,m) = sqrt(1 - l² - m²).  No binning or approximation is made —
each baseline uses its own w value exactly.  Pixels at or below the horizon
(l² + m² ≥ 1) are set to zero.

Memory
------
The core operation builds a (pixel_chunk × N_vis) complex64 phase matrix per
chunk.  Reduce `pixel_chunk` if memory is tight; increase it for faster
throughput on a GPU.
"""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

from riflow.imaging.wstack import lm_grid


# ---------------------------------------------------------------------------
# JIT-compiled inner kernels
# ---------------------------------------------------------------------------

@jax.jit
def _sum_chunk_2d(
    l_chunk: jax.Array,
    m_chunk: jax.Array,
    u_j: jax.Array,
    v_j: jax.Array,
    wvis_j: jax.Array,
) -> jax.Array:
    """DFT sum over baselines for a pixel chunk (no w-correction).

    Returns Re{ sum_j wvis_j * exp(2πi(u_j·l + v_j·m)) } of shape (chunk,).
    """
    phase = 2.0 * jnp.pi * (
        l_chunk[:, None] * u_j[None, :]
        + m_chunk[:, None] * v_j[None, :]
    )
    return jnp.real(jnp.sum(wvis_j[None, :] * jnp.exp(1j * phase), axis=-1))


@jax.jit
def _sum_chunk_3d(
    l_chunk: jax.Array,
    m_chunk: jax.Array,
    nm1_chunk: jax.Array,
    u_j: jax.Array,
    v_j: jax.Array,
    w_j: jax.Array,
    wvis_j: jax.Array,
) -> jax.Array:
    """DFT sum over baselines for a pixel chunk with exact w-correction.

    Returns Re{ sum_j wvis_j * exp(2πi(u_j·l + v_j·m + w_j·(n-1))) }
    of shape (chunk,).
    """
    phase = 2.0 * jnp.pi * (
        l_chunk[:, None] * u_j[None, :]
        + m_chunk[:, None] * v_j[None, :]
        + nm1_chunk[:, None] * w_j[None, :]
    )
    return jnp.real(jnp.sum(wvis_j[None, :] * jnp.exp(1j * phase), axis=-1))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def dirty_image_dft(
    u_wl: np.ndarray,
    v_wl: np.ndarray,
    vis: np.ndarray,
    n_pix: int,
    pixel_rad: float,
    weights: np.ndarray | None = None,
    w_wl: np.ndarray | None = None,
    pixel_chunk: int = 4096,
) -> np.ndarray:
    """Compute a dirty image using the exact discrete Fourier transform.

    Parameters
    ----------
    u_wl, v_wl : baseline coordinates in wavelengths, shape (N,)
    vis         : visibilities, complex64, shape (N,)
    n_pix       : image size in pixels (square)
    pixel_rad   : pixel scale in radians
    weights     : imaging weights, shape (N,).  None → natural (all ones).
    w_wl        : w-coordinates in wavelengths, shape (N,).
                  If supplied, the full w-phase is applied per baseline
                  (exact, no binning).  If None, w-terms are ignored.
    pixel_chunk : number of pixels processed per JAX kernel call.
                  Reduce to lower peak memory; increase for GPU throughput.

    Returns
    -------
    image : float32 array of shape (n_pix, n_pix)

    Notes
    -----
    Normalisation is consistent with the NUFFT imager: a unit point source
    at the phase centre produces a peak value of ≈ 1.
    Hermitian conjugate baselines are implicit (only Re{·} is taken).
    """
    N = len(u_wl)
    if weights is None:
        weights = np.ones(N, dtype=np.float32)

    l_grid, m_grid = lm_grid(n_pix, pixel_rad)  # (n_pix, n_pix)

    # Flatten image coordinates for chunk processing
    l_flat = l_grid.ravel().astype(np.float32)    # (n_pix²,)
    m_flat = m_grid.ravel().astype(np.float32)

    # Horizon mask: pixels where l² + m² ≥ 1 are below the horizon
    lm2_flat = (l_flat ** 2 + m_flat ** 2)
    valid_pix = lm2_flat < 1.0

    # Pre-compute n-1 for w-correction (zero at horizon → safe to use always)
    if w_wl is not None:
        nm1_flat = (np.sqrt(np.where(valid_pix, 1.0 - lm2_flat, 1.0)) - 1.0).astype(
            np.float32
        )
        nm1_flat[~valid_pix] = 0.0
        w_j = jnp.array(w_wl.astype(np.float32))
    else:
        nm1_flat = None
        w_j = None

    wvis = (vis * weights).astype(np.complex64)
    u_j    = jnp.array(u_wl.astype(np.float32))
    v_j    = jnp.array(v_wl.astype(np.float32))
    wvis_j = jnp.array(wvis)

    n_pixels = n_pix * n_pix
    image_flat = np.zeros(n_pixels, dtype=np.float64)

    for start in range(0, n_pixels, pixel_chunk):
        end = min(start + pixel_chunk, n_pixels)
        l_c   = jnp.array(l_flat[start:end])
        m_c   = jnp.array(m_flat[start:end])

        if w_wl is not None:
            nm1_c = jnp.array(nm1_flat[start:end])
            chunk_vals = _sum_chunk_3d(l_c, m_c, nm1_c, u_j, v_j, w_j, wvis_j)
        else:
            chunk_vals = _sum_chunk_2d(l_c, m_c, u_j, v_j, wvis_j)

        image_flat[start:end] = np.array(chunk_vals)

    # Zero pixels below the horizon
    image_flat[~valid_pix] = 0.0

    weight_sum = float(weights.sum())
    if weight_sum == 0.0:
        return np.zeros((n_pix, n_pix), dtype=np.float32)

    image = (image_flat / weight_sum).reshape(n_pix, n_pix)
    return image.astype(np.float32)
