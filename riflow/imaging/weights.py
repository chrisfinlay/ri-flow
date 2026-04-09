"""
weights.py - Imaging weight schemes shared across NUFFT and DFT imagers.

Supported schemes (WSClean-compatible):
  natural  — all weights equal to 1
  uniform  — inverse UV-cell occupancy
  briggs   — Robust weighting (Briggs 1995); robust parameter in [-2, +2]
"""

from __future__ import annotations

import numpy as np


def parse_weight(weight_str: str) -> tuple[str, float | None]:
    """Parse a WSClean-style weight string into (scheme, robust).

    Examples
    --------
    'natural'     → ('natural', None)
    'uniform'     → ('uniform', None)
    'briggs -0.5' → ('briggs', -0.5)
    """
    parts = weight_str.strip().lower().split()
    scheme = parts[0]
    if scheme == "briggs":
        return "briggs", float(parts[1]) if len(parts) > 1 else 0.0
    if scheme == "uniform":
        return "uniform", None
    return "natural", None


def compute_weights(
    u_wl: np.ndarray,
    v_wl: np.ndarray,
    n_pix: int,
    pixel_rad: float,
    scheme: str,
    robust: float | None,
) -> np.ndarray:
    """Return per-baseline imaging weights of shape (N,).

    The UV density grid uses the same cell size as the image pixel grid:
        uv_cell = 1 / (n_pix * pixel_rad)  [wavelengths]

    Hermitian conjugates are included when computing cell occupancy so that
    both (u,v) and (-u,-v) count toward the same cell density.

    Schemes
    -------
    natural : all weights = 1
    uniform : w_k = 1 / n_k  (inverse cell occupancy)
    briggs  : w_k = 1 / (1 + n_k / f²)
              f² = (5 × 10^{-R})² × N_total / Σ n_i²
    """
    N = len(u_wl)
    if scheme == "natural":
        return np.ones(N, dtype=np.float32)

    uv_cell = 1.0 / (n_pix * pixel_rad)

    u_idx   = np.clip(np.floor( u_wl / uv_cell + n_pix / 2).astype(int), 0, n_pix - 1)
    v_idx   = np.clip(np.floor( v_wl / uv_cell + n_pix / 2).astype(int), 0, n_pix - 1)
    u_idx_c = np.clip(np.floor(-u_wl / uv_cell + n_pix / 2).astype(int), 0, n_pix - 1)
    v_idx_c = np.clip(np.floor(-v_wl / uv_cell + n_pix / 2).astype(int), 0, n_pix - 1)

    density = np.zeros((n_pix, n_pix), dtype=np.float64)
    np.add.at(density, (v_idx,   u_idx),   1)
    np.add.at(density, (v_idx_c, u_idx_c), 1)
    n_k = density[v_idx, u_idx]

    if scheme == "uniform":
        w = 1.0 / np.maximum(n_k, 1.0)
    else:  # briggs
        sum_n2 = float(np.sum(density[density > 0] ** 2))
        f2 = (5.0 * 10.0 ** (-robust)) ** 2 * (2 * N) / sum_n2  # type: ignore[operator]
        w = 1.0 / (1.0 + f2 * n_k)

    return w.astype(np.float32)
