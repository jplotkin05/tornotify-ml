"""
Extract TorNet-format radar chips from NEXRAD volumes.

Model input format (per variable):
  - Shape: [A=120, R=240, S=2]  (azimuth, range, tilts) — tilt_last=True
  - Variables (in order): ['DBZ', 'VEL', 'KDP', 'RHOHV', 'ZDR', 'WIDTH']
  - NO manual normalization — the model does it internally via Keras Normalization layers

The model also requires two additional inputs:
  - 'range_folded_mask': [A=120, R=240, S=2]  — marks velocity-folded gates
  - 'coordinates':       [A=120, R=240, 2]    — r and r_inv coordinate fields

Full input dict keys:
  'DBZ', 'VEL', 'KDP', 'RHOHV', 'ZDR', 'WIDTH', 'range_folded_mask', 'coordinates'

Source: tornet/tornet/models/keras/cnn_baseline.py build_model()
        tornet/tornet/data/preprocess.py compute_coordinates()
        tornet/tornet/data/constants.py ALL_VARIABLES
"""
import logging
from typing import NamedTuple

import numpy as np
import pyart
from scipy.ndimage import zoom

from tornotify.config import (
    CHIP_AZIMUTH_SPAN_DEG,
    CHIP_RANGE_KM,
    CHIP_N_AZ_BINS,
    CHIP_N_RANGE_GATES,
    CHIP_N_ELEVATIONS,
    TARGET_ELEVATIONS,
)
from tornotify.ingest.nexrad import get_field, find_split_cut_sweeps
from tornotify.preprocess.cells import StormCell

logger = logging.getLogger(__name__)

# Variable order matches tornet/tornet/data/constants.py ALL_VARIABLES
# Maps TorNet variable name → pyart field name(s).
# KDP may be available as 'specific_differential_phase' (newer radars) or
# must be approximated from 'differential_phase' (PhiDP) on older data.
VARIABLE_MAP = {
    'DBZ':   'reflectivity',
    'VEL':   'velocity',
    'KDP':   'specific_differential_phase',
    'RHOHV': 'cross_correlation_ratio',
    'ZDR':   'differential_reflectivity',
    'WIDTH': 'spectrum_width',
}

ALL_VARIABLES = ['DBZ', 'VEL', 'KDP', 'RHOHV', 'ZDR', 'WIDTH']

# Range of the chip in meters
CHIP_RANGE_M = CHIP_RANGE_KM * 1000.0

# Coordinate scale factor (matches tornet/data/preprocess.py)
_COORD_SCALE = 1e-5


class RangeWindow(NamedTuple):
    indices: np.ndarray
    lower_m: float
    upper_m: float


def extract_chip(
    radar: pyart.core.Radar,
    cell: StormCell,
) -> dict[str, np.ndarray] | None:
    """
    Extract a TorNet-format chip dict for a storm cell.

    Returns a dict with keys matching the model's input signature:
      - Each variable in ALL_VARIABLES: shape [A=120, R=240, S=2], float32
      - 'range_folded_mask':             shape [A=120, R=240, S=2], float32
      - 'coordinates':                   shape [A=120, R=240, 2],   float32
      - '_az_lower', '_az_upper':        scalar metadata (degrees) for coordinate computation
      - '_rng_lower', '_rng_upper':      scalar metadata (meters)

    Returns None if extraction fails.
    """
    A, R, S = CHIP_N_AZ_BINS, CHIP_N_RANGE_GATES, CHIP_N_ELEVATIONS

    # Azimuth window centered on storm cell. Keep unwrapped limits, because
    # TorNet samples can use azimuth bounds that extend below 0 or above 360.
    az_lower = cell.az_deg - CHIP_AZIMUTH_SPAN_DEG / 2.0
    az_upper = cell.az_deg + CHIP_AZIMUTH_SPAN_DEG / 2.0

    range_window = _range_window_for_cell(radar.range['data'], cell, CHIP_RANGE_M, R)
    if len(range_window.indices) == 0:
        logger.warning("Empty range window for cell %s", cell.cell_id)
        return None
    rng_lower_m = range_window.lower_m
    rng_upper_m = range_window.upper_m

    chip = {}
    folded_mask = np.zeros((A, R, S), dtype=np.float32)

    for s_idx, target_elev in enumerate(TARGET_ELEVATIONS):
        # Use split-cut-aware sweep selection: maps each pyart field name
        # to the sweep that actually contains data for it.
        field_sweep_map = find_split_cut_sweeps(radar, target_elev)
        if not field_sweep_map:
            logger.warning("No sweeps near %.1f° for chip extraction", target_elev)
            continue

        # Use the reflectivity sweep as the reference for azimuth/range grids
        ref_sweep = field_sweep_map.get('reflectivity', next(iter(field_sweep_map.values())))
        start_ray = radar.sweep_start_ray_index['data'][ref_sweep]
        end_ray = radar.sweep_end_ray_index['data'][ref_sweep] + 1
        azimuths = radar.azimuth['data'][start_ray:end_ray]

        # Select azimuthal window (handles 360° wrap and preserves spatial order)
        az_indices = _azimuth_indices_for_window(azimuths, az_lower, az_upper)

        # Select range window
        range_indices = range_window.indices

        if len(az_indices) == 0 or len(range_indices) == 0:
            logger.warning("Empty azimuth or range window at %.1f°", target_elev)
            continue

        for var_name in ALL_VARIABLES:
            field_name = VARIABLE_MAP[var_name]

            # Find the sweep that has data for this field
            sweep_for_field = field_sweep_map.get(field_name)

            if field_name not in radar.fields or sweep_for_field is None:
                logger.debug("Field %s not available — filling zeros", field_name)
                slice_data = np.zeros((len(az_indices), len(range_indices)), dtype=np.float32)
            else:
                # The sweep for this field may differ from ref_sweep (split-cut).
                # Recompute azimuth indices for this sweep's ray layout.
                fs_start = radar.sweep_start_ray_index['data'][sweep_for_field]
                fs_end = radar.sweep_end_ray_index['data'][sweep_for_field] + 1
                fs_az = radar.azimuth['data'][fs_start:fs_end]
                fs_az_indices = _azimuth_indices_for_window(fs_az, az_lower, az_upper)

                if len(fs_az_indices) == 0:
                    slice_data = np.zeros((len(az_indices), len(range_indices)), dtype=np.float32)
                else:
                    raw = get_field(radar, field_name, sweep_for_field)
                    arr = np.ma.filled(raw, fill_value=np.nan).astype(np.float32)
                    slice_data = arr[np.ix_(fs_az_indices, range_indices)]

            resampled = _resample_2d(slice_data, A, R)  # [A, R]

            if var_name not in chip:
                chip[var_name] = np.zeros((A, R, S), dtype=np.float32)
            chip[var_name][:, :, s_idx] = resampled

        # Range-folded mask: 1 where velocity is masked (folded), 0 elsewhere
        vel_sweep = field_sweep_map.get('velocity')
        if vel_sweep is not None and 'velocity' in radar.fields:
            vel_raw = get_field(radar, 'velocity', vel_sweep)
            folded = vel_raw.mask.astype(np.float32) if hasattr(vel_raw, 'mask') else np.zeros_like(vel_raw.data, dtype=np.float32)
            vs_start = radar.sweep_start_ray_index['data'][vel_sweep]
            vs_end = radar.sweep_end_ray_index['data'][vel_sweep] + 1
            vs_az = radar.azimuth['data'][vs_start:vs_end]
            vs_az_indices = _azimuth_indices_for_window(vs_az, az_lower, az_upper)
            if len(vs_az_indices) > 0:
                folded_slice = folded[np.ix_(vs_az_indices, range_indices)]
                folded_mask[:, :, s_idx] = _resample_2d(folded_slice, A, R)

    # Fill any missing variables with zeros
    for var_name in ALL_VARIABLES:
        if var_name not in chip:
            chip[var_name] = np.zeros((A, R, S), dtype=np.float32)

    chip['range_folded_mask'] = folded_mask

    # Compute CoordConv coordinate fields
    chip['coordinates'] = _compute_coordinates(
        az_lower_deg=az_lower,
        az_upper_deg=az_upper,
        rng_lower_m=rng_lower_m,
        rng_upper_m=rng_upper_m,
        n_az=A,
        n_rng=R,
    )

    # Replace NaN with 0.0 in all model-input arrays.
    # The TorNet model's FillNaNs layer handles this internally, but some
    # Keras backends choke on NaN in tensor conversion, so we clean here.
    model_keys = ALL_VARIABLES + ['range_folded_mask', 'coordinates']
    for k in model_keys:
        if k in chip:
            chip[k] = np.nan_to_num(chip[k], nan=0.0)

    # Store metadata for downstream use
    chip['_az_lower'] = az_lower
    chip['_az_upper'] = az_upper
    chip['_rng_lower_m'] = rng_lower_m
    chip['_rng_upper_m'] = rng_upper_m

    return chip


def _azimuth_indices_for_window(
    azimuths: np.ndarray,
    az_lower_deg: float,
    az_upper_deg: float,
) -> np.ndarray:
    """
    Return indices for an unwrapped azimuth window, ordered from lower to upper.

    Py-ART azimuths are usually stored in [0, 360), but TorNet chip metadata can
    extend outside that interval. Unwrapping around the chip center keeps a
    north-crossing window continuous instead of concatenating 0° rays before
    359° rays.
    """
    center_deg = (az_lower_deg + az_upper_deg) / 2.0
    az_unwrapped = (
        center_deg
        + ((np.asarray(azimuths, dtype=float) - center_deg + 180.0) % 360.0)
        - 180.0
    )
    mask = (az_unwrapped >= az_lower_deg) & (az_unwrapped <= az_upper_deg)
    indices = np.where(mask)[0]
    if len(indices) == 0:
        return indices
    return indices[np.argsort(az_unwrapped[indices])]


def _range_window_for_cell(
    ranges_m: np.ndarray,
    cell: StormCell,
    span_m: float,
    target_gates: int,
) -> RangeWindow:
    """
    Return a fixed-size range gate window centered on the detected storm cell.
    """
    ranges_m = np.asarray(ranges_m, dtype=float)
    if len(ranges_m) == 0:
        return RangeWindow(indices=np.array([], dtype=int), lower_m=0.0, upper_m=0.0)

    gate_spacing = _median_gate_spacing_m(ranges_m)
    center_m = cell.range_km * 1000.0
    center_idx = int(np.nanargmin(np.abs(ranges_m - center_m)))
    span_gates = max(1, int(round(span_m / gate_spacing)))
    n_gates = min(target_gates, span_gates, len(ranges_m))

    start = center_idx - n_gates // 2
    end = start + n_gates
    if start < 0:
        start = 0
        end = n_gates
    if end > len(ranges_m):
        end = len(ranges_m)
        start = max(0, end - n_gates)

    indices = np.arange(start, end)
    if len(indices) == 0:
        return RangeWindow(indices=indices, lower_m=0.0, upper_m=0.0)

    rng_lower_m = max(0.0, float(ranges_m[indices[0]]) - gate_spacing)
    rng_upper_m = float(ranges_m[indices[-1]]) + gate_spacing
    return RangeWindow(indices=indices, lower_m=rng_lower_m, upper_m=rng_upper_m)


def _median_gate_spacing_m(ranges_m: np.ndarray) -> float:
    if len(ranges_m) < 2:
        return 250.0
    spacing = np.nanmedian(np.diff(ranges_m))
    return float(spacing) if np.isfinite(spacing) and spacing > 0 else 250.0


def _compute_coordinates(
    az_lower_deg: float,
    az_upper_deg: float,
    rng_lower_m: float,
    rng_upper_m: float,
    n_az: int,
    n_rng: int,
    min_range_m: float = 2125.0,
) -> np.ndarray:
    """
    Compute CoordConv coordinate fields matching tornet/data/preprocess.py logic.
    Returns array of shape [n_az, n_rng, 2] containing (r, r_inv).
    """
    SCALE = _COORD_SCALE
    rng_lo = (rng_lower_m + 250) * SCALE
    rng_hi = (rng_upper_m - 250) * SCALE
    min_r = min_range_m * SCALE

    # Convert azimuth to math convention (0=east, counterclockwise)
    az_lo_rad = (90.0 - az_lower_deg) * np.pi / 180.0
    az_hi_rad = (90.0 - az_upper_deg) * np.pi / 180.0

    az = np.linspace(az_lo_rad, az_hi_rad, n_az)
    rg = np.linspace(rng_lo, rng_hi, n_rng)

    R_grid, _ = np.meshgrid(rg, az, indexing='xy')  # [n_az, n_rng]
    R_grid = np.where(R_grid >= min_r, R_grid, min_r)
    R_inv = 1.0 / R_grid

    coords = np.stack([R_grid, R_inv], axis=-1).astype(np.float32)  # [n_az, n_rng, 2]
    return coords


def _resample_2d(arr: np.ndarray, target_rows: int, target_cols: int) -> np.ndarray:
    """Bilinear resample to (target_rows, target_cols), NaN-safe."""
    if arr.shape == (target_rows, target_cols):
        return arr
    src_rows, src_cols = arr.shape
    if src_rows == 0 or src_cols == 0:
        return np.zeros((target_rows, target_cols), dtype=np.float32)
    zoom_r = target_rows / src_rows
    zoom_c = target_cols / src_cols
    nan_mask = np.isnan(arr)
    arr_clean = np.where(nan_mask, 0.0, arr)
    resampled = zoom(arr_clean, (zoom_r, zoom_c), order=1)
    nan_resampled = zoom(nan_mask.astype(float), (zoom_r, zoom_c), order=0)
    resampled[nan_resampled > 0.5] = np.nan
    return resampled.astype(np.float32)


def chip_to_batch(chip: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """
    Add a batch dimension (size 1) to each model-input array in the chip dict.
    Skips metadata keys (those starting with '_').
    """
    model_keys = ALL_VARIABLES + ['range_folded_mask', 'coordinates']
    return {k: chip[k][np.newaxis, ...] for k in model_keys}
