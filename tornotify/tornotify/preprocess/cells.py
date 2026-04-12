"""
Storm cell identification from NEXRAD radar volumes.

Uses reflectivity threshold + connected-component clustering to locate
convective cells. Returns cell centroids in (azimuth, range) polar coords.
"""
import logging
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import label, center_of_mass
import pyart

from tornotify.config import (
    CELL_DBZ_THRESHOLD,
    CELL_MIN_GATES,
    CELL_MIN_RANGE_KM,
)
from tornotify.ingest.nexrad import (
    get_field,
    find_sweep_for_elevation,
    find_split_cut_sweeps,
)

logger = logging.getLogger(__name__)


@dataclass
class StormCell:
    """A detected storm cell in polar coordinates relative to the radar."""
    cell_id: str           # e.g. "KLWX-20240507230134-C00"
    az_deg: float          # azimuth of cell centroid (degrees from north)
    range_km: float        # range of cell centroid from radar (km)
    az_center_idx: int     # azimuthal bin index in sweep array
    range_center_idx: int  # range gate index in sweep array
    dbz_max: float         # peak reflectivity in the cell (dBZ)
    n_gates: int           # number of gates in the cell
    shear: float = 0.0     # azimuthal velocity shear (m/s/km)


def identify_cells(
    radar: pyart.core.Radar,
    scan_key: str,
    elevation_deg: float = 0.5,
    dbz_threshold: float = CELL_DBZ_THRESHOLD,
    min_gates: int = CELL_MIN_GATES,
    min_range_km: float = CELL_MIN_RANGE_KM,
    shear_threshold: float = 0.0,
) -> list[StormCell]:
    """
    Identify storm cells at a given elevation sweep via reflectivity threshold.

    Args:
       core.Radar object
        scan_key: unique ide radar: parsed pyart.ntifier for this scan (used in cell IDs)
        elevation_deg: target elevation angle (degrees)
        dbz_threshold: minimum dBZ to classify as convective
        min_gates: minimum cluster size to count as a cell
        shear_threshold: minimum azimuthal shear (m/s/km) to keep a cell.
            Set to 0 to keep all cells (shear is still computed).

    Returns:
        List of StormCell objects sorted by max reflectivity (descending)
    """
    sweep_idx = find_sweep_for_elevation(radar, elevation_deg)
    if sweep_idx is None:
        logger.warning(
            "No sweep near %.1f° — skipping cell identification", elevation_deg)
        return []

    dbz_data = get_field(radar, 'reflectivity', sweep_idx)
    dbz_arr = np.ma.filled(dbz_data, fill_value=-9999.0)

    # Binary mask: 1 where reflectivity exceeds threshold
    mask = (dbz_arr >= dbz_threshold).astype(int)

    # Exclude near-range gates (ground clutter, cone of silence, and
    # near-field chip geometry breakdown). mask is [n_rays, n_gates] so
    # we zero out the inner gate columns.
    ranges_km_all = radar.range['data'] / 1000.0
    near_range_gates = int(np.sum(ranges_km_all < min_range_km))
    if near_range_gates > 0:
        mask[:, :near_range_gates] = 0

    # Label connected components
    labeled, n_clusters = label(mask)
    if n_clusters == 0:
        logger.debug("No cells found above %.0f dBZ at %.1f°",
                     dbz_threshold, elevation_deg)
        return []

    # Get range array (km) and azimuth array (degrees) for this sweep
    start_ray = radar.sweep_start_ray_index['data'][sweep_idx]
    end_ray = radar.sweep_end_ray_index['data'][sweep_idx] + 1
    azimuths = radar.azimuth['data'][start_ray:end_ray]      # (n_rays,)
    ranges = radar.range['data'] / 1000.0                     # convert m → km

    cells = []
    for cluster_id in range(1, n_clusters + 1):
        cluster_mask = (labeled == cluster_id)
        n_gates = int(cluster_mask.sum())

        if n_gates < min_gates:
            continue

        # Center of mass in (ray_idx, gate_idx) space
        cy, cx = center_of_mass(cluster_mask)
        az_idx = int(round(cy))
        rng_idx = int(round(cx))

        # Clamp indices
        az_idx = max(0, min(az_idx, len(azimuths) - 1))
        rng_idx = max(0, min(rng_idx, len(ranges) - 1))

        az_deg = float(azimuths[az_idx])
        range_km = float(ranges[rng_idx])
        dbz_max = float(np.nanmax(dbz_arr[cluster_mask]))

        cell = StormCell(
            cell_id=f"{scan_key}-C{cluster_id:02d}",
            az_deg=az_deg,
            range_km=range_km,
            az_center_idx=az_idx,
            range_center_idx=rng_idx,
            dbz_max=dbz_max,
            n_gates=n_gates,
        )
        cells.append(cell)
        logger.debug(
            "Cell %s: az=%.1f° range=%.1f km dBZ_max=%.0f n_gates=%d",
            cell.cell_id, az_deg, range_km, dbz_max, n_gates,
        )

    # Compute azimuthal shear for each cell
    for cell in cells:
        cell.shear = compute_azimuthal_shear(radar, cell, elevation_deg)
        logger.debug("Cell %s: shear=%.1f m/s/km", cell.cell_id, cell.shear)

    # Filter by shear if threshold is set
    if shear_threshold > 0:
        before = len(cells)
        cells = [c for c in cells if c.shear >= shear_threshold]
        logger.info(
            "Shear filter (>=%.1f m/s/km): %d → %d cells",
            shear_threshold, before, len(cells),
        )

    cells.sort(key=lambda c: c.dbz_max, reverse=True)
    logger.info("Identified %d storm cells at %.1f°",
                len(cells), elevation_deg)
    return cells


def compute_azimuthal_shear(
    radar: pyart.core.Radar,
    cell: StormCell,
    elevation_deg: float = 0.5,
    window_az_bins: int = 10,
    window_range_bins: int = 10,
) -> float:
    """
    Compute azimuthal velocity shear around a cell centroid.

    Measures the max velocity difference across adjacent azimuths in a window
    around the cell, normalized by the azimuthal separation distance.
    This detects velocity couplets (inbound/outbound pairs) indicative of
    mesocyclone rotation.

    Args:
        radar: parsed pyart Radar object
        cell: StormCell with centroid indices
        elevation_deg: elevation angle to use for velocity
        window_az_bins: half-width of the azimuthal window (in ray indices)
        window_range_bins: half-width of the range window (in gate indices)

    Returns:
        Azimuthal shear in m/s/km. Higher values indicate stronger rotation.
        Returns 0.0 if velocity data is unavailable.
    """
    # Find the velocity sweep (may differ from reflectivity sweep in split-cut)
    field_sweep_map = find_split_cut_sweeps(radar, elevation_deg)
    vel_sweep = field_sweep_map.get('velocity')
    if vel_sweep is None or 'velocity' not in radar.fields:
        logger.debug("No velocity data for shear computation")
        return 0.0

    vel_data = get_field(radar, 'velocity', vel_sweep)
    vel_arr = np.ma.filled(vel_data, fill_value=np.nan)

    # Get azimuths and ranges for this sweep
    start_ray = radar.sweep_start_ray_index['data'][vel_sweep]
    end_ray = radar.sweep_end_ray_index['data'][vel_sweep] + 1
    azimuths = radar.azimuth['data'][start_ray:end_ray]
    ranges_km = radar.range['data'] / 1000.0
    n_rays, n_gates = vel_arr.shape

    # Find the ray closest to the cell azimuth in this sweep
    az_diffs = np.abs(azimuths - cell.az_deg)
    az_diffs = np.minimum(az_diffs, 360.0 - az_diffs)
    center_ray = int(np.argmin(az_diffs))

    # Find the gate closest to the cell range
    center_gate = int(np.argmin(np.abs(ranges_km - cell.range_km)))

    # Extract window around centroid
    ray_lo = center_ray - window_az_bins
    ray_hi = center_ray + window_az_bins + 1
    gate_lo = max(0, center_gate - window_range_bins)
    gate_hi = min(n_gates, center_gate + window_range_bins + 1)

    # Handle azimuthal wrap-around
    ray_indices = np.arange(ray_lo, ray_hi) % n_rays
    window = vel_arr[np.ix_(ray_indices, np.arange(gate_lo, gate_hi))]

    # Compute azimuthal shear: for each range gate, find max velocity
    # difference between adjacent azimuths
    if window.shape[0] < 2:
        return 0.0

    # Azimuthal difference along each range gate
    az_diff = np.diff(window, axis=0)  # [n_az-1, n_range]

    # Ignore NaN differences
    valid = ~np.isnan(az_diff)
    if not np.any(valid):
        return 0.0

    max_delta_v = float(np.nanmax(np.abs(az_diff)))

    # Normalize by azimuthal spacing in km at the cell's range
    az_spacing_deg = np.median(np.diff(azimuths)) if len(azimuths) > 1 else 1.0
    az_spacing_km = cell.range_km * np.radians(az_spacing_deg)
    if az_spacing_km < 0.01:
        az_spacing_km = 0.01

    shear = max_delta_v / az_spacing_km  # m/s per km
    return float(shear)
