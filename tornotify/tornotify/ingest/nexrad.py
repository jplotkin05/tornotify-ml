"""
NEXRAD Level-II data ingestion from AWS S3.

Handles polling for new scans, downloading, and parsing via pyart.
"""
import tempfile
import logging
from datetime import datetime, timezone
from pathlib import Path

import nexradaws
import pyart

from tornotify.config import PRIMARY_SITE

logger = logging.getLogger(__name__)

_conn = nexradaws.NexradAwsInterface()


def get_available_scans(site: str, dt: datetime | None = None):
    """
    Return list of available AwsNexradFile objects for a site on a given date.
    Defaults to today (UTC).
    """
    if dt is None:
        dt = datetime.now(timezone.utc)
    scans = [
        scan
        for scan in _conn.get_avail_scans(dt.year, dt.month, dt.day, site)
        if _is_level2_volume_scan(scan)
    ]
    logger.debug("Found %d scans for %s on %s", len(scans), site, dt.date())
    return scans


def _is_level2_volume_scan(scan_file) -> bool:
    filename = getattr(scan_file, "filename", "") or getattr(scan_file, "key", "")
    return not filename.endswith("_MDM")


def get_latest_scan(site: str, dt: datetime | None = None):
    """Return the most recent AwsNexradFile for a site."""
    scans = get_available_scans(site, dt)
    if not scans:
        raise RuntimeError(f"No scans found for site {site} on {dt}")
    return scans[-1]


def download_scan(scan_file, download_dir: str) -> str:
    """
    Download a single AwsNexradFile to download_dir.
    Returns the local file path.
    """
    results = _conn.download(scan_file, download_dir)
    if not results.success:
        raise RuntimeError(f"Download failed for {scan_file.filename}: {results.failed}")
    local_path = results.success[0].filepath
    logger.info("Downloaded %s → %s", scan_file.filename, local_path)
    return local_path


def parse_scan(filepath: str) -> pyart.core.Radar:
    """
    Parse a NEXRAD Level-II file using pyart.
    Derives KDP from PhiDP if not already present.
    Returns a pyart.core.Radar object.
    """
    radar = pyart.io.read_nexrad_archive(filepath)
    logger.debug(
        "Parsed %s: %d sweeps, fields=%s",
        Path(filepath).name,
        radar.nsweeps,
        list(radar.fields.keys()),
    )

    # Derive KDP (specific_differential_phase) from PhiDP if not natively present.
    # NEXRAD Level-II typically has differential_phase (PhiDP) but not KDP.
    if 'specific_differential_phase' not in radar.fields and 'differential_phase' in radar.fields:
        logger.info("Deriving KDP from PhiDP (Vulpiani method, S-band)")
        try:
            kdp_dict, _ = pyart.retrieve.kdp_vulpiani(
                radar,
                phidp_field='differential_phase',
                band='S',  # NEXRAD is S-band
            )
            radar.add_field('specific_differential_phase', kdp_dict, replace_existing=True)
            logger.info("KDP derived successfully")
        except Exception as e:
            logger.warning("KDP derivation failed: %s — KDP channel will be zeros", e)

    return radar


def fetch_latest_radar(site: str = PRIMARY_SITE, dt: datetime | None = None) -> pyart.core.Radar:
    """
    Convenience: download and parse the latest scan for a site.
    Uses a temporary directory (auto-cleaned after function returns).
    Returns a pyart.core.Radar object.

    Note: For persistent caching between scans, pass a stable download_dir
    and handle cleanup yourself.
    """
    scan = get_latest_scan(site, dt)
    with tempfile.TemporaryDirectory() as tmpdir:
        local_path = download_scan(scan, tmpdir)
        radar = parse_scan(local_path)
    return radar


def get_scan_time(radar: pyart.core.Radar) -> datetime:
    """Extract UTC scan time from a parsed radar object."""
    # pyart stores time in seconds since epoch in radar.time['data']
    t0 = radar.time['units']  # e.g. "seconds since 2024-05-07T23:01:34Z"
    dt_str = t0.replace("seconds since ", "").strip()
    return datetime.fromisoformat(dt_str.replace("Z", "+00:00"))


def get_field(radar: pyart.core.Radar, field_name: str, sweep_idx: int = 0):
    """
    Return the masked array for a field at a given sweep index.
    Common field names: 'reflectivity', 'velocity', 'spectrum_width',
                        'differential_reflectivity', 'cross_correlation_ratio',
                        'specific_differential_phase'
    """
    start = radar.sweep_start_ray_index['data'][sweep_idx]
    end = radar.sweep_end_ray_index['data'][sweep_idx] + 1
    return radar.fields[field_name]['data'][start:end, :]


def get_elevation_angle(radar: pyart.core.Radar, sweep_idx: int) -> float:
    """Return the target elevation angle for a sweep in degrees."""
    start = radar.sweep_start_ray_index['data'][sweep_idx]
    return float(radar.elevation['data'][start])


def find_sweep_for_elevation(radar: pyart.core.Radar, target_deg: float, tol: float = 0.3) -> int | None:
    """
    Find the sweep index closest to target_deg elevation.
    Returns None if no sweep is within tol degrees.
    """
    best_idx = None
    best_diff = float('inf')
    for i in range(radar.nsweeps):
        angle = get_elevation_angle(radar, i)
        diff = abs(angle - target_deg)
        if diff < best_diff:
            best_diff = diff
            best_idx = i
    return best_idx if best_diff <= tol else None


def find_split_cut_sweeps(
    radar: pyart.core.Radar,
    target_deg: float,
    tol: float = 0.5,
) -> dict:
    """
    Find the split-cut sweep pair near target_deg elevation.

    NEXRAD VCPs use split cuts at low elevations: one sweep is optimized for
    Doppler (has VEL/WIDTH, no dual-pol) and the other is optimized for
    surveillance/dual-pol (has ZDR/RHOHV/PhiDP, no VEL/WIDTH).

    Returns a dict mapping field name → sweep index, covering all available
    fields from the best sweep pair near the target elevation.
    """
    import numpy as np

    candidates = []
    for i in range(radar.nsweeps):
        angle = get_elevation_angle(radar, i)
        if abs(angle - target_deg) <= tol:
            candidates.append((i, angle))

    if not candidates:
        return {}

    field_sweep_map = {}
    for sweep_idx, _angle in candidates:
        start = radar.sweep_start_ray_index['data'][sweep_idx]
        end = radar.sweep_end_ray_index['data'][sweep_idx] + 1
        for fname in radar.fields:
            data = radar.fields[fname]['data'][start:end, :]
            valid = np.count_nonzero(~np.ma.getmaskarray(data))
            if valid > 0 and fname not in field_sweep_map:
                field_sweep_map[fname] = sweep_idx

    return field_sweep_map
