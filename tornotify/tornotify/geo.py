"""
Geographic utilities for radar-relative coordinate conversion.
"""
from pyproj import Geod

from tornotify.config import RADAR_SITES

_GEOD = Geod(ellps="WGS84")


def radar_to_latlon(site_id: str, azimuth_deg: float, range_km: float) -> tuple[float, float]:
    """
    Convert NEXRAD polar coordinates to geographic lat/lon.

    Args:
        site_id: e.g. "KLWX"
        azimuth_deg: degrees clockwise from north
        range_km: distance from radar in km

    Returns:
        (lat, lon) in decimal degrees
    """
    site = RADAR_SITES[site_id]
    lon, lat, _ = _GEOD.fwd(site["lon"], site["lat"], azimuth_deg, range_km * 1000)
    return float(lat), float(lon)
