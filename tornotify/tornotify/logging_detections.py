"""
CSV logger for temporally confirmed tornado detection tracks.

This log is intentionally separate from results.csv. The raw results log keeps
every scored cell, while this file records only tracks that become actionable
after temporal confirmation.
"""
from __future__ import annotations

import csv
import threading
from pathlib import Path
from typing import Any

from tornotify.geo import radar_to_latlon

_FIELDNAMES = [
    "confirmed_at",
    "site",
    "track_id",
    "first_seen",
    "latest_scan_time",
    "cell_id",
    "probability",
    "max_probability",
    "hits_in_window",
    "confirmation_window",
    "observations",
    "az_deg",
    "range_km",
    "lat",
    "lon",
    "dbz_max",
    "n_gates",
    "shear",
    "quality_pass",
    "quality_reasons",
    "activation_az_deg",
    "activation_range_km",
    "heatmap_edge_margin",
]

_write_lock = threading.Lock()


def log_detection_event(filepath: str, tracking_update: Any) -> None:
    """Append one newly confirmed detection track event."""
    if tracking_update.track is None:
        raise ValueError("tracking_update must include a confirmed track")

    candidate = tracking_update.candidate
    track = tracking_update.track
    cell = candidate.cell
    result = candidate.detection_result
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    lat, lon = radar_to_latlon(candidate.site, cell.az_deg, cell.range_km)
    first_seen = track.created_at
    confirmed_at = track.confirmed_at or candidate.scan_time
    row = {
        "confirmed_at": confirmed_at.isoformat(),
        "site": candidate.site,
        "track_id": track.track_id,
        "first_seen": first_seen.isoformat(),
        "latest_scan_time": candidate.scan_time.isoformat(),
        "cell_id": cell.cell_id,
        "probability": round(candidate.probability, 6),
        "max_probability": round(track.max_probability, 6),
        "hits_in_window": tracking_update.hits_in_window,
        "confirmation_window": tracking_update.window_size,
        "observations": track.observations_count,
        "az_deg": round(cell.az_deg, 2),
        "range_km": round(cell.range_km, 2),
        "lat": round(lat, 5),
        "lon": round(lon, 5),
        "dbz_max": round(cell.dbz_max, 1),
        "n_gates": cell.n_gates,
        "shear": round(cell.shear, 3),
        "quality_pass": candidate.quality_pass,
        "quality_reasons": "; ".join(candidate.quality_reasons),
        "activation_az_deg": _round_optional(getattr(result, "activation_az_deg", None), 2),
        "activation_range_km": _round_optional(getattr(result, "activation_range_km", None), 2),
        "heatmap_edge_margin": _round_optional(getattr(result, "heatmap_edge_margin", None), 4),
    }

    with _write_lock:
        write_header = not path.exists() or path.stat().st_size == 0
        with open(path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=_FIELDNAMES)
            if write_header:
                writer.writeheader()
            writer.writerow(row)


def _round_optional(value: Any, ndigits: int) -> float | None:
    if value is None:
        return None
    return round(float(value), ndigits)
