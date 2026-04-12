"""
CSV logger for inference results.

Appends one row per cell per scan to a CSV file. All cells are logged for
analysis and threshold tuning. The probability and above_threshold flag are
raw single-frame model outputs; confirmed tracks are logged separately.
"""
import csv
import threading
from datetime import datetime
from pathlib import Path

from tornotify.geo import radar_to_latlon
from tornotify.preprocess.cells import StormCell

_FIELDNAMES = [
    "scan_time",
    "site",
    "cell_id",
    "az_deg",
    "range_km",
    "lat",
    "lon",
    "dbz_max",
    "n_gates",
    "probability",
    "above_threshold",
]

_write_lock = threading.Lock()

def log_result(
    filepath: str,
    site: str,
    scan_time: datetime,
    cell: StormCell,
    probability: float,
    threshold: float,
    above_threshold: bool | None = None,
) -> None:
    """Append a single inference result to the CSV file."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    lat, lon = radar_to_latlon(site, cell.az_deg, cell.range_km)

    row = {
        "scan_time": scan_time.isoformat(),
        "site": site,
        "cell_id": cell.cell_id,
        "az_deg": round(cell.az_deg, 2),
        "range_km": round(cell.range_km, 2),
        "lat": round(lat, 5),
        "lon": round(lon, 5),
        "dbz_max": round(cell.dbz_max, 1),
        "n_gates": cell.n_gates,
        "probability": round(probability, 6),
        "above_threshold": probability >= threshold if above_threshold is None else above_threshold,
    }

    with _write_lock:
        write_header = not path.exists() or path.stat().st_size == 0
        with open(path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=_FIELDNAMES)
            if write_header:
                writer.writeheader()
            writer.writerow(row)
