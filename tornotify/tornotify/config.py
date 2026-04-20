"""
Central configuration for TorNotify.
Load runtime settings from environment / .env file.
"""
import os
from dotenv import load_dotenv

from tornotify.radar_sites import CORE_RADAR_SITES, build_radar_site_catalog

load_dotenv()

# ---------------------------------------------------------------------------
# Radar sites
# ---------------------------------------------------------------------------
# Phase 1 target site
PRIMARY_SITE = "KLWX"

# All NEXRAD site configs: site_id -> lat/lon/elevation/name metadata.
# The catalog is built from Py-ART's NEXRAD_LOCATIONS source table without
# importing Py-ART, with locally validated site metadata overlaid.
RADAR_SITES = build_radar_site_catalog(CORE_RADAR_SITES)

# ---------------------------------------------------------------------------
# ML inference
# ---------------------------------------------------------------------------
# HuggingFace model identifier for pretrained TorNet baseline
TORNET_MODEL_REPO = "tornet-ml/tornado_detector_baseline_v1"
TORNET_MODEL_FILE = "tornado_detector_baseline.keras"

# Confidence threshold: raw candidate if P(tornado) >= this value
CONFIDENCE_THRESHOLD = 0.6

# Candidate quality flags for review and visualization. Temporal tracking
# controls whether a raw over-threshold score becomes actionable.
DETECTION_MIN_RANGE_KM = float(os.getenv("TORNOTIFY_DETECTION_MIN_RANGE_KM", "40.0"))
DETECTION_MIN_GATES = int(os.getenv("TORNOTIFY_DETECTION_MIN_GATES", "100"))
DETECTION_MIN_HEATMAP_EDGE_MARGIN = float(
    os.getenv("TORNOTIFY_DETECTION_MIN_HEATMAP_EDGE_MARGIN", "0.10")
)

# Temporal tracking confirmation. Raw over-threshold candidates create or update
# tracks; confirmed detection events are logged only after a track persists.
TRACK_CONFIRM_SCANS = int(os.getenv("TORNOTIFY_TRACK_CONFIRM_SCANS", "2"))
TRACK_CONFIRM_WINDOW_SCANS = int(os.getenv("TORNOTIFY_TRACK_CONFIRM_WINDOW_SCANS", "3"))
TRACK_MATCH_BASE_KM = float(os.getenv("TORNOTIFY_TRACK_MATCH_BASE_KM", "15.0"))
TRACK_MAX_SPEED_KMH = float(os.getenv("TORNOTIFY_TRACK_MAX_SPEED_KMH", "120.0"))
TRACK_ACTIVATION_MATCH_KM = float(os.getenv("TORNOTIFY_TRACK_ACTIVATION_MATCH_KM", "45.0"))
TRACK_MAX_MISSED_SCANS = int(os.getenv("TORNOTIFY_TRACK_MAX_MISSED_SCANS", "2"))

# How often to poll for new scans (seconds)
POLL_INTERVAL_SECONDS = 300  # 5 minutes

# Adaptive multi-site scanner defaults. Active sites are polled faster while
# quiet/erroring sites back off to avoid wasting worker capacity.
SCANNER_DEFAULT_WORKERS = int(os.getenv("TORNOTIFY_SCANNER_WORKERS", "2"))
SCANNER_HOT_POLL_SECONDS = int(os.getenv("TORNOTIFY_SCANNER_HOT_POLL_SECONDS", "60"))
SCANNER_ACTIVE_POLL_SECONDS = int(os.getenv("TORNOTIFY_SCANNER_ACTIVE_POLL_SECONDS", "120"))
SCANNER_QUIET_POLL_SECONDS = int(os.getenv("TORNOTIFY_SCANNER_QUIET_POLL_SECONDS", "300"))
SCANNER_MAX_QUIET_POLL_SECONDS = int(os.getenv("TORNOTIFY_SCANNER_MAX_QUIET_POLL_SECONDS", "900"))
SCANNER_NO_SCAN_POLL_SECONDS = int(os.getenv("TORNOTIFY_SCANNER_NO_SCAN_POLL_SECONDS", "600"))
SCANNER_ERROR_BACKOFF_SECONDS = int(os.getenv("TORNOTIFY_SCANNER_ERROR_BACKOFF_SECONDS", "300"))
SCANNER_MAX_ERROR_BACKOFF_SECONDS = int(os.getenv("TORNOTIFY_SCANNER_MAX_ERROR_BACKOFF_SECONDS", "1800"))

# Distributed scanner state. Redis is used only for queue/state coordination;
# raw cell scores and confirmed tracks continue to be written to CSV files.
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
DISTRIBUTED_REDIS_PREFIX = os.getenv("TORNOTIFY_REDIS_PREFIX", "tornotify")
DISTRIBUTED_QUEUE_KEY = os.getenv(
    "TORNOTIFY_DISTRIBUTED_QUEUE_KEY",
    f"{DISTRIBUTED_REDIS_PREFIX}:scan_queue",
)
DISTRIBUTED_DEFAULT_WORKERS = int(
    os.getenv("TORNOTIFY_DISTRIBUTED_WORKERS", str(max(1, min(os.cpu_count() or 1, 8))))
)
DISTRIBUTED_SITE_LOCK_TTL_SECONDS = int(
    os.getenv("TORNOTIFY_DISTRIBUTED_SITE_LOCK_TTL_SECONDS", "900")
)
DISTRIBUTED_PROCESSED_TTL_SECONDS = int(
    os.getenv("TORNOTIFY_DISTRIBUTED_PROCESSED_TTL_SECONDS", "259200")
)
DISTRIBUTED_TRACK_TTL_SECONDS = int(
    os.getenv("TORNOTIFY_DISTRIBUTED_TRACK_TTL_SECONDS", "21600")
)
DISTRIBUTED_WORKER_SLEEP_SECONDS = float(
    os.getenv("TORNOTIFY_DISTRIBUTED_WORKER_SLEEP_SECONDS", "2.0")
)

# ---------------------------------------------------------------------------
# Storm cell identification
# ---------------------------------------------------------------------------
# Reflectivity threshold (dBZ) for identifying convective cells
CELL_DBZ_THRESHOLD = 40.0

# Minimum cluster size (number of gates) to count as a storm cell
CELL_MIN_GATES = 20

# Minimum range from radar (km) for cell identification. Gates inside this
# radius are dominated by ground clutter, sidelobe contamination, and the
# cone of silence; storm-centered chip geometry also breaks down in the
# near field. Tornado detection within this radius is not meaningful.
CELL_MIN_RANGE_KM = 10.0

# TorNet chip geometry
CHIP_AZIMUTH_SPAN_DEG = 60      # total azimuthal width of chip
CHIP_RANGE_KM = 60              # range depth of chip
CHIP_N_AZ_BINS = 120            # azimuthal resolution
CHIP_N_RANGE_GATES = 240        # range resolution
CHIP_N_ELEVATIONS = 2           # 0.5° and 0.9°
TARGET_ELEVATIONS = [0.5, 0.9]  # degrees

# ---------------------------------------------------------------------------
# Keras backend
# ---------------------------------------------------------------------------
# Set before importing keras: KERAS_BACKEND=torch
KERAS_BACKEND = os.getenv("KERAS_BACKEND", "torch")
