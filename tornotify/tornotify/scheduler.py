"""
APScheduler-based polling loop for real-time radar ingestion + inference.

Can be used by any host process that needs background radar polling.
"""
import logging
from datetime import datetime, timezone

from apscheduler.schedulers.background import BackgroundScheduler

from tornotify.config import (
    PRIMARY_SITE,
    POLL_INTERVAL_SECONDS,
)

logger = logging.getLogger(__name__)

# Track processed S3 keys per site to avoid reprocessing.
_processed_keys: dict[str, set[str]] = {}
_track_managers: dict[str, object] = {}


def _poll_site(
    site: str,
    csv_path: str = "data/results.csv",
    detection_csv_path: str | None = "data/detections.csv",
    image_dir: str | None = "data/radar_images",
) -> None:
    """
    Poll a single NEXRAD site for new scans, run inference, and log detections.
    """
    from tornotify.pipeline import process_latest_scan
    from tornotify.tracking import TrackManager

    processed = _processed_keys.setdefault(site, set())
    track_manager = _track_managers.setdefault(site, TrackManager())
    process_latest_scan(
        site,
        datetime.now(timezone.utc),
        processed,
        csv_path=csv_path,
        image_dir=image_dir,
        detection_csv_path=detection_csv_path,
        track_manager=track_manager,
    )


def create_scheduler(
    sites: list[str] | None = None,
    csv_path: str = "data/results.csv",
    detection_csv_path: str | None = "data/detections.csv",
    image_dir: str | None = "data/radar_images",
) -> BackgroundScheduler:
    """
    Create and return (but don't start) an APScheduler that polls NEXRAD sites.
    """
    if sites is None:
        sites = [PRIMARY_SITE]

    scheduler = BackgroundScheduler()

    for site in sites:
        scheduler.add_job(
            _poll_site,
            "interval",
            seconds=POLL_INTERVAL_SECONDS,
            args=[site, csv_path, detection_csv_path, image_dir],
            id=f"poll_{site}",
            name=f"Poll NEXRAD {site}",
            max_instances=1,
            coalesce=True,
        )
        logger.info("Scheduled polling for site %s every %ds", site, POLL_INTERVAL_SECONDS)

    return scheduler
