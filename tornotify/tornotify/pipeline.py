"""
Shared NEXRAD scan processing flow.

The CLI runner and APScheduler both use this module so scan selection,
preprocessing, inference, detection logging, and image artifact generation stay
in one place.
"""
import logging
import tempfile
from dataclasses import dataclass
from datetime import datetime

from tornotify.config import CONFIDENCE_THRESHOLD
from tornotify.ingest.nexrad import (
    download_scan,
    get_available_scans,
    get_scan_time,
    parse_scan,
)
from tornotify.logging_detections import log_detection_event
from tornotify.logging_csv import log_result
from tornotify.ml.detector import predict_batch_detailed
from tornotify.ml.quality import evaluate_detection
from tornotify.preprocess.cells import identify_cells
from tornotify.preprocess.chips import extract_chip
from tornotify.tracking import DetectionCandidate, TrackManager

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ScanProcessResult:
    site: str
    status: str
    scan_filename: str | None = None
    scan_time: datetime | None = None
    cells_found: int = 0
    chips_processed: int = 0
    raw_detections: int = 0
    pending_tracks: int = 0
    detections: int = 0
    image_path: str | None = None


def process_latest_scan(
    site: str,
    dt: datetime | None,
    processed_keys: set[str],
    csv_path: str = "data/results.csv",
    image_dir: str | None = "data/radar_images",
    threshold: float = CONFIDENCE_THRESHOLD,
    detection_csv_path: str | None = "data/detections.csv",
    track_manager: TrackManager | None = None,
) -> ScanProcessResult:
    """
    Fetch and process the newest unprocessed scan for a site.

    processed_keys is mutated after a scan is fully handled or skipped because
    it has no usable cells/chips.
    """
    scans = get_available_scans(site, dt)
    if not scans:
        logger.warning("No scans available for %s on %s", site, dt)
        return ScanProcessResult(site=site, status="no_scans")

    new_scan = next((scan for scan in reversed(scans) if scan.key not in processed_keys), None)
    if new_scan is None:
        logger.info("No new scans for %s", site)
        return ScanProcessResult(site=site, status="no_new_scan")

    logger.info("[%s] Processing scan: %s", site, new_scan.filename)

    with tempfile.TemporaryDirectory() as tmpdir:
        local_path = download_scan(new_scan, tmpdir)
        radar = parse_scan(local_path)

    scan_time = get_scan_time(radar)
    scan_key = f"{site}-{scan_time.strftime('%Y%m%d%H%M%S')}"

    cells = identify_cells(radar, scan_key)
    if not cells:
        logger.info("[%s] No storm cells in %s", site, new_scan.filename)
        if track_manager is not None:
            track_manager.update_scan(site, scan_time, [])
        processed_keys.add(new_scan.key)
        return ScanProcessResult(
            site=site,
            status="no_cells",
            scan_filename=new_scan.filename,
            scan_time=scan_time,
        )

    chips = []
    valid_cells = []
    for cell in cells:
        chip = extract_chip(radar, cell)
        if chip is not None:
            chips.append(chip)
            valid_cells.append(cell)

    if not chips:
        logger.info("[%s] No valid TorNet chips in %s", site, new_scan.filename)
        if track_manager is not None:
            track_manager.update_scan(site, scan_time, [])
        processed_keys.add(new_scan.key)
        return ScanProcessResult(
            site=site,
            status="no_chips",
            scan_filename=new_scan.filename,
            scan_time=scan_time,
            cells_found=len(cells),
        )

    detection_results = predict_batch_detailed(chips)
    probs = [result.probability for result in detection_results]
    decisions = [
        evaluate_detection(cell, result, threshold)
        for cell, result in zip(valid_cells, detection_results)
    ]
    candidates = [
        DetectionCandidate(
            site=site,
            scan_time=scan_time,
            cell=cell,
            detection_result=result,
            quality_decision=decision,
            threshold=threshold,
        )
        for cell, result, decision in zip(valid_cells, detection_results, decisions)
    ]
    raw_detections = sum(1 for candidate in candidates if candidate.above_threshold)
    tracking_updates = (
        track_manager.update_scan(site, scan_time, candidates)
        if track_manager is not None
        else []
    )
    tracking_by_cell = {
        update.candidate.cell.cell_id: update
        for update in tracking_updates
    }
    pending_tracks = sum(1 for update in tracking_updates if update.status == "pending")
    detections = 0

    for cell, result, decision in zip(valid_cells, detection_results, decisions):
        prob = result.probability
        logger.info("Cell %s -> P(tornado)=%.4f", cell.cell_id, prob)
        if result.heatmap_edge_margin is not None and result.heatmap_edge_margin < 0.10:
            logger.info(
                "Cell %s heatmap max is near chip edge: az=%.1f range=%.1fkm edge_margin=%.3f",
                cell.cell_id,
                result.activation_az_deg,
                result.activation_range_km,
                result.heatmap_edge_margin,
            )
        if decision.above_threshold and not decision.actionable:
            logger.info(
                "Cell %s flagged by detection quality gates: %s",
                cell.cell_id,
                "; ".join(decision.reasons),
            )
        tracking_update = tracking_by_cell.get(cell.cell_id)
        if tracking_update and tracking_update.track_id:
            logger.info(
                "Cell %s tracking: track=%s status=%s hits=%d/%d",
                cell.cell_id,
                tracking_update.track_id,
                tracking_update.status,
                tracking_update.hits_in_window,
                tracking_update.window_size,
            )
        log_result(
            csv_path,
            site,
            scan_time,
            cell,
            prob,
            threshold,
            above_threshold=decision.above_threshold,
        )

        if tracking_update and tracking_update.newly_confirmed:
            detections += 1
            if detection_csv_path:
                log_detection_event(detection_csv_path, tracking_update)

    if raw_detections and track_manager is None:
        logger.info(
            "[%s] %d raw over-threshold candidates; temporal tracking is disabled for this run.",
            site,
            raw_detections,
        )

    image_path = None
    if image_dir:
        try:
            from tornotify.visualization import save_marked_radar_image

            image_path = save_marked_radar_image(
                radar=radar,
                site=site,
                scan_time=scan_time,
                cells=valid_cells,
                probabilities=probs,
                detection_results=detection_results,
                detection_decisions=decisions,
                output_dir=image_dir,
                threshold=threshold,
            )
        except Exception as exc:
            logger.warning("[%s] Radar image logging failed: %s", site, exc)

    processed_keys.add(new_scan.key)
    logger.info(
        "[%s] Scan complete. %d cells, %d valid chips, %d raw candidates, %d actionable detections.",
        site,
        len(cells),
        len(chips),
        raw_detections,
        detections,
    )

    return ScanProcessResult(
        site=site,
        status="processed",
        scan_filename=new_scan.filename,
        scan_time=scan_time,
        cells_found=len(cells),
        chips_processed=len(chips),
        raw_detections=raw_detections,
        pending_tracks=pending_tracks,
        detections=detections,
        image_path=image_path,
    )
