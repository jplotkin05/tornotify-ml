"""
Re-process a specific NEXRAD scan by filename. Used to verify filter changes
against a known case.

Usage:
    python scripts/reprocess_scan.py --site KMAF --filename KMAF20260411_034638_V06
"""
import os
import sys

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_script_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

os.environ.setdefault("KERAS_BACKEND", "torch")

import argparse
import logging
import tempfile
from datetime import datetime, timezone

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("tornotify.reprocess")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--site", required=True)
    parser.add_argument("--filename", required=True,
                        help="Exact NEXRAD scan filename, e.g. KMAF20260411_034638_V06")
    parser.add_argument("--image-dir", default=os.path.join(_project_root, "data", "radar_images_postfilter"))
    args = parser.parse_args()

    # Parse date from filename: SSSSYYYYMMDD_HHMMSS_V06
    site_prefix = args.filename[:4]
    date_part = args.filename[4:12]
    dt = datetime.strptime(date_part, "%Y%m%d").replace(tzinfo=timezone.utc)

    from tornotify.config import CONFIDENCE_THRESHOLD
    from tornotify.ingest.nexrad import download_scan, get_available_scans, get_scan_time, parse_scan
    from tornotify.ml.detector import predict_batch_detailed
    from tornotify.ml.quality import evaluate_detection
    from tornotify.preprocess.cells import identify_cells
    from tornotify.preprocess.chips import extract_chip
    from tornotify.visualization import save_marked_radar_image

    logger.info("Fetching scan list for %s on %s", args.site, dt.date())
    scans = get_available_scans(args.site, dt)
    match = next((s for s in scans if s.filename == args.filename), None)
    if match is None:
        logger.error("Scan %s not found in %d available scans", args.filename, len(scans))
        sys.exit(1)

    with tempfile.TemporaryDirectory() as tmpdir:
        local_path = download_scan(match, tmpdir)
        radar = parse_scan(local_path)

    scan_time = get_scan_time(radar)
    scan_key = f"{args.site}-{scan_time.strftime('%Y%m%d%H%M%S')}"

    cells = identify_cells(radar, scan_key)
    logger.info("Identified %d cells after near-range filter", len(cells))
    for c in cells:
        logger.info("  %s: az=%.1f° range=%.1f km dBZ_max=%.0f",
                    c.cell_id, c.az_deg, c.range_km, c.dbz_max)

    if not cells:
        logger.info("No cells to score. Done.")
        return

    chips = []
    valid_cells = []
    for cell in cells:
        chip = extract_chip(radar, cell)
        if chip is not None:
            chips.append(chip)
            valid_cells.append(cell)

    if not chips:
        logger.info("No valid chips. Done.")
        return

    detection_results = predict_batch_detailed(chips)
    probs = [result.probability for result in detection_results]
    decisions = [
        evaluate_detection(cell, result, CONFIDENCE_THRESHOLD)
        for cell, result in zip(valid_cells, detection_results)
    ]
    for cell, result, decision in zip(valid_cells, detection_results, decisions):
        prob = result.probability
        status = "quality-pass" if decision.actionable else "quality-gated" if decision.above_threshold else "below"
        logger.info("  %s: P(tornado)=%.4f [%s]", cell.cell_id, prob, status)
        if decision.reasons:
            logger.info("    gate reasons: %s", "; ".join(decision.reasons))
        if result.activation_az_deg is not None:
            logger.info(
                "    model max: az=%.1f° range=%.1f km edge_margin=%.3f",
                result.activation_az_deg,
                result.activation_range_km,
                result.heatmap_edge_margin,
            )

    image_path = save_marked_radar_image(
        radar=radar,
        site=args.site,
        scan_time=scan_time,
        cells=valid_cells,
        probabilities=probs,
        detection_results=detection_results,
        detection_decisions=decisions,
        output_dir=args.image_dir,
        threshold=CONFIDENCE_THRESHOLD,
    )
    logger.info("Wrote %s", image_path)


if __name__ == "__main__":
    main()
