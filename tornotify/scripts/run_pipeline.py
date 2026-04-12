"""
TorNotify pipeline runner.

Usage:
    python scripts/run_pipeline.py [--site KLWX] [--date YYYY-MM-DD] [--loop]
    python scripts/run_pipeline.py --sites KMAF KTWX KTLX --loop --workers 3
    python scripts/run_pipeline.py --all-sites --loop --workers 4

Fetches the latest NEXRAD scan, identifies storm cells, runs inference,
logs raw cell probabilities, and records temporally confirmed detection tracks.
"""
import os
import sys

# Ensure project root is on sys.path so `tornotify` package is importable
# regardless of working directory.
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_script_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

os.environ.setdefault("KERAS_BACKEND", "torch")

import argparse
import logging
from datetime import datetime, timezone

from tornotify.config import (
    PRIMARY_SITE,
    RADAR_SITES,
    SCANNER_DEFAULT_WORKERS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("tornotify.pipeline")


def main():
    parser = argparse.ArgumentParser(description="TorNotify Pipeline")
    parser.add_argument("--site", default=PRIMARY_SITE, help="NEXRAD site ID (default: KLWX)")
    parser.add_argument("--sites", nargs="+", help="One or more NEXRAD site IDs to scan with workers")
    parser.add_argument("--all-sites", action="store_true", help="Scan all known non-TDWR radar sites")
    parser.add_argument("--list-sites", action="store_true", help="Print known radar sites and exit")
    parser.add_argument("--workers", type=int, default=SCANNER_DEFAULT_WORKERS,
                        help=f"Worker count for multi-site/adaptive scanning (default: {SCANNER_DEFAULT_WORKERS})")
    parser.add_argument("--date", default=None, help="Date to process (YYYY-MM-DD), default: today")
    parser.add_argument("--loop", action="store_true", help="Run continuously with adaptive site scheduling")
    parser.add_argument("--csv", default=os.path.join(_project_root, "data", "results.csv"),
                        help="Path for CSV results log (default: data/results.csv)")
    parser.add_argument("--detections-csv", default=os.path.join(_project_root, "data", "detections.csv"),
                        help="Path for confirmed detection track log (default: data/detections.csv)")
    parser.add_argument("--image-dir", default=os.path.join(_project_root, "data", "radar_images"),
                        help="Directory for marked radar PNGs (default: data/radar_images)")
    parser.add_argument("--no-images", action="store_true", help="Disable marked radar image logging")
    args = parser.parse_args()

    if args.list_sites:
        for site, metadata in RADAR_SITES.items():
            print(
                f"{site}\t{metadata.get('name', site)}\t"
                f"{metadata.get('lat'):.5f}\t{metadata.get('lon'):.5f}\t"
                f"{metadata.get('elev_ft')}"
            )
        print(f"Total sites: {len(RADAR_SITES)}")
        return

    dt = None
    if args.date:
        dt = datetime.strptime(args.date, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    sites = _resolve_sites(args, parser)
    image_dir = None if args.no_images else args.image_dir

    if args.loop or len(sites) > 1:
        from tornotify.scanner import AdaptiveSiteScanner, ScannerConfig

        scanner = AdaptiveSiteScanner(
            sites=sites,
            csv_path=args.csv,
            detection_csv_path=args.detections_csv,
            image_dir=image_dir,
            config=ScannerConfig(workers=args.workers),
        )
        if args.loop:
            logger.info("Starting adaptive scanner for sites: %s", ", ".join(sites))
            scanner.run_forever(dt)
        else:
            scanner.run_once(dt)
        return

    from tornotify.pipeline import process_latest_scan

    processed_keys: set[str] = set()
    site = sites[0]
    process_latest_scan(
        site,
        dt,
        processed_keys,
        csv_path=args.csv,
        image_dir=image_dir,
        detection_csv_path=args.detections_csv,
    )


def _resolve_sites(args: argparse.Namespace, parser: argparse.ArgumentParser) -> list[str]:
    if args.all_sites and args.sites:
        parser.error("--all-sites cannot be combined with --sites")

    if args.all_sites:
        sites = list(RADAR_SITES.keys())
    elif args.sites:
        sites = args.sites
    else:
        sites = [args.site]

    normalized = []
    seen = set()
    for site in sites:
        value = site.strip().upper()
        if value and value not in seen:
            seen.add(value)
            normalized.append(value)

    unknown = [site for site in normalized if site not in RADAR_SITES]
    if unknown:
        parser.error(
            "Unknown radar site(s): "
            + ", ".join(unknown)
            + ". Add site metadata to tornotify.config.RADAR_SITES first."
        )

    return normalized


if __name__ == "__main__":
    main()
