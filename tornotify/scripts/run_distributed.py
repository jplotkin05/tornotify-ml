"""
One-machine distributed scanner launcher.

This seeds Redis and starts local worker processes. For multi-machine runs, use
distributed_scheduler.py on one host and distributed_worker.py on each worker
host.
"""
import multiprocessing as mp
import os
import sys
import time

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_script_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

os.environ.setdefault("KERAS_BACKEND", "torch")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import argparse
import logging

from tornotify.config import (
    CONFIDENCE_THRESHOLD,
    DISTRIBUTED_DEFAULT_WORKERS,
    DISTRIBUTED_QUEUE_KEY,
    PRIMARY_SITE,
    RADAR_SITES,
    REDIS_URL,
)
from tornotify.distributed import redis_from_url, seed_sites, worker_loop

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("tornotify.distributed")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Redis-backed distributed radar scanning")
    parser.add_argument("--site", default=PRIMARY_SITE, help="NEXRAD site ID")
    parser.add_argument("--sites", nargs="+", help="One or more NEXRAD site IDs")
    parser.add_argument("--all-sites", action="store_true", help="Seed all known non-TDWR radar sites")
    parser.add_argument("--list-sites", action="store_true", help="Print known radar sites and exit")
    parser.add_argument("--workers", type=int, default=DISTRIBUTED_DEFAULT_WORKERS,
                        help=f"Local worker process count (default: {DISTRIBUTED_DEFAULT_WORKERS})")
    parser.add_argument("--redis-url", default=REDIS_URL, help="Redis URL")
    parser.add_argument("--queue-key", default=DISTRIBUTED_QUEUE_KEY, help="Redis sorted-set queue key")
    parser.add_argument("--csv", default=os.path.join(_project_root, "data", "results.csv"),
                        help="Path for raw cell result CSV")
    parser.add_argument("--detections-csv", default=os.path.join(_project_root, "data", "detections.csv"),
                        help="Path for confirmed track CSV")
    parser.add_argument("--image-dir", default=None,
                        help="Directory for marked radar PNGs; omitted by default for throughput")
    parser.add_argument("--threshold", type=float, default=None, help="Override confidence threshold")
    parser.add_argument("--requeue-now", action="store_true", help="Overwrite due times and queue sites now")
    parser.add_argument("--seed-only", action="store_true", help="Seed queue and exit without workers")
    parser.add_argument("--max-scans-per-worker", type=int, default=None,
                        help="Stop each worker after this many processed site attempts")
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

    if args.workers < 1:
        parser.error("--workers must be >= 1")

    sites = _resolve_sites(args, parser)
    redis_client = redis_from_url(args.redis_url)
    redis_client.ping()
    added = seed_sites(
        redis_client,
        sites,
        queue_key=args.queue_key,
        overwrite=args.requeue_now,
    )
    logger.info(
        "Seeded queue %s for %d sites; added/requeued %d",
        args.queue_key,
        len(sites),
        added,
    )
    if args.seed_only:
        return

    threshold = CONFIDENCE_THRESHOLD if args.threshold is None else args.threshold
    worker_kwargs = {
        "redis_url": args.redis_url,
        "queue_key": args.queue_key,
        "csv_path": args.csv,
        "detection_csv_path": args.detections_csv,
        "image_dir": args.image_dir,
        "threshold": threshold,
        "max_scans": args.max_scans_per_worker,
    }

    prefix = f"{os.uname().nodename}-{os.getpid()}"
    processes = []
    for index in range(args.workers):
        worker_id = f"{prefix}-{index + 1}"
        process = mp.Process(
            target=worker_loop,
            kwargs={"worker_id": worker_id, **worker_kwargs},
            name=f"radar-worker-{index + 1}",
        )
        process.start()
        processes.append(process)

    try:
        while any(process.is_alive() for process in processes):
            time.sleep(1.0)
    except KeyboardInterrupt:
        logger.info("Stopping distributed scanner")
        for process in processes:
            process.terminate()
    finally:
        for process in processes:
            process.join()


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
        parser.error("Unknown radar site(s): " + ", ".join(unknown))
    return normalized


if __name__ == "__main__":
    main()
