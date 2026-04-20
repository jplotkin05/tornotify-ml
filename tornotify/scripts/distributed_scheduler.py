"""
Seed and maintain the Redis radar-site queue.

Run this once before workers, or keep it looping as a lightweight queue
reconciler while distributed workers run elsewhere.
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
import time

from tornotify.config import (
    DISTRIBUTED_QUEUE_KEY,
    PRIMARY_SITE,
    RADAR_SITES,
    REDIS_URL,
)
from tornotify.distributed import redis_from_url, seed_sites

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("tornotify.distributed.scheduler")


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed the distributed radar scan queue")
    parser.add_argument("--site", default=PRIMARY_SITE, help="NEXRAD site ID")
    parser.add_argument("--sites", nargs="+", help="One or more NEXRAD site IDs")
    parser.add_argument("--all-sites", action="store_true", help="Seed all known non-TDWR radar sites")
    parser.add_argument("--list-sites", action="store_true", help="Print known radar sites and exit")
    parser.add_argument("--redis-url", default=REDIS_URL, help="Redis URL")
    parser.add_argument("--queue-key", default=DISTRIBUTED_QUEUE_KEY, help="Redis sorted-set queue key")
    parser.add_argument("--loop", action="store_true", help="Keep ensuring sites exist in the queue")
    parser.add_argument("--interval", type=float, default=60.0, help="Queue reconcile interval in seconds")
    parser.add_argument("--requeue-now", action="store_true", help="Overwrite due times and queue sites now")
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

    sites = _resolve_sites(args, parser)
    redis_client = redis_from_url(args.redis_url)
    redis_client.ping()

    while True:
        added = seed_sites(
            redis_client,
            sites,
            queue_key=args.queue_key,
            overwrite=args.requeue_now,
        )
        logger.info(
            "Queue %s reconciled for %d sites; added/requeued %d",
            args.queue_key,
            len(sites),
            added,
        )
        if not args.loop:
            return
        time.sleep(args.interval)


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
