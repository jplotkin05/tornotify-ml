"""
Run one or more distributed radar scan workers.

Workers claim due radar sites from Redis, run the shared scan pipeline, append
CSV outputs, update Redis tracking state, and reschedule each site.
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
    DISTRIBUTED_WORKER_SLEEP_SECONDS,
    REDIS_URL,
)
from tornotify.distributed import redis_from_url, worker_loop

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("tornotify.distributed.worker")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run distributed radar scan workers")
    parser.add_argument("--workers", type=int, default=DISTRIBUTED_DEFAULT_WORKERS,
                        help=f"Worker process count (default: {DISTRIBUTED_DEFAULT_WORKERS})")
    parser.add_argument("--redis-url", default=REDIS_URL, help="Redis URL")
    parser.add_argument("--queue-key", default=DISTRIBUTED_QUEUE_KEY, help="Redis sorted-set queue key")
    parser.add_argument("--csv", default=os.path.join(_project_root, "data", "results.csv"),
                        help="Path for raw cell result CSV")
    parser.add_argument("--detections-csv", default=os.path.join(_project_root, "data", "detections.csv"),
                        help="Path for confirmed track CSV")
    parser.add_argument("--image-dir", default=None,
                        help="Directory for marked radar PNGs; omitted by default for throughput")
    parser.add_argument("--threshold", type=float, default=None, help="Override confidence threshold")
    parser.add_argument("--sleep", type=float, default=DISTRIBUTED_WORKER_SLEEP_SECONDS,
                        help="Idle sleep when no site is due")
    parser.add_argument("--once", action="store_true", help="Each worker processes at most one due site")
    parser.add_argument("--max-scans-per-worker", type=int, default=None,
                        help="Stop each worker after this many processed site attempts")
    parser.add_argument("--worker-id-prefix", default=None, help="Worker ID prefix for logs")
    args = parser.parse_args()

    if args.workers < 1:
        parser.error("--workers must be >= 1")

    redis_client = redis_from_url(args.redis_url)
    redis_client.ping()

    threshold = CONFIDENCE_THRESHOLD if args.threshold is None else args.threshold
    prefix = args.worker_id_prefix or f"{os.uname().nodename}-{os.getpid()}"
    worker_kwargs = {
        "redis_url": args.redis_url,
        "queue_key": args.queue_key,
        "csv_path": args.csv,
        "detection_csv_path": args.detections_csv,
        "image_dir": args.image_dir,
        "threshold": threshold,
        "sleep_seconds": args.sleep,
        "once": args.once,
        "max_scans": args.max_scans_per_worker,
    }

    if args.workers == 1:
        worker_loop(worker_id=f"{prefix}-1", **worker_kwargs)
        return

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
        logger.info("Stopping distributed workers")
        for process in processes:
            process.terminate()
    finally:
        for process in processes:
            process.join()


if __name__ == "__main__":
    main()
