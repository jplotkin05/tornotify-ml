"""
Adaptive multi-site radar scanning.

This module runs the existing single-scan pipeline behind a local worker pool.
Sites with active cells or raw/confirmed detections are rescheduled sooner,
while quiet sites and erroring sites back off so workers spend more time where
radar has signal.
"""
from __future__ import annotations

import heapq
import itertools
import logging
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Callable, Iterable

from tornotify.config import (
    CONFIDENCE_THRESHOLD,
    SCANNER_ACTIVE_POLL_SECONDS,
    SCANNER_DEFAULT_WORKERS,
    SCANNER_ERROR_BACKOFF_SECONDS,
    SCANNER_HOT_POLL_SECONDS,
    SCANNER_MAX_ERROR_BACKOFF_SECONDS,
    SCANNER_MAX_QUIET_POLL_SECONDS,
    SCANNER_NO_SCAN_POLL_SECONDS,
    SCANNER_QUIET_POLL_SECONDS,
)
from tornotify.pipeline import ScanProcessResult, process_latest_scan
from tornotify.tracking import TrackManager

logger = logging.getLogger(__name__)

ProcessScanFunc = Callable[..., ScanProcessResult]


@dataclass(frozen=True)
class ScannerConfig:
    """Tuning knobs for adaptive site scheduling."""

    workers: int = SCANNER_DEFAULT_WORKERS
    hot_poll_seconds: int = SCANNER_HOT_POLL_SECONDS
    active_poll_seconds: int = SCANNER_ACTIVE_POLL_SECONDS
    quiet_poll_seconds: int = SCANNER_QUIET_POLL_SECONDS
    max_quiet_poll_seconds: int = SCANNER_MAX_QUIET_POLL_SECONDS
    no_scan_poll_seconds: int = SCANNER_NO_SCAN_POLL_SECONDS
    error_backoff_seconds: int = SCANNER_ERROR_BACKOFF_SECONDS
    max_error_backoff_seconds: int = SCANNER_MAX_ERROR_BACKOFF_SECONDS
    quiet_backoff_multiplier: float = 1.5

    def __post_init__(self) -> None:
        if self.workers < 1:
            raise ValueError("workers must be >= 1")


@dataclass
class SiteScanState:
    """Mutable scheduling state for one radar site."""

    site: str
    processed_keys: set[str] = field(default_factory=set)
    attempts: int = 0
    scans_processed: int = 0
    activity_score: int = 0
    consecutive_quiet: int = 0
    consecutive_no_new: int = 0
    consecutive_errors: int = 0
    last_started_at: datetime | None = None
    last_finished_at: datetime | None = None
    next_due_at: datetime | None = None
    last_result: ScanProcessResult | None = None
    last_error: str | None = None
    track_manager: TrackManager = field(default_factory=TrackManager)


@dataclass(frozen=True)
class ScanAttempt:
    """Result of one worker attempt on one site."""

    site: str
    result: ScanProcessResult | None
    error: str | None
    delay_seconds: int
    priority: int
    reason: str


@dataclass(order=True)
class _ScheduledSite:
    due_at: float
    priority: int
    seq: int
    site: str = field(compare=False)


class AdaptiveSiteScanner:
    """
    Coordinate multi-site radar scanning with a bounded local worker pool.

    The scanner is intentionally process-local for now. Site tasks are
    independent, priority is derived from recent scan activity, and each site
    keeps its own processed scan key cache.
    """

    def __init__(
        self,
        sites: Iterable[str],
        *,
        csv_path: str = "data/results.csv",
        detection_csv_path: str | None = "data/detections.csv",
        image_dir: str | None = "data/radar_images",
        threshold: float = CONFIDENCE_THRESHOLD,
        config: ScannerConfig | None = None,
        process_func: ProcessScanFunc | None = None,
    ) -> None:
        self.sites = tuple(_dedupe_sites(sites))
        if not self.sites:
            raise ValueError("AdaptiveSiteScanner requires at least one site")

        self.csv_path = csv_path
        self.detection_csv_path = detection_csv_path
        self.image_dir = image_dir
        self.threshold = threshold
        self.config = config or ScannerConfig()
        self._process_func = process_func or process_latest_scan

        self.states = {site: SiteScanState(site=site) for site in self.sites}

        self._counter = itertools.count()
        self._heap: list[_ScheduledSite] = []
        self._condition = threading.Condition()
        self._stop_event = threading.Event()
        self._in_flight: set[str] = set()
        self._executor: ThreadPoolExecutor | None = None
        self._dispatcher: threading.Thread | None = None
        self._started = False
        self._scan_dt: datetime | None = None

    def run_once(self, dt: datetime | None = None) -> list[ScanAttempt]:
        """Process each configured site once using the worker pool, then exit."""
        logger.info(
            "Starting one multi-site radar sweep for %d sites with %d workers",
            len(self.sites),
            self.config.workers,
        )
        attempts: list[ScanAttempt] = []
        with ThreadPoolExecutor(
            max_workers=self.config.workers,
            thread_name_prefix="radar-worker",
        ) as executor:
            futures = [executor.submit(self._process_site, site, dt) for site in self.sites]
            for future in as_completed(futures):
                attempts.append(future.result())
        return attempts

    def start(self, dt: datetime | None = None) -> None:
        """Start the adaptive continuous scanner in the background."""
        with self._condition:
            if self._started:
                return

            self._stop_event.clear()
            self._scan_dt = dt
            self._executor = ThreadPoolExecutor(
                max_workers=self.config.workers,
                thread_name_prefix="radar-worker",
            )
            now = time.monotonic()
            for site in self.sites:
                self._schedule_locked(site, due_at=now, priority=0)

            self._dispatcher = threading.Thread(
                target=self._dispatch_loop,
                name="radar-dispatcher",
                daemon=True,
            )
            self._started = True
            self._dispatcher.start()

        logger.info(
            "Adaptive radar scanner started for %d sites with %d workers",
            len(self.sites),
            self.config.workers,
        )

    def run_forever(self, dt: datetime | None = None) -> None:
        """Run the adaptive scanner until interrupted."""
        self.start(dt)
        try:
            while not self._stop_event.is_set():
                time.sleep(1.0)
        except KeyboardInterrupt:
            logger.info("Stopping adaptive radar scanner")
        finally:
            self.stop()

    def stop(self, wait: bool = True) -> None:
        """Stop the continuous scanner and optionally wait for active workers."""
        self._stop_event.set()
        with self._condition:
            self._condition.notify_all()

        dispatcher = self._dispatcher
        if dispatcher and dispatcher is not threading.current_thread():
            dispatcher.join(timeout=5.0)

        executor = self._executor
        if executor:
            executor.shutdown(wait=wait, cancel_futures=not wait)

        with self._condition:
            self._heap.clear()
            self._in_flight.clear()
            self._executor = None
            self._dispatcher = None
            self._started = False

        logger.info("Adaptive radar scanner stopped")

    def _dispatch_loop(self) -> None:
        while not self._stop_event.is_set():
            site = self._next_due_site()
            if site is None:
                continue

            executor = self._executor
            if executor is None:
                return

            future = executor.submit(self._process_site, site, self._scan_dt)
            future.add_done_callback(lambda done, site=site: self._finish_site(site, done))

    def _next_due_site(self) -> str | None:
        with self._condition:
            while not self._stop_event.is_set():
                if not self._heap:
                    self._condition.wait(timeout=1.0)
                    continue

                item = self._heap[0]
                wait_seconds = item.due_at - time.monotonic()
                if wait_seconds > 0:
                    self._condition.wait(timeout=wait_seconds)
                    continue

                heapq.heappop(self._heap)
                if item.site in self._in_flight:
                    continue

                self._in_flight.add(item.site)
                return item.site
        return None

    def _finish_site(self, site: str, future: Future[ScanAttempt]) -> None:
        try:
            attempt = future.result()
        except Exception as exc:
            logger.exception("[%s] Worker crashed outside scan handling", site)
            state = self.states[site]
            state.consecutive_errors += 1
            state.last_error = str(exc)
            attempt = ScanAttempt(
                site=site,
                result=None,
                error=str(exc),
                delay_seconds=self._error_delay(state.consecutive_errors),
                priority=30,
                reason="worker_crash",
            )

        with self._condition:
            self._in_flight.discard(site)
            if not self._stop_event.is_set():
                due_at = time.monotonic() + attempt.delay_seconds
                self._schedule_locked(site, due_at=due_at, priority=attempt.priority)
                state = self.states[site]
                state.next_due_at = (
                    datetime.now(timezone.utc) + timedelta(seconds=attempt.delay_seconds)
                ).replace(microsecond=0)
                logger.info(
                    "[%s] Next scan in %ds (%s)",
                    site,
                    attempt.delay_seconds,
                    attempt.reason,
                )
            self._condition.notify_all()

    def _schedule_locked(self, site: str, *, due_at: float, priority: int) -> None:
        heapq.heappush(
            self._heap,
            _ScheduledSite(
                due_at=due_at,
                priority=priority,
                seq=next(self._counter),
                site=site,
            ),
        )
        self._condition.notify_all()

    def _process_site(self, site: str, dt: datetime | None) -> ScanAttempt:
        state = self.states[site]
        state.attempts += 1
        state.last_started_at = datetime.now(timezone.utc)
        scan_dt = dt or state.last_started_at

        logger.info("[%s] Worker scan attempt %d started", site, state.attempts)
        try:
            result = self._process_func(
                site,
                scan_dt,
                state.processed_keys,
                csv_path=self.csv_path,
                image_dir=self.image_dir,
                threshold=self.threshold,
                detection_csv_path=self.detection_csv_path,
                track_manager=state.track_manager,
            )
        except Exception as exc:
            state.consecutive_errors += 1
            state.last_error = str(exc)
            state.last_result = None
            state.last_finished_at = datetime.now(timezone.utc)
            delay = self._error_delay(state.consecutive_errors)
            logger.error(
                "[%s] Worker scan failed; retrying in %ds: %s",
                site,
                delay,
                exc,
                exc_info=True,
            )
            return ScanAttempt(
                site=site,
                result=None,
                error=str(exc),
                delay_seconds=delay,
                priority=30,
                reason="error_backoff",
            )

        state.last_finished_at = datetime.now(timezone.utc)
        state.last_result = result
        state.last_error = None
        state.consecutive_errors = 0
        if result.status == "processed":
            state.scans_processed += 1

        self._update_activity_state(state, result)
        delay, priority, reason = self._next_delay(state, result)
        return ScanAttempt(
            site=site,
            result=result,
            error=None,
            delay_seconds=delay,
            priority=priority,
            reason=reason,
        )

    def _update_activity_state(self, state: SiteScanState, result: ScanProcessResult) -> None:
        if result.status == "processed" and (result.detections > 0 or result.raw_detections > 0):
            state.activity_score = 3
            state.consecutive_quiet = 0
            state.consecutive_no_new = 0
        elif result.status == "processed" and (result.chips_processed > 0 or result.cells_found > 0):
            state.activity_score = 2
            state.consecutive_quiet = 0
            state.consecutive_no_new = 0
        elif result.status == "no_new_scan":
            state.consecutive_no_new += 1
            if state.consecutive_no_new >= 3 and state.activity_score > 0:
                state.activity_score -= 1
                state.consecutive_no_new = 0
        else:
            state.activity_score = 0
            state.consecutive_quiet += 1
            state.consecutive_no_new = 0

    def _next_delay(
        self,
        state: SiteScanState,
        result: ScanProcessResult,
    ) -> tuple[int, int, str]:
        if result.status == "no_scans":
            return self.config.no_scan_poll_seconds, 25, "no_scans_backoff"
        if state.activity_score >= 3:
            return self.config.hot_poll_seconds, 0, "hot_detection_candidate"
        if state.activity_score >= 2:
            return self.config.active_poll_seconds, 5, "active_cells"
        if result.status == "no_new_scan":
            return self.config.quiet_poll_seconds, 15, "waiting_for_new_scan"
        return self._quiet_delay(state.consecutive_quiet), 20, "quiet_backoff"

    def _quiet_delay(self, consecutive_quiet: int) -> int:
        steps = max(0, consecutive_quiet - 1)
        delay = self.config.quiet_poll_seconds * (self.config.quiet_backoff_multiplier ** steps)
        return int(min(delay, self.config.max_quiet_poll_seconds))

    def _error_delay(self, consecutive_errors: int) -> int:
        steps = max(0, consecutive_errors - 1)
        delay = self.config.error_backoff_seconds * (2 ** steps)
        return int(min(delay, self.config.max_error_backoff_seconds))


def _dedupe_sites(sites: Iterable[str]) -> list[str]:
    seen = set()
    result = []
    for site in sites:
        normalized = site.strip().upper()
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return result
