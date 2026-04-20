"""
Redis-backed distributed radar scanning.

Redis coordinates work only. Radar probabilities and confirmed track events
remain plain CSV artifacts so the scanner can scale without adding Postgres or
a service API.
"""
from __future__ import annotations

import json
import logging
import os
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Iterable, Sequence

from tornotify.config import (
    CONFIDENCE_THRESHOLD,
    DISTRIBUTED_PROCESSED_TTL_SECONDS,
    DISTRIBUTED_QUEUE_KEY,
    DISTRIBUTED_REDIS_PREFIX,
    DISTRIBUTED_SITE_LOCK_TTL_SECONDS,
    DISTRIBUTED_TRACK_TTL_SECONDS,
    DISTRIBUTED_WORKER_SLEEP_SECONDS,
    REDIS_URL,
    SCANNER_ACTIVE_POLL_SECONDS,
    SCANNER_ERROR_BACKOFF_SECONDS,
    SCANNER_HOT_POLL_SECONDS,
    SCANNER_MAX_ERROR_BACKOFF_SECONDS,
    SCANNER_MAX_QUIET_POLL_SECONDS,
    SCANNER_NO_SCAN_POLL_SECONDS,
    SCANNER_QUIET_POLL_SECONDS,
)
from tornotify.tracking import (
    DetectionCandidate,
    DetectionTrack,
    TrackObservation,
    TrackingConfig,
    TrackingUpdate,
    _activation_xy,
    _cell_xy,
    _distance_km,
    _hours_between,
    _last_activation_xy,
    _miss_observation,
    _observation_from_candidate,
)
if TYPE_CHECKING:
    from tornotify.pipeline import ScanProcessResult

logger = logging.getLogger(__name__)

_POP_DUE_SCRIPT = """
local item = redis.call('ZRANGEBYSCORE', KEYS[1], '-inf', ARGV[1], 'LIMIT', 0, 1)[1]
if not item then
    return nil
end
redis.call('ZREM', KEYS[1], item)
return item
"""

_RELEASE_LOCK_SCRIPT = """
if redis.call('GET', KEYS[1]) == ARGV[1] then
    return redis.call('DEL', KEYS[1])
end
return 0
"""


@dataclass
class DistributedSiteState:
    """Persisted scheduling state for one radar site."""

    site: str
    attempts: int = 0
    scans_processed: int = 0
    activity_score: int = 0
    consecutive_quiet: int = 0
    consecutive_no_new: int = 0
    consecutive_errors: int = 0
    last_status: str | None = None
    last_error: str | None = None
    last_finished_at: str | None = None


@dataclass(frozen=True)
class DistributedScheduleDecision:
    """Next queue placement after one worker attempt."""

    delay_seconds: int
    priority: int
    reason: str
    state: DistributedSiteState


class RedisProcessedKeys:
    """Set-like processed scan store backed by Redis."""

    def __init__(
        self,
        redis_client: Any,
        site: str,
        *,
        prefix: str = DISTRIBUTED_REDIS_PREFIX,
        ttl_seconds: int = DISTRIBUTED_PROCESSED_TTL_SECONDS,
    ) -> None:
        self.redis = redis_client
        self.site = site.upper()
        self.key = f"{prefix}:processed:{self.site}"
        self.ttl_seconds = ttl_seconds

    def __contains__(self, scan_key: object) -> bool:
        if not isinstance(scan_key, str):
            return False
        return bool(self.redis.sismember(self.key, scan_key))

    def add(self, scan_key: str) -> None:
        self.redis.sadd(self.key, scan_key)
        self.redis.expire(self.key, self.ttl_seconds)


class RedisTrackManager:
    """Track manager with per-site track history persisted in Redis."""

    def __init__(
        self,
        redis_client: Any,
        *,
        prefix: str = DISTRIBUTED_REDIS_PREFIX,
        ttl_seconds: int = DISTRIBUTED_TRACK_TTL_SECONDS,
        config: TrackingConfig | None = None,
    ) -> None:
        self.redis = redis_client
        self.prefix = prefix
        self.ttl_seconds = ttl_seconds
        self.config = config or TrackingConfig()

    def update_scan(
        self,
        site: str,
        scan_time: datetime,
        candidates: list[DetectionCandidate],
    ) -> list[TrackingUpdate]:
        tracks = self._load_tracks(site)
        updates_by_candidate: dict[int, TrackingUpdate] = {}

        pairs = self._candidate_track_pairs(candidates, tracks, scan_time)
        used_candidates: set[int] = set()
        used_tracks: set[int] = set()

        for cost, candidate_index, track_index in pairs:
            if candidate_index in used_candidates or track_index in used_tracks:
                continue

            candidate = candidates[candidate_index]
            track = tracks[track_index]
            was_confirmed = track.confirmed_at is not None
            track.observations.append(_observation_from_candidate(candidate))
            self._trim_track(track)
            newly_confirmed = self._confirm_track_if_ready(track, scan_time) and not was_confirmed
            hits, window_size = self._window_counts(track)
            updates_by_candidate[candidate_index] = TrackingUpdate(
                candidate=candidate,
                track=track,
                status=self._candidate_status(candidate, track, newly_confirmed),
                hits_in_window=hits,
                window_size=window_size,
                newly_confirmed=newly_confirmed,
            )
            used_candidates.add(candidate_index)
            used_tracks.add(track_index)
            logger.debug(
                "[%s] Candidate %s matched Redis track %s with cost %.1f",
                site,
                candidate.cell.cell_id,
                track.track_id,
                cost,
            )

        for track_index, track in enumerate(list(tracks)):
            if track_index in used_tracks:
                continue
            track.observations.append(_miss_observation(scan_time))
            self._trim_track(track)

        for candidate_index, candidate in enumerate(candidates):
            if candidate_index in used_candidates:
                continue
            if candidate.above_threshold:
                track = self._new_track(candidate)
                tracks.append(track)
                newly_confirmed = self._confirm_track_if_ready(track, scan_time)
                hits, window_size = self._window_counts(track)
                updates_by_candidate[candidate_index] = TrackingUpdate(
                    candidate=candidate,
                    track=track,
                    status=self._candidate_status(candidate, track, newly_confirmed),
                    hits_in_window=hits,
                    window_size=window_size,
                    newly_confirmed=newly_confirmed,
                )
                logger.info(
                    "[%s] Started Redis detection track %s from %s P=%.4f",
                    site,
                    track.track_id,
                    candidate.cell.cell_id,
                    candidate.probability,
                )
            else:
                updates_by_candidate[candidate_index] = TrackingUpdate(
                    candidate=candidate,
                    track=None,
                    status="below_threshold",
                )

        tracks = [
            track for track in tracks if track.missed_scans <= self.config.max_missed_scans
        ]
        self._save_tracks(site, tracks)
        return [updates_by_candidate[i] for i in range(len(candidates))]

    def _candidate_track_pairs(
        self,
        candidates: Sequence[DetectionCandidate],
        tracks: Sequence[DetectionTrack],
        scan_time: datetime,
    ) -> list[tuple[float, int, int]]:
        pairs: list[tuple[float, int, int]] = []
        for candidate_index, candidate in enumerate(candidates):
            candidate_xy = _cell_xy(candidate.cell)
            candidate_activation_xy = _activation_xy(candidate.detection_result)
            for track_index, track in enumerate(tracks):
                last_observation = track.last_observation
                if (
                    last_observation is None
                    or last_observation.x_km is None
                    or last_observation.y_km is None
                ):
                    continue

                predicted_x, predicted_y = self._predict_track_xy(track, scan_time)
                centroid_distance = _distance_km(candidate_xy, (predicted_x, predicted_y))
                max_distance = self._max_match_distance(track, scan_time)
                if centroid_distance > max_distance:
                    continue

                activation_distance = None
                last_activation = _last_activation_xy(track)
                if candidate_activation_xy is not None and last_activation is not None:
                    activation_distance = _distance_km(candidate_activation_xy, last_activation)
                    if activation_distance > self.config.activation_match_km:
                        continue

                cost = centroid_distance
                if activation_distance is not None:
                    cost += 0.35 * activation_distance
                pairs.append((cost, candidate_index, track_index))

        pairs.sort(key=lambda item: item[0])
        return pairs

    def _predict_track_xy(self, track: DetectionTrack, scan_time: datetime) -> tuple[float, float]:
        observations = [
            obs for obs in track.observations
            if not obs.missed and obs.x_km is not None and obs.y_km is not None
        ]
        last = observations[-1]
        if len(observations) < 2:
            return float(last.x_km), float(last.y_km)

        previous = observations[-2]
        prev_dt_hours = _hours_between(previous.scan_time, last.scan_time)
        next_dt_hours = _hours_between(last.scan_time, scan_time)
        if prev_dt_hours <= 0 or next_dt_hours <= 0:
            return float(last.x_km), float(last.y_km)

        vx = (float(last.x_km) - float(previous.x_km)) / prev_dt_hours
        vy = (float(last.y_km) - float(previous.y_km)) / prev_dt_hours
        return float(last.x_km) + vx * next_dt_hours, float(last.y_km) + vy * next_dt_hours

    def _max_match_distance(self, track: DetectionTrack, scan_time: datetime) -> float:
        last = track.last_observation
        if last is None:
            return self.config.match_base_km
        elapsed_hours = max(0.0, _hours_between(last.scan_time, scan_time))
        return self.config.match_base_km + self.config.max_speed_kmh * elapsed_hours

    def _new_track(self, candidate: DetectionCandidate) -> DetectionTrack:
        track = DetectionTrack(
            track_id=self._new_track_id(candidate.site, candidate.scan_time),
            site=candidate.site,
            created_at=candidate.scan_time,
        )
        track.observations.append(_observation_from_candidate(candidate))
        return track

    def _new_track_id(self, site: str, scan_time: datetime) -> str:
        counter_key = f"{self.prefix}:track_counter:{site.upper()}"
        sequence = int(self.redis.incr(counter_key))
        self.redis.expire(counter_key, self.ttl_seconds)
        return f"{site.upper()}-{scan_time.strftime('%Y%m%d%H%M%S')}-T{sequence:04d}"

    def _confirm_track_if_ready(self, track: DetectionTrack, scan_time: datetime) -> bool:
        if track.confirmed_at is not None:
            return False
        hits, _window_size = self._window_counts(track)
        if hits >= self.config.confirm_scans:
            track.confirmed_at = scan_time
            logger.warning(
                "[%s] Confirmed Redis detection track %s with %d/%d recent over-threshold scans",
                track.site,
                track.track_id,
                hits,
                self.config.confirm_window_scans,
            )
            return True
        return False

    def _window_counts(self, track: DetectionTrack) -> tuple[int, int]:
        window = track.observations[-self.config.confirm_window_scans:]
        hits = sum(1 for obs in window if obs.above_threshold)
        return hits, len(window)

    def _candidate_status(
        self,
        candidate: DetectionCandidate,
        track: DetectionTrack,
        newly_confirmed: bool,
    ) -> str:
        if newly_confirmed:
            return "newly_confirmed"
        if track.confirmed_at is not None:
            return "confirmed"
        if candidate.above_threshold:
            return "pending"
        return "tracked_below_threshold"

    def _trim_track(self, track: DetectionTrack) -> None:
        max_history = max(self.config.confirm_window_scans, self.config.max_missed_scans + 1) + 3
        if len(track.observations) > max_history:
            del track.observations[: len(track.observations) - max_history]

    def _tracks_key(self, site: str) -> str:
        return f"{self.prefix}:tracks:{site.upper()}"

    def _load_tracks(self, site: str) -> list[DetectionTrack]:
        payload = self.redis.get(self._tracks_key(site))
        if not payload:
            return []
        data = json.loads(payload)
        return [_track_from_dict(item) for item in data.get("tracks", [])]

    def _save_tracks(self, site: str, tracks: Sequence[DetectionTrack]) -> None:
        key = self._tracks_key(site)
        payload = json.dumps({"tracks": [_track_to_dict(track) for track in tracks]})
        self.redis.set(key, payload, ex=self.ttl_seconds)


def redis_from_url(redis_url: str | None = None) -> Any:
    """Create a Redis client with string responses."""
    try:
        import redis
    except ImportError as exc:  # pragma: no cover - dependency message path
        raise RuntimeError(
            "Redis support requires the 'redis' package. Install tornotify/requirements.txt."
        ) from exc
    return redis.Redis.from_url(redis_url or REDIS_URL, decode_responses=True)


def enqueue_site(
    redis_client: Any,
    site: str,
    *,
    due_at: float | None = None,
    priority: int = 10,
    queue_key: str = DISTRIBUTED_QUEUE_KEY,
) -> None:
    """Put a site in the due-time priority queue."""
    due_time = time.time() if due_at is None else due_at
    score = due_time + (max(0, priority) * 0.001)
    redis_client.zadd(queue_key, {_normalize_site(site): score})


def seed_sites(
    redis_client: Any,
    sites: Iterable[str],
    *,
    queue_key: str = DISTRIBUTED_QUEUE_KEY,
    overwrite: bool = False,
) -> int:
    """Seed missing sites into the queue without disturbing active due times."""
    now = time.time()
    added = 0
    for site in _dedupe_sites(sites):
        if overwrite or redis_client.zscore(queue_key, site) is None:
            redis_client.zadd(queue_key, {site: now})
            added += 1
    return added


def pop_due_site(
    redis_client: Any,
    *,
    queue_key: str = DISTRIBUTED_QUEUE_KEY,
    now: float | None = None,
) -> str | None:
    """Atomically pop one due site from the queue."""
    site = redis_client.eval(_POP_DUE_SCRIPT, 1, queue_key, now or time.time())
    return None if site is None else str(site)


def acquire_site_lock(
    redis_client: Any,
    site: str,
    *,
    prefix: str = DISTRIBUTED_REDIS_PREFIX,
    ttl_seconds: int = DISTRIBUTED_SITE_LOCK_TTL_SECONDS,
) -> str | None:
    """Take the per-site processing lock and return its token."""
    token = f"{os.getpid()}-{uuid.uuid4()}"
    ok = redis_client.set(_site_lock_key(site, prefix), token, nx=True, ex=ttl_seconds)
    return token if ok else None


def release_site_lock(
    redis_client: Any,
    site: str,
    token: str,
    *,
    prefix: str = DISTRIBUTED_REDIS_PREFIX,
) -> None:
    """Release a site lock only if this worker still owns it."""
    redis_client.eval(_RELEASE_LOCK_SCRIPT, 1, _site_lock_key(site, prefix), token)


def process_distributed_site_once(
    redis_client: Any,
    site: str,
    dt: datetime | None = None,
    *,
    csv_path: str = "data/results.csv",
    detection_csv_path: str | None = "data/detections.csv",
    image_dir: str | None = "data/radar_images",
    threshold: float = CONFIDENCE_THRESHOLD,
    prefix: str = DISTRIBUTED_REDIS_PREFIX,
) -> ScanProcessResult:
    """Run the shared scan pipeline with Redis-backed scan and track state."""
    from tornotify.pipeline import process_latest_scan

    processed_keys = RedisProcessedKeys(redis_client, site, prefix=prefix)
    track_manager = RedisTrackManager(redis_client, prefix=prefix)
    return process_latest_scan(
        site,
        dt,
        processed_keys,
        csv_path=csv_path,
        image_dir=image_dir,
        threshold=threshold,
        detection_csv_path=detection_csv_path,
        track_manager=track_manager,
    )


def worker_loop(
    *,
    worker_id: str,
    redis_url: str | None = None,
    queue_key: str = DISTRIBUTED_QUEUE_KEY,
    csv_path: str = "data/results.csv",
    detection_csv_path: str | None = "data/detections.csv",
    image_dir: str | None = "data/radar_images",
    threshold: float = CONFIDENCE_THRESHOLD,
    sleep_seconds: float = DISTRIBUTED_WORKER_SLEEP_SECONDS,
    once: bool = False,
    max_scans: int | None = None,
    prefix: str = DISTRIBUTED_REDIS_PREFIX,
) -> int:
    """Continuously claim due sites, process one scan, and reschedule them."""
    redis_client = redis_from_url(redis_url)
    processed_count = 0
    logger.info("Distributed worker %s started on queue %s", worker_id, queue_key)

    while True:
        site = pop_due_site(redis_client, queue_key=queue_key)
        if site is None:
            if once:
                return processed_count
            time.sleep(sleep_seconds)
            continue

        token = acquire_site_lock(redis_client, site, prefix=prefix)
        if token is None:
            enqueue_site(
                redis_client,
                site,
                due_at=time.time() + sleep_seconds,
                priority=10,
                queue_key=queue_key,
            )
            if once:
                return processed_count
            continue

        decision: DistributedScheduleDecision | None = None
        try:
            mark_attempt_started(redis_client, site, prefix=prefix)
            logger.info("[%s] Worker %s processing due site", site, worker_id)
            result = process_distributed_site_once(
                redis_client,
                site,
                datetime.now(timezone.utc),
                csv_path=csv_path,
                detection_csv_path=detection_csv_path,
                image_dir=image_dir,
                threshold=threshold,
                prefix=prefix,
            )
            processed_count += 1
            decision = update_schedule_after_result(redis_client, site, result, prefix=prefix)
            logger.info(
                "[%s] Worker %s finished status=%s; next scan in %ds (%s)",
                site,
                worker_id,
                result.status,
                decision.delay_seconds,
                decision.reason,
            )
        except Exception as exc:
            logger.exception("[%s] Worker %s failed", site, worker_id)
            decision = update_schedule_after_error(redis_client, site, str(exc), prefix=prefix)
        finally:
            if decision is None:
                decision = update_schedule_after_error(
                    redis_client,
                    site,
                    "worker exited without schedule decision",
                    prefix=prefix,
                )
            enqueue_site(
                redis_client,
                site,
                due_at=time.time() + decision.delay_seconds,
                priority=decision.priority,
                queue_key=queue_key,
            )
            release_site_lock(redis_client, site, token, prefix=prefix)

        if once or (max_scans is not None and processed_count >= max_scans):
            return processed_count


def mark_attempt_started(
    redis_client: Any,
    site: str,
    *,
    prefix: str = DISTRIBUTED_REDIS_PREFIX,
) -> DistributedSiteState:
    state = load_site_state(redis_client, site, prefix=prefix)
    state.attempts += 1
    save_site_state(redis_client, state, prefix=prefix)
    return state


def update_schedule_after_result(
    redis_client: Any,
    site: str,
    result: ScanProcessResult,
    *,
    prefix: str = DISTRIBUTED_REDIS_PREFIX,
) -> DistributedScheduleDecision:
    state = load_site_state(redis_client, site, prefix=prefix)
    state.last_status = result.status
    state.last_error = None
    state.consecutive_errors = 0
    state.last_finished_at = datetime.now(timezone.utc).isoformat()
    if result.status == "processed":
        state.scans_processed += 1

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

    delay, priority, reason = _next_delay(state, result)
    save_site_state(redis_client, state, prefix=prefix)
    return DistributedScheduleDecision(delay, priority, reason, state)


def update_schedule_after_error(
    redis_client: Any,
    site: str,
    error: str,
    *,
    prefix: str = DISTRIBUTED_REDIS_PREFIX,
) -> DistributedScheduleDecision:
    state = load_site_state(redis_client, site, prefix=prefix)
    state.consecutive_errors += 1
    state.last_error = error
    state.last_status = "error"
    state.last_finished_at = datetime.now(timezone.utc).isoformat()
    delay = _error_delay(state.consecutive_errors)
    save_site_state(redis_client, state, prefix=prefix)
    return DistributedScheduleDecision(delay, 30, "error_backoff", state)


def load_site_state(
    redis_client: Any,
    site: str,
    *,
    prefix: str = DISTRIBUTED_REDIS_PREFIX,
) -> DistributedSiteState:
    normalized = _normalize_site(site)
    payload = redis_client.get(_site_state_key(normalized, prefix))
    if not payload:
        return DistributedSiteState(site=normalized)
    data = json.loads(payload)
    return DistributedSiteState(
        site=normalized,
        attempts=int(data.get("attempts", 0)),
        scans_processed=int(data.get("scans_processed", 0)),
        activity_score=int(data.get("activity_score", 0)),
        consecutive_quiet=int(data.get("consecutive_quiet", 0)),
        consecutive_no_new=int(data.get("consecutive_no_new", 0)),
        consecutive_errors=int(data.get("consecutive_errors", 0)),
        last_status=data.get("last_status"),
        last_error=data.get("last_error"),
        last_finished_at=data.get("last_finished_at"),
    )


def save_site_state(
    redis_client: Any,
    state: DistributedSiteState,
    *,
    prefix: str = DISTRIBUTED_REDIS_PREFIX,
) -> None:
    redis_client.set(_site_state_key(state.site, prefix), json.dumps(state.__dict__))


def _next_delay(
    state: DistributedSiteState,
    result: ScanProcessResult,
) -> tuple[int, int, str]:
    if result.status == "no_scans":
        return SCANNER_NO_SCAN_POLL_SECONDS, 25, "no_scans_backoff"
    if state.activity_score >= 3:
        return SCANNER_HOT_POLL_SECONDS, 0, "hot_detection_candidate"
    if state.activity_score >= 2:
        return SCANNER_ACTIVE_POLL_SECONDS, 5, "active_cells"
    if result.status == "no_new_scan":
        return SCANNER_QUIET_POLL_SECONDS, 15, "waiting_for_new_scan"
    return _quiet_delay(state.consecutive_quiet), 20, "quiet_backoff"


def _quiet_delay(consecutive_quiet: int) -> int:
    steps = max(0, consecutive_quiet - 1)
    delay = SCANNER_QUIET_POLL_SECONDS * (1.5 ** steps)
    return int(min(delay, SCANNER_MAX_QUIET_POLL_SECONDS))


def _error_delay(consecutive_errors: int) -> int:
    steps = max(0, consecutive_errors - 1)
    delay = SCANNER_ERROR_BACKOFF_SECONDS * (2 ** steps)
    return int(min(delay, SCANNER_MAX_ERROR_BACKOFF_SECONDS))


def _track_to_dict(track: DetectionTrack) -> dict[str, Any]:
    return {
        "track_id": track.track_id,
        "site": track.site,
        "created_at": track.created_at.isoformat(),
        "confirmed_at": None if track.confirmed_at is None else track.confirmed_at.isoformat(),
        "observations": [_observation_to_dict(obs) for obs in track.observations],
    }


def _track_from_dict(data: dict[str, Any]) -> DetectionTrack:
    track = DetectionTrack(
        track_id=str(data["track_id"]),
        site=str(data["site"]),
        created_at=datetime.fromisoformat(data["created_at"]),
    )
    confirmed_at = data.get("confirmed_at")
    if confirmed_at:
        track.confirmed_at = datetime.fromisoformat(confirmed_at)
    track.observations = [
        _observation_from_dict(item) for item in data.get("observations", [])
    ]
    return track


def _observation_to_dict(observation: TrackObservation) -> dict[str, Any]:
    return {
        "scan_time": observation.scan_time.isoformat(),
        "cell_id": observation.cell_id,
        "probability": observation.probability,
        "above_threshold": observation.above_threshold,
        "x_km": observation.x_km,
        "y_km": observation.y_km,
        "activation_x_km": observation.activation_x_km,
        "activation_y_km": observation.activation_y_km,
        "quality_pass": observation.quality_pass,
        "quality_reasons": list(observation.quality_reasons),
        "missed": observation.missed,
    }


def _observation_from_dict(data: dict[str, Any]) -> TrackObservation:
    return TrackObservation(
        scan_time=datetime.fromisoformat(data["scan_time"]),
        cell_id=data.get("cell_id"),
        probability=float(data.get("probability", 0.0)),
        above_threshold=bool(data.get("above_threshold", False)),
        x_km=_optional_float(data.get("x_km")),
        y_km=_optional_float(data.get("y_km")),
        activation_x_km=_optional_float(data.get("activation_x_km")),
        activation_y_km=_optional_float(data.get("activation_y_km")),
        quality_pass=bool(data.get("quality_pass", False)),
        quality_reasons=tuple(data.get("quality_reasons", ())),
        missed=bool(data.get("missed", False)),
    )


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _site_lock_key(site: str, prefix: str) -> str:
    return f"{prefix}:site_lock:{_normalize_site(site)}"


def _site_state_key(site: str, prefix: str) -> str:
    return f"{prefix}:site_state:{_normalize_site(site)}"


def _normalize_site(site: str) -> str:
    return site.strip().upper()


def _dedupe_sites(sites: Iterable[str]) -> list[str]:
    seen = set()
    result = []
    for site in sites:
        normalized = _normalize_site(site)
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return result
