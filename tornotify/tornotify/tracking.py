"""
Forward temporal tracking for raw tornado-score candidates.

The tracker is intentionally process-local. It associates scored cells from one
scan to the next and confirms a detection only when the same track is elevated
across the configured recent-scan window.
"""
from __future__ import annotations

import itertools
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

from tornotify.config import (
    CONFIDENCE_THRESHOLD,
    TRACK_ACTIVATION_MATCH_KM,
    TRACK_CONFIRM_SCANS,
    TRACK_CONFIRM_WINDOW_SCANS,
    TRACK_MATCH_BASE_KM,
    TRACK_MAX_MISSED_SCANS,
    TRACK_MAX_SPEED_KMH,
)
if TYPE_CHECKING:
    from tornotify.preprocess.cells import StormCell

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrackingConfig:
    """Tuning knobs for frame-to-frame candidate association."""

    threshold: float = CONFIDENCE_THRESHOLD
    confirm_scans: int = TRACK_CONFIRM_SCANS
    confirm_window_scans: int = TRACK_CONFIRM_WINDOW_SCANS
    match_base_km: float = TRACK_MATCH_BASE_KM
    max_speed_kmh: float = TRACK_MAX_SPEED_KMH
    activation_match_km: float = TRACK_ACTIVATION_MATCH_KM
    max_missed_scans: int = TRACK_MAX_MISSED_SCANS

    def __post_init__(self) -> None:
        if self.confirm_scans < 1:
            raise ValueError("confirm_scans must be >= 1")
        if self.confirm_window_scans < self.confirm_scans:
            raise ValueError("confirm_window_scans must be >= confirm_scans")
        if self.max_missed_scans < 0:
            raise ValueError("max_missed_scans must be >= 0")


@dataclass(frozen=True)
class DetectionCandidate:
    """One model-scored cell from one radar scan."""

    site: str
    scan_time: datetime
    cell: StormCell
    detection_result: Any
    quality_decision: Any | None = None
    threshold: float = CONFIDENCE_THRESHOLD

    @property
    def probability(self) -> float:
        return float(getattr(self.detection_result, "probability", self.detection_result))

    @property
    def above_threshold(self) -> bool:
        return self.probability >= self.threshold

    @property
    def quality_pass(self) -> bool:
        if self.quality_decision is None:
            return True
        return bool(getattr(self.quality_decision, "actionable", True))

    @property
    def quality_reasons(self) -> tuple[str, ...]:
        if self.quality_decision is None:
            return ()
        return tuple(getattr(self.quality_decision, "reasons", ()))


@dataclass(frozen=True)
class TrackObservation:
    """One scan's contribution to a track history."""

    scan_time: datetime
    cell_id: str | None
    probability: float
    above_threshold: bool
    x_km: float | None
    y_km: float | None
    activation_x_km: float | None
    activation_y_km: float | None
    quality_pass: bool
    quality_reasons: tuple[str, ...] = ()
    missed: bool = False


@dataclass
class DetectionTrack:
    """Persistent candidate track for a single radar site."""

    track_id: str
    site: str
    created_at: datetime
    observations: list[TrackObservation] = field(default_factory=list)
    confirmed_at: datetime | None = None

    @property
    def last_observation(self) -> TrackObservation | None:
        for observation in reversed(self.observations):
            if not observation.missed:
                return observation
        return None

    @property
    def last_scan_time(self) -> datetime:
        if self.observations:
            return self.observations[-1].scan_time
        return self.created_at

    @property
    def missed_scans(self) -> int:
        count = 0
        for observation in reversed(self.observations):
            if not observation.missed:
                break
            count += 1
        return count

    @property
    def max_probability(self) -> float:
        values = [obs.probability for obs in self.observations if not obs.missed]
        return max(values, default=0.0)

    @property
    def observations_count(self) -> int:
        return sum(1 for obs in self.observations if not obs.missed)


@dataclass(frozen=True)
class TrackingUpdate:
    """Association result for one current-scan candidate."""

    candidate: DetectionCandidate
    track: DetectionTrack | None
    status: str
    hits_in_window: int = 0
    window_size: int = 0
    newly_confirmed: bool = False

    @property
    def track_id(self) -> str | None:
        return None if self.track is None else self.track.track_id

    @property
    def confirmed(self) -> bool:
        return self.track is not None and self.track.confirmed_at is not None


class TrackManager:
    """Associate raw model candidates across scans for one or more sites."""

    def __init__(self, config: TrackingConfig | None = None) -> None:
        self.config = config or TrackingConfig()
        self._tracks_by_site: dict[str, list[DetectionTrack]] = {}
        self._counter = itertools.count(1)

    def update_scan(
        self,
        site: str,
        scan_time: datetime,
        candidates: list[DetectionCandidate],
    ) -> list[TrackingUpdate]:
        """
        Update tracks with one scan's candidates.

        All candidates may match existing tracks, but only raw over-threshold
        candidates start new tracks. A track is confirmed when it has enough
        over-threshold observations inside the recent observation window.
        """
        tracks = self._tracks_by_site.setdefault(site, [])
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
                "[%s] Candidate %s matched track %s with cost %.1f",
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
                    "[%s] Started detection track %s from %s P=%.4f",
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

        self._tracks_by_site[site] = [
            track for track in tracks if track.missed_scans <= self.config.max_missed_scans
        ]
        return [updates_by_candidate[i] for i in range(len(candidates))]

    def _candidate_track_pairs(
        self,
        candidates: list[DetectionCandidate],
        tracks: list[DetectionTrack],
        scan_time: datetime,
    ) -> list[tuple[float, int, int]]:
        pairs: list[tuple[float, int, int]] = []
        for candidate_index, candidate in enumerate(candidates):
            candidate_xy = _cell_xy(candidate.cell)
            candidate_activation_xy = _activation_xy(candidate.detection_result)
            for track_index, track in enumerate(tracks):
                last_observation = track.last_observation
                if last_observation is None or last_observation.x_km is None or last_observation.y_km is None:
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
        return f"{site}-{scan_time.strftime('%Y%m%d%H%M%S')}-T{next(self._counter):04d}"

    def _confirm_track_if_ready(self, track: DetectionTrack, scan_time: datetime) -> bool:
        if track.confirmed_at is not None:
            return False
        hits, _window_size = self._window_counts(track)
        if hits >= self.config.confirm_scans:
            track.confirmed_at = scan_time
            logger.warning(
                "[%s] Confirmed detection track %s with %d/%d recent over-threshold scans",
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


def _observation_from_candidate(candidate: DetectionCandidate) -> TrackObservation:
    x_km, y_km = _cell_xy(candidate.cell)
    activation_xy = _activation_xy(candidate.detection_result)
    activation_x = None if activation_xy is None else activation_xy[0]
    activation_y = None if activation_xy is None else activation_xy[1]
    return TrackObservation(
        scan_time=candidate.scan_time,
        cell_id=candidate.cell.cell_id,
        probability=candidate.probability,
        above_threshold=candidate.above_threshold,
        x_km=x_km,
        y_km=y_km,
        activation_x_km=activation_x,
        activation_y_km=activation_y,
        quality_pass=candidate.quality_pass,
        quality_reasons=candidate.quality_reasons,
    )


def _miss_observation(scan_time: datetime) -> TrackObservation:
    return TrackObservation(
        scan_time=scan_time,
        cell_id=None,
        probability=0.0,
        above_threshold=False,
        x_km=None,
        y_km=None,
        activation_x_km=None,
        activation_y_km=None,
        quality_pass=False,
        missed=True,
    )


def _cell_xy(cell: StormCell) -> tuple[float, float]:
    az_rad = math.radians(cell.az_deg)
    return cell.range_km * math.sin(az_rad), cell.range_km * math.cos(az_rad)


def _activation_xy(detection_result: Any) -> tuple[float, float] | None:
    az_deg = getattr(detection_result, "activation_az_deg", None)
    range_km = getattr(detection_result, "activation_range_km", None)
    if az_deg is None or range_km is None:
        return None
    az_rad = math.radians(float(az_deg))
    range_km = float(range_km)
    return range_km * math.sin(az_rad), range_km * math.cos(az_rad)


def _last_activation_xy(track: DetectionTrack) -> tuple[float, float] | None:
    for observation in reversed(track.observations):
        if observation.missed:
            continue
        if observation.activation_x_km is not None and observation.activation_y_km is not None:
            return observation.activation_x_km, observation.activation_y_km
    return None


def _distance_km(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _hours_between(start: datetime, end: datetime) -> float:
    return (end - start).total_seconds() / 3600.0
