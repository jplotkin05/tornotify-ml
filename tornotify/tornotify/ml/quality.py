"""
Quality flags for model-scored detection candidates.

The TorNet model can assign high probabilities to small or near-radar artifacts.
These flags keep raw probabilities visible for analysis while marking weak cell
candidates for temporal confirmation and review.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from tornotify.config import (
    CONFIDENCE_THRESHOLD,
    DETECTION_MIN_GATES,
    DETECTION_MIN_HEATMAP_EDGE_MARGIN,
    DETECTION_MIN_RANGE_KM,
)
if TYPE_CHECKING:
    from tornotify.preprocess.cells import StormCell


@dataclass(frozen=True)
class DetectionDecision:
    """Single-frame threshold and quality status."""

    probability: float
    above_threshold: bool
    actionable: bool  # passes quality gates; temporal tracking still controls confirmation
    reasons: tuple[str, ...] = ()


def evaluate_detection(
    cell: StormCell,
    detection_result: Any,
    threshold: float = CONFIDENCE_THRESHOLD,
) -> DetectionDecision:
    """Return single-frame threshold and quality status for a scored cell."""
    probability = float(getattr(detection_result, "probability", detection_result))
    above_threshold = probability >= threshold
    if not above_threshold:
        return DetectionDecision(
            probability=probability,
            above_threshold=False,
            actionable=False,
        )

    reasons = []
    if cell.range_km < DETECTION_MIN_RANGE_KM:
        reasons.append(f"range {cell.range_km:.1f}km < {DETECTION_MIN_RANGE_KM:.1f}km")
    if cell.n_gates < DETECTION_MIN_GATES:
        reasons.append(f"n_gates {cell.n_gates} < {DETECTION_MIN_GATES}")

    edge_margin = getattr(detection_result, "heatmap_edge_margin", None)
    if edge_margin is not None and edge_margin < DETECTION_MIN_HEATMAP_EDGE_MARGIN:
        reasons.append(
            f"heatmap edge margin {edge_margin:.3f} < {DETECTION_MIN_HEATMAP_EDGE_MARGIN:.3f}"
        )

    return DetectionDecision(
        probability=probability,
        above_threshold=True,
        actionable=not reasons,
        reasons=tuple(reasons),
    )
