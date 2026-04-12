"""
Tornado detection inference using the pretrained TorNet baseline CNN.

Model: tornet-ml/tornado_detector_baseline_v1 (HuggingFace)
Framework: Keras 3 with PyTorch backend

Input: dict with keys 'DBZ','VEL','KDP','RHOHV','ZDR','WIDTH',
       'range_folded_mask', 'coordinates'
       Each value shape: [batch, A=120, R=240, S=2] (or [batch, A, R, 2] for coords)
Output: logit (pre-sigmoid) — we apply sigmoid to get P(tornado) ∈ [0, 1]
"""
import logging
import os
import threading
from dataclasses import dataclass

import numpy as np

from tornotify.config import (
    TORNET_MODEL_REPO,
    TORNET_MODEL_FILE,
    CONFIDENCE_THRESHOLD,
    KERAS_BACKEND,
)
from tornotify.preprocess.chips import chip_to_batch, ALL_VARIABLES

logger = logging.getLogger(__name__)

os.environ.setdefault("KERAS_BACKEND", KERAS_BACKEND)

_model = None
_heatmap_model = None
_model_lock = threading.RLock()


@dataclass(frozen=True)
class DetectionResult:
    """Model score plus the heatmap location that produced the score."""

    probability: float
    logit: float
    heatmap_row: int | None = None
    heatmap_col: int | None = None
    heatmap_shape: tuple[int, int] | None = None
    heatmap_edge_margin: float | None = None
    activation_az_deg: float | None = None
    activation_range_km: float | None = None


def load_model():
    """Download and load the pretrained TorNet model. Cached after first call."""
    global _model
    if _model is not None:
        return _model

    with _model_lock:
        if _model is not None:
            return _model

        from huggingface_hub import hf_hub_download
        import keras

        # Import TorNet custom layers so Keras can deserialize them.
        # These are decorated with @keras.saving.register_keras_serializable()
        # and must be imported before load_model.
        import sys, os as _os
        _tornet_root = _os.path.join(_os.path.dirname(__file__), '..', '..', 'tornet')
        if _os.path.isdir(_tornet_root) and _tornet_root not in sys.path:
            sys.path.insert(0, _os.path.abspath(_tornet_root))
        from tornet.models.keras.layers import CoordConv2D, FillNaNs  # noqa: F401

        logger.info("Downloading pretrained model: %s", TORNET_MODEL_REPO)
        model_path = hf_hub_download(
            repo_id=TORNET_MODEL_REPO,
            filename=TORNET_MODEL_FILE,
        )
        logger.info("Loading model from %s", model_path)
        _model = keras.models.load_model(model_path)

        logger.info("Model loaded successfully")
        return _model


def load_heatmap_model():
    """Return a model that exposes both final logit and pre-pool heatmap."""
    global _heatmap_model
    if _heatmap_model is not None:
        return _heatmap_model

    with _model_lock:
        if _heatmap_model is not None:
            return _heatmap_model

        model = load_model()
        import keras

        heatmap = model.get_layer("heatmap").output
        _heatmap_model = keras.Model(
            inputs=model.inputs,
            outputs=[model.outputs[0], heatmap],
            name="tornet_detector_with_heatmap",
        )
        return _heatmap_model


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def _model_input_batch(chips: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    model_keys = ALL_VARIABLES + ['range_folded_mask', 'coordinates']
    return {
        k: np.stack([c[k] for c in chips], axis=0)
        for k in model_keys
    }


def predict(chip: dict[str, np.ndarray]) -> float:
    """
    Run inference on a single chip dict.

    Args:
        chip: output of extract_chip() — dict with variable arrays [A, R, S]

    Returns:
        P(tornado) as float in [0, 1]
    """
    model = load_model()
    batch = chip_to_batch(chip)
    logit = model.predict(batch, verbose=0)
    return float(_sigmoid(np.squeeze(logit)))


def predict_batch(chips: list[dict[str, np.ndarray]]) -> list[float]:
    """
    Run inference on a list of chip dicts.

    Args:
        chips: list of dicts from extract_chip()

    Returns:
        list of P(tornado) floats
    """
    if not chips:
        return []

    model = load_model()
    batch = _model_input_batch(chips)
    logits = model.predict(batch, verbose=0)
    probs = 1.0 / (1.0 + np.exp(-np.squeeze(logits).reshape(-1)))
    return [float(p) for p in probs]


def predict_batch_detailed(chips: list[dict[str, np.ndarray]]) -> list[DetectionResult]:
    """
    Run inference and expose the model heatmap max location for each chip.

    The model's final scalar is a global max over a low-resolution heatmap. This
    function maps that max back to the chip's azimuth/range metadata so callers
    can distinguish the storm-cell centroid from the model's strongest sub-cell
    response.
    """
    if not chips:
        return []

    model = load_heatmap_model()
    logits, heatmaps = model.predict(_model_input_batch(chips), verbose=0)
    logits = np.asarray(logits).reshape(-1)
    heatmaps = np.asarray(heatmaps)

    results = []
    for chip, logit, heatmap in zip(chips, logits, heatmaps):
        heatmap_2d = np.squeeze(heatmap)
        if heatmap_2d.ndim != 2 or not np.any(np.isfinite(heatmap_2d)):
            results.append(DetectionResult(probability=float(_sigmoid(logit)), logit=float(logit)))
            continue

        row, col = np.unravel_index(int(np.nanargmax(heatmap_2d)), heatmap_2d.shape)
        activation_az_deg, activation_range_km = _chip_activation_location(
            chip=chip,
            row=row,
            col=col,
            heatmap_shape=heatmap_2d.shape,
        )
        results.append(
            DetectionResult(
                probability=float(_sigmoid(logit)),
                logit=float(logit),
                heatmap_row=int(row),
                heatmap_col=int(col),
                heatmap_shape=(int(heatmap_2d.shape[0]), int(heatmap_2d.shape[1])),
                heatmap_edge_margin=_heatmap_edge_margin(row, col, heatmap_2d.shape),
                activation_az_deg=activation_az_deg,
                activation_range_km=activation_range_km,
            )
        )
    return results


def _chip_activation_location(
    chip: dict[str, np.ndarray],
    row: int,
    col: int,
    heatmap_shape: tuple[int, int],
) -> tuple[float | None, float | None]:
    if heatmap_shape[0] <= 0 or heatmap_shape[1] <= 0:
        return None, None
    required_keys = ("_az_lower", "_az_upper", "_rng_lower_m", "_rng_upper_m")
    if any(key not in chip for key in required_keys):
        return None, None

    az_frac = (row + 0.5) / heatmap_shape[0]
    range_frac = (col + 0.5) / heatmap_shape[1]
    az_deg = chip["_az_lower"] + az_frac * (chip["_az_upper"] - chip["_az_lower"])
    range_m = chip["_rng_lower_m"] + range_frac * (chip["_rng_upper_m"] - chip["_rng_lower_m"])
    return float(az_deg % 360.0), float(range_m / 1000.0)


def _heatmap_edge_margin(row: int, col: int, shape: tuple[int, int]) -> float:
    rows, cols = shape
    if rows <= 0 or cols <= 0:
        return 0.0
    row_frac = (row + 0.5) / rows
    col_frac = (col + 0.5) / cols
    return float(min(row_frac, 1.0 - row_frac, col_frac, 1.0 - col_frac))


def is_above_threshold(prob: float, threshold: float = CONFIDENCE_THRESHOLD) -> bool:
    return prob >= threshold
