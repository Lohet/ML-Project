"""Relative monocular depth estimation for Phase 4."""

from __future__ import annotations

import logging
import os
from typing import Literal

import cv2
import numpy as np


LOGGER = logging.getLogger(__name__)

RelativeDistance = Literal["far", "medium", "close", "very_close"]


class DepthEstimator:
    """Lightweight MiDaS depth estimator.

    MiDaS returns relative inverse depth, not calibrated physical distance.
    Larger normalized values are treated as closer within the current frame.
    """

    def __init__(self, model_type: str = "MiDaS_small", device: str = "auto") -> None:
        self.model_type = model_type
        self.device = self._resolve_device(device)
        self._model, self._transform = self._load_model()

    def estimate(self, frame) -> np.ndarray:
        import torch

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_batch = self._transform(rgb_frame).to(self.device)

        with torch.no_grad():
            prediction = self._model(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=rgb_frame.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth = prediction.detach().cpu().numpy()
        return normalize_depth_map(depth)

    def _load_model(self):
        try:
            os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
            import torch
        except ImportError as exc:
            raise RuntimeError("PyTorch is required for Phase 4 depth estimation.") from exc

        LOGGER.info("Loading depth model %s on %s", self.model_type, self.device)
        model = torch.hub.load("intel-isl/MiDaS", self.model_type)
        model.to(self.device)
        model.eval()

        transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        transform = transforms.small_transform
        return model, transform

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device != "auto":
            return device

        try:
            os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
            import torch
        except ImportError:
            return "cpu"

        return "cuda:0" if torch.cuda.is_available() else "cpu"


def normalize_depth_map(depth_map: np.ndarray) -> np.ndarray:
    """Normalize any depth-like map into the range 0..1."""

    depth = depth_map.astype(np.float32)
    finite_mask = np.isfinite(depth)
    if not finite_mask.any():
        return np.zeros_like(depth, dtype=np.float32)

    finite_values = depth[finite_mask]
    min_value = float(finite_values.min())
    max_value = float(finite_values.max())
    span = max_value - min_value
    if span <= 1e-6:
        return np.zeros_like(depth, dtype=np.float32)

    normalized = (depth - min_value) / span
    normalized[~finite_mask] = 0
    return np.clip(normalized, 0, 1).astype(np.float32)


def classify_relative_distance(depth_score: float) -> RelativeDistance:
    """Classify normalized inverse depth into a relative distance bucket."""

    if depth_score < 0.25:
        return "far"
    if depth_score < 0.50:
        return "medium"
    if depth_score < 0.75:
        return "close"
    return "very_close"


def add_relative_depth(
    detections: list[dict[str, object]],
    depth_map: np.ndarray,
) -> list[dict[str, object]]:
    """Add relative distance labels to detections from a normalized depth map."""

    enriched = []
    height, width = depth_map.shape[:2]

    for detection in detections:
        x1, y1, x2, y2 = detection["bbox"]
        x1 = max(0, min(int(x1), width - 1))
        x2 = max(0, min(int(x2), width))
        y1 = max(0, min(int(y1), height - 1))
        y2 = max(0, min(int(y2), height))

        enriched_detection = dict(detection)
        if x2 <= x1 or y2 <= y1:
            depth_score = 0.0
        else:
            object_depth = depth_map[y1:y2, x1:x2]
            depth_score = float(np.median(object_depth))

        enriched_detection["depth_score"] = depth_score
        enriched_detection["distance"] = classify_relative_distance(depth_score)
        enriched.append(enriched_detection)

    return enriched
