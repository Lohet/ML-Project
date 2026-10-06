"""YOLO object detection for Phase 2."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class Detection:
    """Normalized object detection output used by later reasoning phases."""

    label: str
    confidence: float
    bbox: list[int]
    center: list[int]

    def as_dict(self) -> dict[str, object]:
        return {
            "label": self.label,
            "confidence": self.confidence,
            "bbox": self.bbox,
            "center": self.center,
        }


class ObjectDetector:
    """Lightweight wrapper around an Ultralytics YOLO model."""

    def __init__(
        self,
        model_name: str,
        confidence: float = 0.35,
        device: str = "auto",
    ) -> None:
        if not 0 <= confidence <= 1:
            raise ValueError("confidence must be in the range [0, 1]")

        self.model_name = model_name
        self.confidence = confidence
        self.device = self._resolve_device(device)
        self._model = self._load_model()

    def detect(self, frame) -> list[dict[str, object]]:
        results = self._model.predict(
            frame,
            conf=self.confidence,
            device=self.device,
            verbose=False,
        )
        if not results:
            return []
        return [detection.as_dict() for detection in parse_yolo_result(results[0])]

    def _load_model(self) -> Any:
        try:
            # Set before importing torch through ultralytics. The local Python
            # environment currently has duplicate OpenMP runtimes.
            os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "Ultralytics is required for Phase 2. Install requirements.txt."
            ) from exc

        LOGGER.info("Loading YOLO model %s on %s", self.model_name, self.device)
        return YOLO(self.model_name)

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


def parse_yolo_result(result) -> list[Detection]:
    """Convert one Ultralytics result into the project detection format."""

    detections: list[Detection] = []
    names = getattr(result, "names", {})
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return detections

    for box in boxes:
        xyxy = _tensor_to_list(box.xyxy[0])
        x1, y1, x2, y2 = [int(round(value)) for value in xyxy]
        class_id = int(_tensor_to_scalar(box.cls))
        confidence = float(_tensor_to_scalar(box.conf))
        label = str(names.get(class_id, class_id))
        center_x = int(round((x1 + x2) / 2))
        center_y = int(round((y1 + y2) / 2))

        detections.append(
            Detection(
                label=label,
                confidence=confidence,
                bbox=[x1, y1, x2, y2],
                center=[center_x, center_y],
            )
        )

    return detections


def _tensor_to_list(value) -> list[float]:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)


def _tensor_to_scalar(value) -> float:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "item"):
        return value.item()
    return value[0] if isinstance(value, (list, tuple)) else value
