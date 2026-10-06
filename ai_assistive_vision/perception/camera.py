"""Camera capture utilities for Phase 1."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import cv2


LOGGER = logging.getLogger(__name__)


@dataclass
class CameraConfig:
    """Runtime settings for an OpenCV camera stream."""

    index: int
    width: int
    height: int


class CameraError(RuntimeError):
    """Raised when the camera cannot be opened or read."""


class FpsCounter:
    """Smoothed FPS counter for display overlays."""

    def __init__(self, smoothing: float = 0.9) -> None:
        if not 0 <= smoothing < 1:
            raise ValueError("smoothing must be in the range [0, 1)")
        self.smoothing = smoothing
        self._last_time: float | None = None
        self._fps = 0.0

    @property
    def fps(self) -> float:
        return self._fps

    def update(self, now: float | None = None) -> float:
        current_time = time.perf_counter() if now is None else now
        if self._last_time is None:
            self._last_time = current_time
            return self._fps

        elapsed = current_time - self._last_time
        self._last_time = current_time
        if elapsed <= 0:
            return self._fps

        instant_fps = 1.0 / elapsed
        if self._fps == 0:
            self._fps = instant_fps
        else:
            self._fps = (self.smoothing * self._fps) + (
                (1 - self.smoothing) * instant_fps
            )
        return self._fps


class Camera:
    """Small wrapper around OpenCV VideoCapture."""

    def __init__(self, config: CameraConfig) -> None:
        self.config = config
        self._capture: cv2.VideoCapture | None = None

    def open(self) -> None:
        LOGGER.info("Opening camera index %s", self.config.index)
        capture = cv2.VideoCapture(self.config.index)
        if not capture.isOpened():
            capture.release()
            raise CameraError(
                f"Could not open camera index {self.config.index}. "
                "Check that a webcam is connected and not used by another app."
            )

        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
        self._capture = capture

    def read(self):
        if self._capture is None:
            raise CameraError("Camera has not been opened.")

        ok, frame = self._capture.read()
        if not ok or frame is None:
            raise CameraError("Could not read a frame from the camera.")
        return frame

    def release(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None
            LOGGER.info("Camera released")

    def __enter__(self) -> "Camera":
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.release()
