"""Environmental Text Recognition (OCR) module for Phase 10 using EasyOCR.

Provides both synchronous TextRecognizer and asynchronous OcrWorker thread interface.
"""

from __future__ import annotations

import logging
import os
import queue
import re
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from ai_assistive_vision.reasoning.spatial import classify_horizontal_direction

LOGGER = logging.getLogger(__name__)

# High-priority environmental keywords
ENVIRONMENTAL_KEYWORDS = {
    "EXIT",
    "EMERGENCY",
    "EMERGENCY EXIT",
    "WAY OUT",
    "ENTRANCE",
    "RESTROOM",
    "BATHROOM",
    "TOILET",
    "STAIRCASE",
    "STAIRS",
    "ELEVATOR",
    "LIFT",
    "PULL",
    "PUSH",
    "DANGER",
    "CAUTION",
    "WARNING",
    "NO ENTRY",
    "STOP",
}

# Room label regex patterns (e.g., "ROOM 204", "LAB 302", "BLOCK C", "RM 101", "204")
ROOM_LABEL_PATTERN = re.compile(
    r"^(ROOM|RM|LAB|SUITE|BLOCK|GATE|OFFICE|DEPT|NO\.?|#)?\s?\d{1,4}[A-Z]?$",
    re.IGNORECASE,
)


class TextRecognizer:
    """Wrapper around EasyOCR for environmental text recognition."""

    def __init__(
        self,
        languages: Optional[List[str]] = None,
        confidence: float = 0.50,
        device: str = "auto",
    ) -> None:
        self.languages = languages or ["en"]
        self.confidence = confidence
        self.use_gpu, self.device_name = self._resolve_device(device)
        self._reader = self._init_reader()

    def recognize(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Run OCR on frame and return structured, filtered text detections."""
        if frame is None or frame.size == 0:
            return []

        # EasyOCR accepts RGB image array
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        raw_results = self._reader.readtext(rgb_frame)

        frame_height, frame_width = frame.shape[:2]
        detections: List[Dict[str, Any]] = []

        for bbox_pts, text, conf in raw_results:
            conf_val = float(conf)
            cleaned_text = text.strip().upper()

            # Multi-tier filtering
            accepted, priority_level = classify_text_candidate(
                cleaned_text, conf_val, min_confidence=self.confidence
            )
            if not accepted:
                continue

            # Compute bounding box [x1, y1, x2, y2]
            pts = np.array(bbox_pts, dtype=np.int32)
            x1, y1 = int(pts[:, 0].min()), int(pts[:, 1].min())
            x2, y2 = int(pts[:, 0].max()), int(pts[:, 1].max())
            center_x = int(round((x1 + x2) / 2))
            center_y = int(round((y1 + y2) / 2))

            direction = classify_horizontal_direction(center_x, frame_width)

            detections.append(
                {
                    "event": "text",
                    "text": cleaned_text,
                    "confidence": conf_val,
                    "bbox": [x1, y1, x2, y2],
                    "center": [center_x, center_y],
                    "direction": direction,
                    "priority": priority_level,
                }
            )

        return detections

    def _init_reader(self) -> Any:
        try:
            os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
            import easyocr

            LOGGER.info(
                "Initializing EasyOCR reader (languages=%s, GPU=%s, device=%s)",
                self.languages,
                self.use_gpu,
                self.device_name,
            )
            return easyocr.Reader(self.languages, gpu=self.use_gpu, verbose=False)
        except ImportError as exc:
            raise RuntimeError(
                "easyocr is required for Phase 10 OCR. Install requirements.txt."
            ) from exc

    @staticmethod
    def _resolve_device(device: str) -> Tuple[bool, str]:
        try:
            os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
            import torch

            if device != "auto":
                is_cuda = "cuda" in device and torch.cuda.is_available()
                dev_name = torch.cuda.get_device_name(0) if is_cuda else "cpu"
                return is_cuda, dev_name

            if torch.cuda.is_available():
                return True, torch.cuda.get_device_name(0)
            return False, "cpu"
        except ImportError:
            return False, "cpu"


class OcrWorker:
    """Asynchronous background worker for EasyOCR text recognition.

    Runs EasyOCR in a dedicated thread with a bounded queue (maxsize=1)
    to prevent frame backlog accumulation and eliminate main thread blocking.
    """

    def __init__(
        self,
        languages: Optional[List[str]] = None,
        confidence: float = 0.50,
        device: str = "auto",
        queue_size: int = 1,
    ) -> None:
        self.languages = languages or ["en"]
        self.confidence = confidence
        self.device = device
        self.queue_size = max(1, queue_size)

        self._frame_queue: queue.Queue[Optional[np.ndarray]] = queue.Queue(maxsize=self.queue_size)
        self._result_lock = threading.Lock()
        self._latest_results: List[Dict[str, Any]] = []
        self._ocr_latency_ms: float = 0.0
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._dropped_frames_count = 0
        self._processed_frames_count = 0

    def start(self) -> None:
        """Start the background OCR worker thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the background OCR worker cleanly."""
        if not self._running:
            return
        self._running = False
        try:
            self._frame_queue.put_nowait(None)
        except queue.Full:
            pass
        if self._thread:
            self._thread.join(timeout=1.0)

    def submit_frame(self, frame: np.ndarray) -> bool:
        """Submit a frame for OCR processing. Drops frame if worker queue is full."""
        if not self._running or frame is None or frame.size == 0:
            return False

        frame_copy = frame.copy()
        try:
            self._frame_queue.put_nowait(frame_copy)
            return True
        except queue.Full:
            # Drop stale frame and replace with latest
            self._dropped_frames_count += 1
            try:
                self._frame_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._frame_queue.put_nowait(frame_copy)
                return True
            except queue.Full:
                return False

    def get_latest_results(self) -> List[Dict[str, Any]]:
        """Retrieve the latest structured OCR results thread-safely."""
        with self._result_lock:
            return list(self._latest_results)

    def get_latency_ms(self) -> float:
        """Retrieve the last measured OCR inference latency in milliseconds."""
        with self._result_lock:
            return self._ocr_latency_ms

    def reset(self) -> None:
        """Reset cached OCR results."""
        with self._result_lock:
            self._latest_results.clear()
            self._ocr_latency_ms = 0.0

    def _worker_loop(self) -> None:
        """Background worker thread loop."""
        try:
            recognizer = TextRecognizer(
                languages=self.languages,
                confidence=self.confidence,
                device=self.device,
            )
        except Exception as exc:
            LOGGER.error("Failed to initialize TextRecognizer in OcrWorker thread: %s", exc)
            self._running = False
            return

        while self._running:
            try:
                frame = self._frame_queue.get(timeout=0.1)
                if frame is None or not self._running:
                    break

                t0 = time.perf_counter()
                results = recognizer.recognize(frame)
                t1 = time.perf_counter()
                latency = (t1 - t0) * 1000.0

                with self._result_lock:
                    self._latest_results = results
                    self._ocr_latency_ms = latency
                    self._processed_frames_count += 1

            except queue.Empty:
                continue
            except Exception as exc:
                LOGGER.error("Error in OcrWorker processing loop: %s", exc)


def classify_text_candidate(
    text: str, confidence: float, min_confidence: float = 0.50
) -> Tuple[bool, str]:
    """Filter and classify OCR candidate string into (accepted, priority_level)."""
    if confidence < min_confidence:
        return False, "low"

    cleaned = text.strip().upper()
    if not cleaned:
        return False, "low"

    # Reject noise: single non-digit char or non-alphanumeric gibberish
    if len(cleaned) < 2 and not cleaned.isdigit():
        return False, "low"

    alphanumeric_count = sum(1 for c in cleaned if c.isalnum())
    if alphanumeric_count < len(cleaned) * 0.5:
        return False, "low"

    # 1. High-priority environmental keywords
    if cleaned in ENVIRONMENTAL_KEYWORDS or any(kw in cleaned for kw in ("EXIT", "EMERGENCY", "DANGER", "RESTROOM")):
        return True, "high"

    # 2. Room numbers / room labels
    if ROOM_LABEL_PATTERN.match(cleaned):
        return True, "high"

    # 3. Meaningful general text (valid readable text)
    return True, "medium"
