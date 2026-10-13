"""Object tracking module for Phase 8 using Ultralytics ByteTrack."""

from __future__ import annotations

from typing import Any, Dict, List

from ai_assistive_vision.perception.object_detection import ObjectDetector


class ObjectTracker:
    """Thin wrapper around Ultralytics ByteTrack tracking interface.

    Delegates frame tracking directly to Ultralytics YOLO model.track(...)
    with persist=True to maintain tracking state across consecutive frames.
    """

    def __init__(
        self,
        detector: ObjectDetector,
        tracker_type: str = "bytetrack.yaml",
        persist: bool = True,
    ) -> None:
        self.detector = detector
        self.tracker_type = tracker_type
        self.persist = persist

    def track(self, frame) -> List[Dict[str, Any]]:
        """Run ByteTrack on a single frame and return normalized detections with track_ids."""
        return self.detector.track(
            frame,
            tracker=self.tracker_type,
            persist=self.persist,
        )

    def reset(self) -> None:
        """Reset internal tracker state."""
        self.detector.reset_tracker()
