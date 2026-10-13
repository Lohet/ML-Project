"""Tests for Phase 8 Object Tracking using ByteTrack."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import numpy as np

from ai_assistive_vision.perception.object_detection import Detection, parse_yolo_result
from ai_assistive_vision.perception.tracking import ObjectTracker


def create_mock_box(xyxy: list[float], cls_id: int, conf: float, track_id: int | None = None):
    """Helper to construct a mock Ultralytics Box tensor object."""
    mock_box = MagicMock()
    mock_box.xyxy = [MagicMock(tolist=lambda: xyxy, detach=lambda: mock_box.xyxy[0], cpu=lambda: mock_box.xyxy[0])]
    mock_box.cls = MagicMock(item=lambda: cls_id, detach=lambda: mock_box.cls, cpu=lambda: mock_box.cls)
    mock_box.conf = MagicMock(item=lambda: conf, detach=lambda: mock_box.conf, cpu=lambda: mock_box.conf)
    if track_id is not None:
        mock_box.id = MagicMock(item=lambda: track_id, detach=lambda: mock_box.id, cpu=lambda: mock_box.id)
    else:
        mock_box.id = None
    return mock_box


def test_parse_yolo_result_with_track_id():
    """Verify parse_yolo_result properly extracts track_id when present."""
    mock_result = MagicMock()
    mock_result.names = {0: "person", 1: "car"}
    mock_result.boxes = [
        create_mock_box([10, 20, 100, 200], cls_id=0, conf=0.9, track_id=3),
        create_mock_box([150, 50, 300, 400], cls_id=1, conf=0.85, track_id=None),
    ]

    detections = parse_yolo_result(mock_result)
    assert len(detections) == 2

    assert detections[0].label == "person"
    assert detections[0].track_id == 3
    assert detections[0].bbox == [10, 20, 100, 200]

    assert detections[1].label == "car"
    assert detections[1].track_id is None
    assert detections[1].bbox == [150, 50, 300, 400]

    # Verify dict conversion preserves backward compatibility
    d0_dict = detections[0].as_dict()
    assert d0_dict["track_id"] == 3
    assert d0_dict["label"] == "person"
    assert d0_dict["confidence"] == 0.9
    assert d0_dict["bbox"] == [10, 20, 100, 200]
    assert d0_dict["center"] == [55, 110]


def test_object_tracker_delegates_to_detector():
    """Verify ObjectTracker correctly delegates tracking call to ObjectDetector.track()."""
    mock_detector = MagicMock()
    mock_detector.track.return_value = [
        {
            "track_id": 5,
            "label": "person",
            "confidence": 0.88,
            "bbox": [10, 10, 50, 50],
            "center": [30, 30],
        }
    ]

    tracker = ObjectTracker(detector=mock_detector, tracker_type="bytetrack.yaml", persist=True)
    dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)

    results = tracker.track(dummy_frame)
    assert len(results) == 1
    assert results[0]["track_id"] == 5

    mock_detector.track.assert_called_once_with(
        dummy_frame,
        tracker="bytetrack.yaml",
        persist=True,
    )

    tracker.reset()
    mock_detector.reset_tracker.assert_called_once()


def test_tracker_missed_detections_and_new_id_recovery():
    """Simulate sequence of frames with persistent ID, temporary frame drop, and recovery, plus a new object."""
    mock_detector = MagicMock()

    # Frame 1: Object A (car) -> track_id 3
    frame1_res = [
        {"track_id": 3, "label": "car", "confidence": 0.92, "bbox": [10, 10, 100, 100], "center": [55, 55]}
    ]
    # Frame 2: Object A (car) -> track_id 3
    frame2_res = [
        {"track_id": 3, "label": "car", "confidence": 0.91, "bbox": [15, 12, 105, 102], "center": [60, 57]}
    ]
    # Frame 3: Object A temporarily missing (detection empty)
    frame3_res = []
    # Frame 4: Object A recovers -> track_id 3, and a new Object B appears -> track_id 7
    frame4_res = [
        {"track_id": 3, "label": "car", "confidence": 0.89, "bbox": [20, 15, 110, 105], "center": [65, 60]},
        {"track_id": 7, "label": "person", "confidence": 0.82, "bbox": [300, 300, 400, 400], "center": [350, 350]},
    ]

    mock_detector.track.side_effect = [frame1_res, frame2_res, frame3_res, frame4_res]
    tracker = ObjectTracker(detector=mock_detector, tracker_type="bytetrack.yaml", persist=True)

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Frame 1
    out1 = tracker.track(dummy_frame)
    assert len(out1) == 1
    assert out1[0]["track_id"] == 3

    # Frame 2
    out2 = tracker.track(dummy_frame)
    assert len(out2) == 1
    assert out2[0]["track_id"] == 3

    # Frame 3 (missed detection frame)
    out3 = tracker.track(dummy_frame)
    assert len(out3) == 0

    # Frame 4 (recovered object retains ID 3, new object gets ID 7)
    out4 = tracker.track(dummy_frame)
    assert len(out4) == 2
    recovered_car = next(d for d in out4 if d["label"] == "car")
    new_person = next(d for d in out4 if d["label"] == "person")

    assert recovered_car["track_id"] == 3
    assert new_person["track_id"] == 7
    assert new_person["track_id"] != recovered_car["track_id"]
