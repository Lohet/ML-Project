"""Unit tests for Phase 10 Environmental Text Recognition (OCR) and OcrWorker."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest
import numpy as np

from ai_assistive_vision.perception.ocr import OcrWorker, TextRecognizer, classify_text_candidate
from ai_assistive_vision.reasoning.decision import extract_events
from ai_assistive_vision.reasoning.priority import EventManager


def test_classify_text_candidate_filtering():
    """Verify multi-tier text filtering (confidence, keywords, room numbers, noise)."""
    # 1. Low confidence -> Rejected
    accepted, _ = classify_text_candidate("EXIT", confidence=0.30, min_confidence=0.50)
    assert not accepted

    # 2. High-priority keyword -> Accepted, high priority
    accepted, priority = classify_text_candidate("EXIT", confidence=0.90, min_confidence=0.50)
    assert accepted
    assert priority == "high"

    # 3. Room number pattern -> Accepted, high priority
    accepted, priority = classify_text_candidate("ROOM 204", confidence=0.85, min_confidence=0.50)
    assert accepted
    assert priority == "high"

    accepted_num, priority_num = classify_text_candidate("204", confidence=0.80, min_confidence=0.50)
    assert accepted_num
    assert priority_num == "high"

    # 4. Meaningful non-keyword text -> Accepted, medium priority
    accepted, priority = classify_text_candidate("INFORMATION DESK", confidence=0.75, min_confidence=0.50)
    assert accepted
    assert priority == "medium"

    # 5. Single character noise / gibberish -> Rejected
    accepted_single, _ = classify_text_candidate(".", confidence=0.99, min_confidence=0.50)
    assert not accepted_single

    accepted_noise, _ = classify_text_candidate("#$@!", confidence=0.99, min_confidence=0.50)
    assert not accepted_noise


def test_device_resolution():
    """Verify CUDA availability probe and CPU fallback logic."""
    with patch("torch.cuda.is_available", return_value=True), patch("torch.cuda.get_device_name", return_value="NVIDIA RTX 4060"):
        is_gpu, dev_name = TextRecognizer._resolve_device("auto")
        assert is_gpu is True
        assert "RTX 4060" in dev_name

    with patch("torch.cuda.is_available", return_value=False):
        is_gpu, dev_name = TextRecognizer._resolve_device("auto")
        assert is_gpu is False
        assert dev_name == "cpu"


def test_text_recognizer_mock_frame():
    """Verify TextRecognizer.recognize extracts bbox, center, direction, and text fields."""
    mock_reader = MagicMock()
    # Mock EasyOCR output: [ (bbox_corners, text, confidence) ]
    mock_reader.readtext.return_value = [
        ([[10, 10], [100, 10], [100, 50], [10, 50]], "EXIT", 0.95),
        ([[500, 10], [600, 10], [600, 50], [500, 50]], "ROOM 204", 0.88),
        ([[10, 100], [50, 100], [50, 120], [10, 120]], "#$@", 0.99),  # noise, should be filtered
    ]

    mock_easyocr = MagicMock()
    mock_easyocr.Reader.return_value = mock_reader

    with patch.dict("sys.modules", {"easyocr": mock_easyocr}):
        # Re-import to pick up the mocked module
        import importlib
        import ai_assistive_vision.perception.ocr as ocr_module
        importlib.reload(ocr_module)
        recognizer = ocr_module.TextRecognizer(confidence=0.50, device="cpu")
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        detections = recognizer.recognize(dummy_frame)

        assert len(detections) == 2  # Noise filtered out

        det1 = next(d for d in detections if d["text"] == "EXIT")
        assert det1["event"] == "text"
        assert det1["direction"] == "left"
        assert det1["priority"] == "high"
        assert det1["bbox"] == [10, 10, 100, 50]
        assert det1["center"] == [55, 30]

        det2 = next(d for d in detections if d["text"] == "ROOM 204")
        assert det2["direction"] == "right"
        assert det2["priority"] == "high"


def test_no_text_empty_frame():
    """Verify empty or zero-text frame returns an empty list."""
    mock_reader = MagicMock()
    mock_reader.readtext.return_value = []

    mock_easyocr = MagicMock()
    mock_easyocr.Reader.return_value = mock_reader

    with patch.dict("sys.modules", {"easyocr": mock_easyocr}):
        import importlib
        import ai_assistive_vision.perception.ocr as ocr_module
        importlib.reload(ocr_module)
        recognizer = ocr_module.TextRecognizer(confidence=0.50, device="cpu")
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        results = recognizer.recognize(dummy_frame)
        assert results == []


def test_ocr_worker_lifecycle_submission_retrieval():
    """Verify OcrWorker initialization, submission, result retrieval, and shutdown."""
    mock_recognizer = MagicMock()
    mock_recognizer.recognize.return_value = [
        {"event": "text", "text": "EXIT", "confidence": 0.95, "direction": "center"}
    ]

    with patch("ai_assistive_vision.perception.ocr.TextRecognizer", return_value=mock_recognizer):
        worker = OcrWorker(queue_size=1, device="cpu")
        worker.start()

        dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)

        # Before submission -> empty results
        assert worker.get_latest_results() == []

        # Submit frame
        submitted = worker.submit_frame(dummy_frame)
        assert submitted is True

        # Wait briefly for background thread to process
        time.sleep(0.2)

        results = worker.get_latest_results()
        assert len(results) == 1
        assert results[0]["text"] == "EXIT"
        assert worker.get_latency_ms() >= 0.0

        # Reset & shutdown
        worker.reset()
        assert worker.get_latest_results() == []

        worker.stop()
        assert worker._running is False


def test_ocr_worker_stale_frame_dropping():
    """Verify OcrWorker bounded queue drops stale frames when busy."""
    mock_recognizer = MagicMock()

    # Simulate slow recognizer
    def slow_recognize(frame):
        time.sleep(0.3)
        return [{"event": "text", "text": "SLOW", "confidence": 0.90}]

    mock_recognizer.recognize.side_effect = slow_recognize

    with patch("ai_assistive_vision.perception.ocr.TextRecognizer", return_value=mock_recognizer):
        worker = OcrWorker(queue_size=1, device="cpu")
        worker.start()

        f1 = np.zeros((10, 10, 3), dtype=np.uint8)
        f2 = np.ones((10, 10, 3), dtype=np.uint8)
        f3 = np.full((10, 10, 3), 2, dtype=np.uint8)

        worker.submit_frame(f1)
        # Immediately submit f2 and f3 while worker is busy processing f1
        worker.submit_frame(f2)
        worker.submit_frame(f3)

        assert worker._dropped_frames_count >= 1

        worker.stop()


def test_ocr_worker_exception_handling():
    """Verify OcrWorker handles exceptions in worker loop gracefully."""
    mock_recognizer = MagicMock()
    mock_recognizer.recognize.side_effect = RuntimeError("OCR Internal Error")

    with patch("ai_assistive_vision.perception.ocr.TextRecognizer", return_value=mock_recognizer):
        worker = OcrWorker(queue_size=1, device="cpu")
        worker.start()

        dummy_frame = np.zeros((10, 10, 3), dtype=np.uint8)
        worker.submit_frame(dummy_frame)
        time.sleep(0.15)

        # Worker handled exception without crashing application
        assert worker.get_latest_results() == []
        worker.stop()


def test_text_event_integration_and_deduplication():
    """Verify decision engine integration and EventManager deduplication for text events."""
    event_mgr = EventManager(cooldown_seconds=3.0)

    text_detections = [
        {
            "event": "text",
            "text": "EXIT",
            "confidence": 0.95,
            "bbox": [10, 10, 100, 50],
            "center": [55, 30],
            "direction": "left",
            "priority": "high",
        }
    ]

    extracted = extract_events(detections=[], text_events=text_detections)
    assert len(extracted) == 1
    assert extracted[0]["text"] == "EXIT"

    # First announcement -> allowed
    p1 = event_mgr.filter_and_prioritize(extracted)
    assert len(p1) == 1

    # Immediate duplicate -> suppressed during cooldown
    p2 = event_mgr.filter_and_prioritize(extracted)
    assert len(p2) == 0

    # Same text, but direction changes to center -> allowed!
    moved_text = [dict(text_detections[0], direction="center")]
    extracted_moved = extract_events(detections=[], text_events=moved_text)
    p3 = event_mgr.filter_and_prioritize(extracted_moved)
    assert len(p3) == 1
    assert p3[0]["direction"] == "center"
