"""Tests for Phase 5 decision making."""

from ai_assistive_vision.reasoning.decision import extract_events


def test_extract_events_filters_obstacles():
    detections = [
        {"label": "person", "distance": "far", "direction": "left"},
        {"label": "car", "distance": "medium", "direction": "right"},
        {"label": "chair", "distance": "close", "direction": "center"},
        {"label": "door", "distance": "very_close", "direction": "left"},
    ]
    
    events = extract_events(detections)
    
    assert len(events) == 2
    
    assert events[0]["event"] == "obstacle"
    assert events[0]["object"] == "chair"
    assert events[0]["distance"] == "close"
    assert events[0]["priority"] == "medium"
    
    assert events[1]["event"] == "obstacle"
    assert events[1]["object"] == "door"
    assert events[1]["distance"] == "very_close"
    assert events[1]["priority"] == "high"
