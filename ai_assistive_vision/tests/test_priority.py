"""Tests for Phase 6 priority and deduplication."""

import time
from ai_assistive_vision.reasoning.priority import EventManager

def test_event_manager_prioritizes_correctly():
    manager = EventManager()
    events = [
        {"object": "bottle", "direction": "left", "distance": "medium", "priority": "medium"},
        {"object": "car", "direction": "right", "distance": "far", "priority": "medium"},
        {"object": "person", "direction": "center", "distance": "very_close", "priority": "high"},
    ]
    
    filtered = manager.filter_and_prioritize(events)
    
    assert len(filtered) == 3
    # High priority person should be first
    assert filtered[0]["object"] == "person"
    assert filtered[0]["category"] == "immediate_obstacle"
    
    # Car should be second
    assert filtered[1]["object"] == "car"
    assert filtered[1]["category"] == "vehicle"
    
    # Bottle should be last
    assert filtered[2]["object"] == "bottle"
    assert filtered[2]["category"] == "general"

def test_event_manager_deduplicates():
    manager = EventManager(cooldown_seconds=1.0)
    
    event = {"object": "person", "direction": "center", "distance": "medium"}
    
    # First time should pass
    filtered = manager.filter_and_prioritize([dict(event)])
    assert len(filtered) == 1
    
    # Second time immediately should be filtered out
    filtered2 = manager.filter_and_prioritize([dict(event)])
    assert len(filtered2) == 0
    
    # If distance changes, it should pass
    event_changed = {"object": "person", "direction": "center", "distance": "close"}
    filtered3 = manager.filter_and_prioritize([dict(event_changed)])
    assert len(filtered3) == 1
