"""Deterministic unit tests for Phase 9 Temporal Reasoning."""

from __future__ import annotations

import pytest

from ai_assistive_vision.reasoning.temporal import TemporalReasoner


def test_approaching_object_via_increasing_depth_score():
    """Verify increasing depth_score (higher = closer) triggers an 'approaching' event."""
    reasoner = TemporalReasoner(depth_threshold=0.08, history_len=5)

    # Frame 1: depth_score = 0.20 (farther)
    det1 = [{"track_id": 1, "label": "car", "center": [100, 100], "direction": "center", "distance": "far", "depth_score": 0.20}]
    events1 = reasoner.process_frame(det1)
    assert events1 == []

    # Frame 2: depth_score = 0.35 (closer) -> delta 0.15 >= 0.08
    det2 = [{"track_id": 1, "label": "car", "center": [100, 100], "direction": "center", "distance": "medium", "depth_score": 0.35}]
    events2 = reasoner.process_frame(det2)
    assert len(events2) == 1
    assert events2[0]["event"] == "approaching"
    assert events2[0]["track_id"] == 1


def test_receding_object_via_decreasing_depth_score():
    """Verify decreasing depth_score (lower = farther) triggers a 'receding' event."""
    reasoner = TemporalReasoner(depth_threshold=0.08, history_len=5)

    # Frame 1: depth_score = 0.70 (close)
    det1 = [{"track_id": 2, "label": "person", "center": [200, 200], "direction": "center", "distance": "close", "depth_score": 0.70}]
    reasoner.process_frame(det1)

    # Frame 2: depth_score = 0.30 (medium/farther) -> delta -0.40 <= -0.08
    det2 = [{"track_id": 2, "label": "person", "center": [200, 200], "direction": "center", "distance": "medium", "depth_score": 0.30}]
    events2 = reasoner.process_frame(det2)
    assert len(events2) == 1
    assert events2[0]["event"] == "receding"
    assert events2[0]["track_id"] == 2


def test_distance_bucket_progression_far_medium_close():
    """Verify progression FAR -> MEDIUM -> CLOSE triggers approaching event."""
    reasoner = TemporalReasoner(depth_threshold=0.20, history_len=5)

    events1 = reasoner.process_frame([{"track_id": 3, "label": "bus", "center": [50, 50], "direction": "left", "distance": "far", "depth_score": 0.10}])
    events2 = reasoner.process_frame([{"track_id": 3, "label": "bus", "center": [50, 50], "direction": "left", "distance": "medium", "depth_score": 0.15}])
    events3 = reasoner.process_frame([{"track_id": 3, "label": "bus", "center": [50, 50], "direction": "left", "distance": "close", "depth_score": 0.18}])

    all_events = events1 + events2 + events3
    assert any(e["event"] == "approaching" and e["track_id"] == 3 for e in all_events)


def test_distance_bucket_regression_close_medium_far():
    """Verify regression CLOSE -> MEDIUM -> FAR triggers receding event."""
    reasoner = TemporalReasoner(depth_threshold=0.20, history_len=5)

    events1 = reasoner.process_frame([{"track_id": 4, "label": "person", "center": [50, 50], "direction": "right", "distance": "close", "depth_score": 0.70}])
    events2 = reasoner.process_frame([{"track_id": 4, "label": "person", "center": [50, 50], "direction": "right", "distance": "medium", "depth_score": 0.45}])
    events3 = reasoner.process_frame([{"track_id": 4, "label": "person", "center": [50, 50], "direction": "right", "distance": "far", "depth_score": 0.20}])

    all_events = events1 + events2 + events3
    assert any(e["event"] == "receding" and e["track_id"] == 4 for e in all_events)



def test_event_deduplication_state_guard():
    """Verify approaching event is NOT re-emitted every frame while state remains unchanged."""
    reasoner = TemporalReasoner(depth_threshold=0.05, history_len=5)

    f1 = [{"track_id": 5, "label": "car", "center": [100, 100], "direction": "center", "distance": "far", "depth_score": 0.10}]
    f2 = [{"track_id": 5, "label": "car", "center": [100, 100], "direction": "center", "distance": "medium", "depth_score": 0.30}]
    f3 = [{"track_id": 5, "label": "car", "center": [100, 100], "direction": "center", "distance": "close", "depth_score": 0.50}]

    ev1 = reasoner.process_frame(f1)
    assert ev1 == []

    ev2 = reasoner.process_frame(f2)
    assert len(ev2) == 1
    assert ev2[0]["event"] == "approaching"

    # Subsequent frame in same approaching state should be suppressed
    ev3 = reasoner.process_frame(f3)
    assert ev3 == []


def test_same_track_id_persisting_and_different_tracks():
    """Verify track identity uses track_id and different track IDs are maintained separately."""
    reasoner = TemporalReasoner(history_len=5, persistent_frames=3)

    # Frame 1: Object A (ID 10) & Object B (ID 20)
    det1 = [
        {"track_id": 10, "label": "chair", "center": [100, 100], "direction": "left", "distance": "close", "depth_score": 0.60},
        {"track_id": 20, "label": "person", "center": [500, 500], "direction": "right", "distance": "far", "depth_score": 0.10},
    ]
    det2 = [
        {"track_id": 10, "label": "chair", "center": [100, 100], "direction": "left", "distance": "close", "depth_score": 0.60},
        {"track_id": 20, "label": "person", "center": [500, 500], "direction": "right", "distance": "far", "depth_score": 0.10},
    ]
    det3 = [
        {"track_id": 10, "label": "chair", "center": [100, 100], "direction": "left", "distance": "close", "depth_score": 0.60},
        {"track_id": 20, "label": "person", "center": [500, 500], "direction": "right", "distance": "far", "depth_score": 0.10},
    ]

    reasoner.process_frame(det1)
    reasoner.process_frame(det2)
    ev3 = reasoner.process_frame(det3)

    # ID 10 should trigger persistent_obstacle after 3 close frames; ID 20 (far) should not.
    assert len(ev3) == 1
    assert ev3[0]["track_id"] == 10
    assert ev3[0]["event"] == "persistent_obstacle"


def test_temporary_missed_detection_vs_permanent_disappearance():
    """Verify temporary flicker (1 missed frame) does NOT trigger disappearing, but 3 missed frames do."""
    reasoner = TemporalReasoner(missing_timeout_frames=3, history_len=5)

    obs = [{"track_id": 99, "label": "bottle", "center": [10, 10], "direction": "left", "distance": "close", "depth_score": 0.60}]
    reasoner.process_frame(obs)
    reasoner.process_frame(obs)

    # Frame 3: Missed 1 frame -> no disappearing event
    ev_miss1 = reasoner.process_frame([])
    assert ev_miss1 == []

    # Frame 4: Missed 2 frames -> no disappearing event
    ev_miss2 = reasoner.process_frame([])
    assert ev_miss2 == []

    # Frame 5: Missed 3 frames -> timeout reached -> disappearing event triggered
    ev_miss3 = reasoner.process_frame([])
    assert len(ev_miss3) == 1
    assert ev_miss3[0]["event"] == "disappearing"
    assert ev_miss3[0]["track_id"] == 99


def test_noisy_small_position_changes_filtered():
    """Verify small center shifts below motion_threshold do NOT trigger moving or direction_change events."""
    reasoner = TemporalReasoner(motion_threshold=50.0, history_len=5)

    reasoner.process_frame([{"track_id": 7, "label": "cup", "center": [100, 100], "direction": "center", "distance": "medium", "depth_score": 0.40}])
    # Small jitter shift of 5 pixels
    events = reasoner.process_frame([{"track_id": 7, "label": "cup", "center": [103, 104], "direction": "center", "distance": "medium", "depth_score": 0.40}])

    assert events == []
