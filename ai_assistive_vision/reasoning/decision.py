"""Decision making and event generation for Phase 5.

Extended in Phase 11 to include door events.
Extended in Phase 12 to include stair events.
"""

from typing import Any, Dict, List, Optional


def extract_events(
    detections: List[Dict[str, Any]],
    temporal_events: Optional[List[Dict[str, Any]]] = None,
    text_events: Optional[List[Dict[str, Any]]] = None,
    door_events: Optional[List[Dict[str, Any]]] = None,
    stair_events: Optional[List[Dict[str, Any]]] = None,
    vehicle_events: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Convert detections, temporal events, OCR text events, door events,
    stair events, and vehicle events into actionable system events.
    """
    events: List[Dict[str, Any]] = []

    # 1. Static obstacle events based on distance
    for detection in detections:
        distance = detection.get("distance")

        if distance in ("close", "very_close"):
            events.append(
                {
                    "event": "obstacle",
                    "object": detection["label"],
                    "track_id": detection.get("track_id"),
                    "direction": detection.get("direction", "unknown"),
                    "distance": distance,
                    "priority": "high" if distance == "very_close" else "medium",
                }
            )

    # 2. Append temporal reasoning events
    if temporal_events:
        events.extend(temporal_events)

    # 3. Append OCR text events
    if text_events:
        events.extend(text_events)

    # 4. Append door detection events (Phase 11)
    if door_events:
        events.extend(door_events)

    # 5. Append stair detection events (Phase 12)
    if stair_events:
        events.extend(stair_events)

    # 6. Append vehicle awareness events (Phase 13)
    if vehicle_events:
        events.extend(vehicle_events)

    return events


