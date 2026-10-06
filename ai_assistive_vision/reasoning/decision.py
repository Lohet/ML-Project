"""Decision making and event generation for Phase 5."""

from typing import Any, Dict, List


def extract_events(detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert detections into actionable events.
    
    Currently implements basic obstacle detection based on distance.
    """
    events = []
    
    for detection in detections:
        distance = detection.get("distance")
        
        if distance in ("close", "very_close"):
            events.append(
                {
                    "event": "obstacle",
                    "object": detection["label"],
                    "direction": detection.get("direction", "unknown"),
                    "distance": distance,
                    "priority": "high" if distance == "very_close" else "medium",
                }
            )
            
    return events
