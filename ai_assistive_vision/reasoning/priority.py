"""Phase 6 priority and deduplication logic."""

from typing import Any, Dict, List, Optional
import time


class EventManager:
    """Manages events, filtering out duplicates and prioritizing important ones."""

    # Higher number means higher priority for sorting
    PRIORITY_SCORES = {
        "immediate_obstacle": 100,
        "vehicle": 90,
        "person": 80,
        "stairs": 70,
        "door": 60,
        "important_object": 50,
        "text": 40,
        "general": 10,
    }

    VEHICLE_LABELS = {"car", "truck", "bus", "motorcycle", "bicycle"}
    IMPORTANT_OBJECT_LABELS = {"chair", "table", "sofa", "bed", "toilet"}

    def __init__(self, cooldown_seconds: float = 3.0):
        self.cooldown_seconds = cooldown_seconds
        # Stores the last time an event signature was announced
        self._history: Dict[str, float] = {}

    def _determine_category(self, event: Dict[str, Any]) -> str:
        """Determine the category of an event for prioritization."""
        
        # Immediate obstacle or high-priority temporal event
        if event.get("priority") == "high" or event.get("event") == "approaching":
            return "immediate_obstacle"

        if event.get("event") == "text":
            return "text"

        # Phase 11: door detection events
        if event.get("event") == "door":
            return "door"

        # Phase 12: stair detection events
        if event.get("event") == "stairs":
            return "stairs"

        # Phase 13: vehicle awareness events
        if str(event.get("event", "")).startswith("vehicle_"):
            return "vehicle"

        obj = event.get("object", "")
        if obj in self.VEHICLE_LABELS or obj == "vehicle":
            return "vehicle"
        if obj == "person":
            return "person"
        if obj == "stairs":
            return "stairs"
        if obj == "door":
            return "door"
        if obj in self.IMPORTANT_OBJECT_LABELS:
            return "important_object"
            
        return "general"

    def _get_signature(self, event: Dict[str, Any]) -> str:
        """Create a unique signature for an event to handle deduplication."""
        event_type = event.get("event", "obstacle")
        if event_type == "text":
            text_val = event.get("text", event.get("object", "unknown"))
            direction = event.get("direction", "unknown")
            return f"text_{text_val}_{direction}"

        if event_type == "stairs":
            direction = event.get("direction", "unknown")
            uncertain = event.get("uncertain", False)
            return f"stairs_{direction}_{uncertain}"

        if str(event_type).startswith("vehicle_"):
            v_type = event.get("vehicle_type", "vehicle")
            direction = event.get("direction", "unknown")
            distance = event.get("distance", "unknown")
            return f"{event_type}_{v_type}_{direction}_{distance}"

        track_id = event.get("track_id", "untracked")
        obj = event.get("object", "unknown")
        direction = event.get("direction", "unknown")
        distance = event.get("distance", "unknown")
        return f"{event_type}_{track_id}_{obj}_{direction}_{distance}"



    def filter_and_prioritize(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter out duplicate events and sort by priority."""
        
        current_time = time.time()
        valid_events = []

        for event in events:
            category = self._determine_category(event)
            event["category"] = category
            event["score"] = self.PRIORITY_SCORES.get(category, 0)
            
            signature = self._get_signature(event)
            last_seen = self._history.get(signature, 0.0)
            
            # Announce if it's new or the cooldown has expired
            if current_time - last_seen > self.cooldown_seconds:
                valid_events.append((event, signature))

        # Sort by score descending
        valid_events.sort(key=lambda x: x[0]["score"], reverse=True)
        
        final_events = []
        for event, signature in valid_events:
            self._history[signature] = current_time
            final_events.append(event)
            
        return final_events
