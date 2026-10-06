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
        
        # Phase 5 obstacle logic added priority: 'high' or 'medium'
        if event.get("priority") == "high":
            return "immediate_obstacle"

        obj = event.get("object", "")
        if obj in self.VEHICLE_LABELS:
            return "vehicle"
        if obj == "person":
            return "person"
        if obj in self.IMPORTANT_OBJECT_LABELS:
            return "important_object"
            
        return "general"

    def _get_signature(self, event: Dict[str, Any]) -> str:
        """Create a unique signature for an event to handle deduplication."""
        # Simple deduplication based on object label, direction, and distance
        # e.g. "person_center_close"
        obj = event.get("object", "unknown")
        direction = event.get("direction", "unknown")
        distance = event.get("distance", "unknown")
        return f"{obj}_{direction}_{distance}"

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
