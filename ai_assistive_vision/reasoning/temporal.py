"""Phase 9 Temporal Reasoning module.

Tracks motion trends, relative distance changes, persistence, and disappearance across video frames.
Uses relative inverse depth where higher depth_score indicates a closer object.
"""

from __future__ import annotations

from collections import deque
import math
from typing import Any, Dict, List, Optional

from ai_assistive_vision.config import (
    TEMPORAL_DEPTH_THRESHOLD,
    TEMPORAL_HISTORY_LEN,
    TEMPORAL_MISSING_TIMEOUT_FRAMES,
    TEMPORAL_MOTION_THRESHOLD,
    TEMPORAL_PERSISTENT_FRAMES,
)


class TemporalReasoner:
    """Maintains bounded history per track_id to reason about temporal events."""

    DISTANCE_ORDER = {"far": 0, "medium": 1, "close": 2, "very_close": 3}

    def __init__(
        self,
        history_len: int = TEMPORAL_HISTORY_LEN,
        motion_threshold: float = TEMPORAL_MOTION_THRESHOLD,
        depth_threshold: float = TEMPORAL_DEPTH_THRESHOLD,
        missing_timeout_frames: int = TEMPORAL_MISSING_TIMEOUT_FRAMES,
        persistent_frames: int = TEMPORAL_PERSISTENT_FRAMES,
    ) -> None:
        self.history_len = history_len
        self.motion_threshold = motion_threshold
        self.depth_threshold = depth_threshold
        self.missing_timeout_frames = missing_timeout_frames
        self.persistent_frames = persistent_frames

        self._histories: Dict[int, deque[Dict[str, Any]]] = {}
        self._last_event_state: Dict[int, str] = {}
        self._missed_counts: Dict[int, int] = {}
        self._last_observation: Dict[int, Dict[str, Any]] = {}
        self._frame_count = 0

    def process_frame(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process detections for the current frame and return temporal events."""
        self._frame_count += 1
        events: List[Dict[str, Any]] = []
        observed_track_ids = set()

        # Update observed tracks
        for detection in detections:
            track_id = detection.get("track_id")
            if track_id is None:
                continue

            observed_track_ids.add(track_id)
            self._missed_counts[track_id] = 0
            self._last_observation[track_id] = detection

            if track_id not in self._histories:
                self._histories[track_id] = deque(maxlen=self.history_len)

            obs = {
                "center": detection.get("center", [0, 0]),
                "bbox": detection.get("bbox", [0, 0, 0, 0]),
                "direction": detection.get("direction", "unknown"),
                "distance": detection.get("distance", "unknown"),
                "depth_score": float(detection.get("depth_score", 0.0)),
                "label": detection.get("label", "object"),
                "confidence": float(detection.get("confidence", 0.0)),
            }
            self._histories[track_id].append(obs)

            # Evaluate state transitions for active track
            event = self._evaluate_active_track(track_id, detection)
            if event:
                events.append(event)

        # Handle missing tracks (disappearing logic)
        missing_ids = set(self._histories.keys()) - observed_track_ids
        for track_id in list(missing_ids):
            self._missed_counts[track_id] = self._missed_counts.get(track_id, 0) + 1
            if self._missed_counts[track_id] >= self.missing_timeout_frames:
                history = self._histories[track_id]
                last_obs = self._last_observation.get(track_id, {})
                if len(history) >= 2:
                    events.append(
                        {
                            "event": "disappearing",
                            "object": last_obs.get("label", "object"),
                            "track_id": track_id,
                            "direction": last_obs.get("direction", "unknown"),
                            "distance": last_obs.get("distance", "unknown"),
                            "priority": "low",
                        }
                    )
                # Cleanup track data
                del self._histories[track_id]
                self._last_event_state.pop(track_id, None)
                self._missed_counts.pop(track_id, None)
                self._last_observation.pop(track_id, None)

        return events

    def _evaluate_active_track(
        self, track_id: int, current_detection: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        history = self._histories[track_id]
        if len(history) < 2:
            return None

        first = history[0]
        last = history[-1]

        # 1. Depth score / distance bucket evaluation
        # Depth score in perception/depth.py is inverse depth: LARGER = CLOSER
        depth_delta = last["depth_score"] - first["depth_score"]
        first_dist_rank = self.DISTANCE_ORDER.get(first["distance"], 0)
        last_dist_rank = self.DISTANCE_ORDER.get(last["distance"], 0)

        is_approaching = (
            depth_delta >= self.depth_threshold or last_dist_rank > first_dist_rank
        )
        is_receding = (
            depth_delta <= -self.depth_threshold or last_dist_rank < first_dist_rank
        )

        # 2. Movement evaluation
        dx = last["center"][0] - first["center"][0]
        dy = last["center"][1] - first["center"][1]
        center_dist = math.hypot(dx, dy)
        is_moving = center_dist >= self.motion_threshold

        # 3. Direction change evaluation
        has_direction_change = (
            first["direction"] != "unknown"
            and last["direction"] != "unknown"
            and first["direction"] != last["direction"]
            and abs(dx) >= self.motion_threshold
        )

        # 4. Persistent obstacle evaluation
        is_persistent_obstacle = (
            len(history) >= self.persistent_frames
            and all(obs["distance"] in ("close", "very_close") for obs in list(history)[-self.persistent_frames:])
        )

        # Determine target temporal state
        target_state: Optional[str] = None
        event_priority = "medium"

        if is_approaching:
            target_state = "approaching"
            event_priority = "high" if last["distance"] in ("close", "very_close") else "medium"
        elif is_persistent_obstacle:
            target_state = "persistent_obstacle"
            event_priority = "high"
        elif has_direction_change:
            target_state = "direction_change"
            event_priority = "medium"
        elif is_receding:
            target_state = "receding"
            event_priority = "low"
        elif is_moving:
            target_state = "moving"
            event_priority = "medium"

        # State transition guard: emit ONLY when state changes
        last_state = self._last_event_state.get(track_id)
        if target_state and target_state != last_state:
            self._last_event_state[track_id] = target_state
            return {
                "event": target_state,
                "object": current_detection["label"],
                "track_id": track_id,
                "direction": current_detection.get("direction", "unknown"),
                "distance": current_detection.get("distance", "unknown"),
                "depth_score": current_detection.get("depth_score", 0.0),
                "priority": event_priority,
            }

        return None

    def reset(self) -> None:
        """Reset all temporal history and track states."""
        self._histories.clear()
        self._last_event_state.clear()
        self._missed_counts.clear()
        self._last_observation.clear()
        self._frame_count = 0
