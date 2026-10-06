"""Spatial reasoning for Phase 3."""

from __future__ import annotations

from typing import Literal


Direction = Literal["left", "center", "right"]


def classify_horizontal_direction(center_x: int | float, frame_width: int) -> Direction:
    """Classify an object's horizontal position into left, center, or right."""

    if frame_width <= 0:
        raise ValueError("frame_width must be greater than zero")

    left_boundary = frame_width / 3
    right_boundary = (frame_width * 2) / 3

    if center_x < left_boundary:
        return "left"
    if center_x < right_boundary:
        return "center"
    return "right"


def add_spatial_directions(
    detections: list[dict[str, object]],
    frame_width: int,
) -> list[dict[str, object]]:
    """Return detections with a direction field added."""

    enriched = []
    for detection in detections:
        center = detection["center"]
        center_x = center[0]
        enriched_detection = dict(detection)
        enriched_detection["direction"] = classify_horizontal_direction(
            center_x=center_x,
            frame_width=frame_width,
        )
        enriched.append(enriched_detection)
    return enriched
