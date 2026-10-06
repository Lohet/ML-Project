import pytest

from ai_assistive_vision.reasoning.spatial import (
    add_spatial_directions,
    classify_horizontal_direction,
)


def test_classify_horizontal_direction_left_center_right():
    assert classify_horizontal_direction(center_x=10, frame_width=300) == "left"
    assert classify_horizontal_direction(center_x=150, frame_width=300) == "center"
    assert classify_horizontal_direction(center_x=250, frame_width=300) == "right"


def test_classify_horizontal_direction_rejects_invalid_width():
    with pytest.raises(ValueError, match="frame_width"):
        classify_horizontal_direction(center_x=10, frame_width=0)


def test_add_spatial_directions_preserves_detection_fields():
    detections = [
        {
            "label": "person",
            "confidence": 0.9,
            "bbox": [0, 0, 20, 20],
            "center": [10, 10],
        },
        {
            "label": "chair",
            "confidence": 0.8,
            "bbox": [240, 0, 290, 20],
            "center": [265, 10],
        },
    ]

    enriched = add_spatial_directions(detections, frame_width=300)

    assert enriched[0]["direction"] == "left"
    assert enriched[1]["direction"] == "right"
    assert "direction" not in detections[0]
