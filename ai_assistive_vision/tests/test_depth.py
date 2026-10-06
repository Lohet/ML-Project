import numpy as np
import pytest

from ai_assistive_vision.perception.depth import (
    add_relative_depth,
    classify_relative_distance,
    normalize_depth_map,
)


def test_normalize_depth_map_scales_values_to_zero_one():
    depth_map = np.array([[2.0, 4.0], [6.0, 10.0]], dtype=np.float32)

    normalized = normalize_depth_map(depth_map)

    assert normalized.min() == 0.0
    assert normalized.max() == 1.0


def test_normalize_depth_map_handles_constant_values():
    depth_map = np.full((2, 2), 5.0, dtype=np.float32)

    normalized = normalize_depth_map(depth_map)

    assert np.array_equal(normalized, np.zeros((2, 2), dtype=np.float32))


def test_classify_relative_distance_buckets():
    assert classify_relative_distance(0.10) == "far"
    assert classify_relative_distance(0.30) == "medium"
    assert classify_relative_distance(0.60) == "close"
    assert classify_relative_distance(0.90) == "very_close"


def test_add_relative_depth_uses_detection_bbox_median():
    depth_map = np.array(
        [
            [0.1, 0.1, 0.8, 0.8],
            [0.1, 0.1, 0.8, 0.8],
            [0.1, 0.1, 0.8, 0.8],
            [0.1, 0.1, 0.8, 0.8],
        ],
        dtype=np.float32,
    )
    detections = [
        {
            "label": "person",
            "confidence": 0.9,
            "bbox": [2, 0, 4, 4],
            "center": [3, 2],
            "direction": "right",
        }
    ]

    enriched = add_relative_depth(detections, depth_map)

    assert enriched[0]["distance"] == "very_close"
    assert enriched[0]["depth_score"] == pytest.approx(0.8)
    assert "distance" not in detections[0]
