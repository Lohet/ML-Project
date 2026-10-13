import pytest

from ai_assistive_vision.perception.object_detection import ObjectDetector, parse_yolo_result


class FakeScalar:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value


class FakeVector:
    def __init__(self, values):
        self.values = values

    def tolist(self):
        return self.values


class FakeBox:
    def __init__(self):
        self.xyxy = [FakeVector([10.2, 20.6, 110.7, 220.1])]
        self.cls = FakeScalar(0)
        self.conf = FakeScalar(0.923)


class FakeResult:
    names = {0: "person"}
    boxes = [FakeBox()]


class FakeModel:
    def predict(self, frame, conf, device, verbose):
        return [FakeResult()]


def test_parse_yolo_result_returns_project_detection_format():
    detections = parse_yolo_result(FakeResult())

    assert len(detections) == 1
    assert detections[0].as_dict() == {
        "track_id": None,
        "label": "person",
        "confidence": pytest.approx(0.923),
        "bbox": [10, 21, 111, 220],
        "center": [60, 120],
    }



def test_object_detector_detect_uses_normalized_format(monkeypatch):
    monkeypatch.setattr(ObjectDetector, "_load_model", lambda self: FakeModel())
    monkeypatch.setattr(ObjectDetector, "_resolve_device", staticmethod(lambda device: "cpu"))
    detector = ObjectDetector(model_name="fake.pt", confidence=0.5)

    detections = detector.detect(frame=object())

    assert detections[0]["label"] == "person"
    assert detections[0]["bbox"] == [10, 21, 111, 220]


def test_object_detector_rejects_invalid_confidence():
    with pytest.raises(ValueError, match="confidence"):
        ObjectDetector(model_name="fake.pt", confidence=1.5)
