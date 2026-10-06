import pytest

from ai_assistive_vision.perception.camera import Camera, CameraConfig, CameraError, FpsCounter


def test_fps_counter_starts_at_zero():
    counter = FpsCounter()

    assert counter.update(now=1.0) == 0.0


def test_fps_counter_calculates_smoothed_fps():
    counter = FpsCounter(smoothing=0.5)

    counter.update(now=1.0)
    assert counter.update(now=1.5) == pytest.approx(2.0)
    assert counter.update(now=2.5) == pytest.approx(1.5)


def test_camera_read_before_open_raises_error():
    camera = Camera(CameraConfig(index=0, width=640, height=480))

    with pytest.raises(CameraError, match="not been opened"):
        camera.read()
