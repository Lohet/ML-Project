"""Phase 7 application: Audio feedback (TTS)."""

from __future__ import annotations

import logging

import cv2

from ai_assistive_vision.config import (
    CAMERA_INDEX,
    FPS_SMOOTHING,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    WINDOW_NAME,
    DEPTH_DEVICE,
    DEPTH_MODEL,
    YOLO_CONFIDENCE,
    YOLO_DEVICE,
    YOLO_MODEL,
)
from ai_assistive_vision.perception.camera import Camera, CameraConfig, CameraError, FpsCounter
from ai_assistive_vision.perception.depth import DepthEstimator, add_relative_depth
from ai_assistive_vision.perception.object_detection import ObjectDetector
from ai_assistive_vision.reasoning.spatial import add_spatial_directions
from ai_assistive_vision.reasoning.decision import extract_events
from ai_assistive_vision.reasoning.priority import EventManager
from ai_assistive_vision.feedback.audio import AudioEngine


def draw_fps(frame, fps: float) -> None:
    cv2.putText(
        frame,
        f"FPS: {fps:5.1f}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )


def draw_events(frame, events: list[dict[str, object]]) -> None:
    y_offset = 80
    for i, event in enumerate(events):
        text = f"EVENT: {event['priority'].upper()} priority {event['event']} - {event['object']} {event['direction']} ({event['distance']})"
        color = (0, 0, 255) if event['priority'] == 'high' else (0, 165, 255)
        cv2.putText(
            frame,
            text,
            (20, y_offset + i * 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )


def draw_detections(frame, detections: list[dict[str, object]]) -> None:
    for detection in detections:
        x1, y1, x2, y2 = detection["bbox"]
        label = detection["label"]
        confidence = detection["confidence"]
        direction = detection.get("direction", "unknown")
        distance = detection.get("distance", "unknown")
        text = f"{label} {confidence:.2f} {direction} {distance}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 255), 2)
        cv2.putText(
            frame,
            text,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 220, 255),
            2,
            cv2.LINE_AA,
        )


def run() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    config = CameraConfig(index=CAMERA_INDEX, width=FRAME_WIDTH, height=FRAME_HEIGHT)
    fps_counter = FpsCounter(smoothing=FPS_SMOOTHING)
    event_manager = EventManager(cooldown_seconds=3.0)
    audio_engine = AudioEngine()
    audio_engine.start()

    try:
        with Camera(config) as camera:
            detector = ObjectDetector(
                model_name=YOLO_MODEL,
                confidence=YOLO_CONFIDENCE,
                device=YOLO_DEVICE,
            )
            depth_estimator = DepthEstimator(model_type=DEPTH_MODEL, device=DEPTH_DEVICE)
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            while True:
                frame = camera.read()
                detections = detector.detect(frame)
                detections = add_spatial_directions(
                    detections=detections,
                    frame_width=frame.shape[1],
                )
                if detections:
                    depth_map = depth_estimator.estimate(frame)
                    detections = add_relative_depth(detections, depth_map)
                
                events = extract_events(detections)
                prioritized_events = event_manager.filter_and_prioritize(events)
                
                for p_event in prioritized_events:
                    speech_text = audio_engine.generate_event_text(p_event)
                    logging.info(
                        "ANNOUNCE: %s (Priority: %s)", 
                        speech_text, 
                        p_event.get('score', 0)
                    )
                    audio_engine.speak(speech_text)
                
                fps = fps_counter.update()
                draw_detections(frame, detections)
                draw_events(frame, events)
                draw_fps(frame, fps)
                cv2.imshow(WINDOW_NAME, frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
    except CameraError as error:
        logging.error("%s", error)
        return 1
    except RuntimeError as error:
        logging.error("%s", error)
        return 1
    finally:
        audio_engine.stop()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(run())
