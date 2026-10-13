"""Phase 11 application: Door detection integration."""

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
    DOOR_CONFIDENCE,
    DOOR_DEVICE,
    DOOR_ENABLED,
    DOOR_FRAME_INTERVAL,
    DOOR_MODEL_PATH,
    OCR_CONFIDENCE,
    OCR_DEVICE,
    OCR_ENABLED,
    OCR_FRAME_INTERVAL,
    OCR_LANGUAGES,
    TRACKER_ENABLED,
    TRACKER_PERSIST,
    TRACKER_TYPE,
    YOLO_CONFIDENCE,
    YOLO_DEVICE,
    YOLO_MODEL,
)
from ai_assistive_vision.perception.camera import Camera, CameraConfig, CameraError, FpsCounter
from ai_assistive_vision.perception.depth import DepthEstimator, add_relative_depth
from ai_assistive_vision.perception.door_detection import DoorDetector
from ai_assistive_vision.perception.object_detection import ObjectDetector
from ai_assistive_vision.perception.ocr import OcrWorker
from ai_assistive_vision.perception.tracking import ObjectTracker
from ai_assistive_vision.reasoning.spatial import add_spatial_directions
from ai_assistive_vision.reasoning.temporal import TemporalReasoner
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
        dist_str = f" ({event['distance']})" if 'distance' in event else ""
        text = f"EVENT: {event['priority'].upper()} priority {event['event']} - {event.get('text', event.get('object'))} {event['direction']}{dist_str}"
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
        track_id = detection.get("track_id")
        direction = detection.get("direction", "unknown")
        distance = detection.get("distance", "unknown")

        id_str = f"ID:{track_id} " if track_id is not None else ""
        text = f"{id_str}{label} {confidence:.2f} {direction} {distance}"

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


def draw_ocr_detections(frame, text_events: list[dict[str, object]]) -> None:
    for text_event in text_events:
        bbox = text_event.get("bbox")
        if not bbox:
            continue
        x1, y1, x2, y2 = bbox
        txt = text_event.get("text", "")
        conf = text_event.get("confidence", 0.0)
        direction = text_event.get("direction", "")

        label_text = f"TEXT: {txt} ({conf:.2f}) {direction}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
        cv2.putText(
            frame,
            label_text,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 255),
            2,
            cv2.LINE_AA,
        )


def draw_door_detections(frame, door_detections: list[dict]) -> None:
    """Draw door detection bounding boxes in green with confidence label."""
    for det in door_detections:
        bbox = det.get("bbox")
        if not bbox:
            continue
        x1, y1, x2, y2 = bbox
        conf = det.get("confidence", 0.0)
        source = det.get("source", "")
        direction = det.get("direction", "")
        label_text = f"DOOR ({conf:.2f}) {direction} [{source}]"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 80), 2)
        cv2.putText(
            frame,
            label_text,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 200, 80),
            2,
            cv2.LINE_AA,
        )


def run() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    config = CameraConfig(index=CAMERA_INDEX, width=FRAME_WIDTH, height=FRAME_HEIGHT)
    fps_counter = FpsCounter(smoothing=FPS_SMOOTHING)
    event_manager = EventManager(cooldown_seconds=3.0)
    temporal_reasoner = TemporalReasoner()
    audio_engine = AudioEngine()
    audio_engine.start()

    ocr_worker = (
        OcrWorker(
            languages=OCR_LANGUAGES,
            confidence=OCR_CONFIDENCE,
            device=OCR_DEVICE,
        )
        if OCR_ENABLED
        else None
    )
    if ocr_worker:
        ocr_worker.start()

    # Phase 11: Door detector (geometric fallback if no custom model is available)
    door_detector = (
        DoorDetector(
            model_path=DOOR_MODEL_PATH or None,
            confidence=DOOR_CONFIDENCE,
            device=DOOR_DEVICE,
        )
        if DOOR_ENABLED
        else None
    )
    cached_door_detections: list[dict] = []

    try:
        with Camera(config) as camera:
            detector = ObjectDetector(
                model_name=YOLO_MODEL,
                confidence=YOLO_CONFIDENCE,
                device=YOLO_DEVICE,
            )
            tracker = (
                ObjectTracker(
                    detector=detector,
                    tracker_type=TRACKER_TYPE,
                    persist=TRACKER_PERSIST,
                )
                if TRACKER_ENABLED
                else None
            )
            depth_estimator = DepthEstimator(model_type=DEPTH_MODEL, device=DEPTH_DEVICE)

            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            frame_count = 0

            while True:
                frame = camera.read()
                frame_count += 1

                if tracker and TRACKER_ENABLED:
                    detections = tracker.track(frame)
                else:
                    detections = detector.detect(frame)

                detections = add_spatial_directions(
                    detections=detections,
                    frame_width=frame.shape[1],
                )
                if detections:
                    depth_map = depth_estimator.estimate(frame)
                    detections = add_relative_depth(detections, depth_map)

                # Asynchronously submit frame to OCR worker every OCR_FRAME_INTERVAL frames
                if ocr_worker and OCR_ENABLED and (frame_count % OCR_FRAME_INTERVAL == 0):
                    ocr_worker.submit_frame(frame)

                # Consume latest available OCR results without blocking
                cached_text_events = ocr_worker.get_latest_results() if ocr_worker else []

                # Phase 11: Run door detection every DOOR_FRAME_INTERVAL frames
                if door_detector and DOOR_ENABLED and (frame_count % DOOR_FRAME_INTERVAL == 0):
                    raw_doors = door_detector.detect(frame)
                    # Attach spatial direction to each door detection
                    for d in raw_doors:
                        cx = d["center"][0]
                        fw = frame.shape[1]
                        if cx < fw / 3:
                            d["direction"] = "left"
                        elif cx < 2 * fw / 3:
                            d["direction"] = "center"
                        else:
                            d["direction"] = "right"
                    # Cache as actionable events
                    cached_door_detections = [
                        {
                            "event": "door",
                            "object": "door",
                            "direction": d.get("direction", "center"),
                            "confidence": d["confidence"],
                            "bbox": d["bbox"],
                            "priority": "medium",
                            "source": d.get("source", "geometric"),
                        }
                        for d in raw_doors
                    ]

                temporal_events = temporal_reasoner.process_frame(detections)
                events = extract_events(
                    detections,
                    temporal_events=temporal_events,
                    text_events=cached_text_events,
                    door_events=cached_door_detections,
                )
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
                draw_ocr_detections(frame, cached_text_events)
                draw_door_detections(frame, cached_door_detections)
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
        if ocr_worker:
            ocr_worker.stop()
        audio_engine.stop()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(run())
