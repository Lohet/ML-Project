"""Configuration for the AI Assistive Vision prototype."""

CAMERA_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
WINDOW_NAME = "AI Assistive Vision - Phase 12"
FPS_SMOOTHING = 0.9

YOLO_MODEL = "ai_assistive_vision/models/yolov8s.pt"
YOLO_CONFIDENCE = 0.50
YOLO_DEVICE = "auto"

TRACKER_TYPE = "bytetrack.yaml"
TRACKER_ENABLED = True
TRACKER_PERSIST = True

DEPTH_MODEL = "MiDaS_small"
DEPTH_DEVICE = "auto"

TEMPORAL_HISTORY_LEN = 10
TEMPORAL_MOTION_THRESHOLD = 30
TEMPORAL_DEPTH_THRESHOLD = 0.08
TEMPORAL_MISSING_TIMEOUT_FRAMES = 5
TEMPORAL_PERSISTENT_FRAMES = 5

OCR_ENABLED = True
OCR_CONFIDENCE = 0.50
OCR_FRAME_INTERVAL = 10
OCR_DEVICE = "auto"
OCR_LANGUAGES = ["en"]

# Phase 11 — Door Detection
# Set DOOR_MODEL_PATH to the path of a fine-tuned door YOLO model (.pt).
# If the file does not exist, the geometric fallback detector is used.
DOOR_ENABLED = True
DOOR_MODEL_PATH = ""   # e.g. "ai_assistive_vision/models/door_yolov8n.pt"
DOOR_CONFIDENCE = 0.45
DOOR_DEVICE = "auto"
DOOR_FRAME_INTERVAL = 3  # Run door detection every N frames to save compute

# Phase 12 — Stair Detection
# Set STAIR_MODEL_PATH to the path of a fine-tuned stair YOLO model (.pt).
# If the file does not exist, the geometric horizontal-line fallback is used.
STAIR_ENABLED = True
STAIR_MODEL_PATH = ""    # e.g. "ai_assistive_vision/models/stair_yolov8n.pt"
STAIR_CONFIDENCE = 0.35  # Lower threshold; stairs below 0.55 → "Possible stairs"
STAIR_UNCERTAIN_THRESHOLD = 0.55   # Below this → "Possible stairs ahead."
STAIR_DEVICE = "auto"
STAIR_FRAME_INTERVAL = 5  # Run stair detection every N frames
