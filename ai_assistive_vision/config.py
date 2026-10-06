"""Configuration for the AI Assistive Vision prototype."""

CAMERA_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
WINDOW_NAME = "AI Assistive Vision - Phase 7"
FPS_SMOOTHING = 0.9

YOLO_MODEL = "ai_assistive_vision/models/yolov8s.pt"
YOLO_CONFIDENCE = 0.50
YOLO_DEVICE = "auto"

DEPTH_MODEL = "MiDaS_small"
DEPTH_DEVICE = "auto"
