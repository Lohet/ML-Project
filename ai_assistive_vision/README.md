# AI Assistive Vision

A modular, real-time AI assistive system for people with cortical blindness.

The system acts as a **"second pair of eyes"** by using a camera to understand the user's surroundings and converting important visual information into concise **audio feedback**.

> ⚠️ This is a **research/demo prototype**, not a certified medical device or guaranteed mobility aid.

---

## Architecture

```
Camera
  ↓
AI Perception
  ├── Object Detection  (YOLOv8s)
  ├── Object Tracking   (ByteTrack via Ultralytics)
  ├── Depth Estimation  (MiDaS)
  ├── OCR               (EasyOCR)
  └── Door Detection    (Geometric contour / custom YOLO)
  ↓
Spatial Reasoning       (left / center / right)
  ↓
Temporal Reasoning      (approaching, receding, persistent, disappearing, moving)
  ↓
Obstacle Reasoning      (close / very_close → obstacle event)
  ↓
Priority & Dedup        (cooldown, priority score)
  ↓
Audio Feedback          (pyttsx3 TTS — offline)
```

---

## Completed Phases

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Camera capture & FPS display | ✅ Done |
| 2 | YOLO object detection (YOLOv8s, CUDA) | ✅ Done |
| 3 | Spatial reasoning (left / center / right) | ✅ Done |
| 4 | Relative depth estimation (MiDaS) | ✅ Done |
| 5 | Obstacle event extraction | ✅ Done |
| 6 | Priority scoring & deduplication | ✅ Done |
| 7 | Audio feedback (offline TTS via pyttsx3) | ✅ Done |
| 8 | Object tracking (ByteTrack via Ultralytics) | ✅ Done |
| 9 | Temporal reasoning (approaching, receding, persistent, disappearing) | ✅ Done |
| 10 | OCR / text recognition (EasyOCR, async worker) | ✅ Done |
| 11 | Door detection (geometric contour + custom YOLO support) | ✅ Done |
| 12–17 | Stair detection, vehicle awareness, haptics, optimization | 🔜 Planned |

---

## Detection Output Shape

```python
{
    "track_id":   3,             # int (or None if untracked)
    "label":      "person",
    "confidence": 0.92,
    "bbox":       [x1, y1, x2, y2],
    "center":     [cx, cy],
    "direction":  "center",      # left / center / right
    "distance":   "very_close",  # far / medium / close / very_close
    "depth_score": 0.83,
}
```

### Door Detection Output Shape

```python
{
    "event":      "door",
    "object":     "door",
    "direction":  "center",      # left / center / right
    "confidence": 0.45,          # 0.45 fixed for geometric; actual model score for YOLO
    "bbox":       [x1, y1, x2, y2],
    "priority":   "medium",
    "source":     "geometric",   # "geometric" or "yolo"
}
```

### Spoken Output Examples

```
"Person approaching ahead."
"Car moving on the left."
"Car cleared."
"Obstacle ahead."
"Door ahead."
"Door on the left."
"Exit ahead."
```

---

## Project Structure

```
ai_assistive_vision/
├── app.py                  ← Main entry point
├── config.py               ← All tunable parameters
├── requirements.txt
├── README.md
│
├── perception/
│   ├── camera.py           ← Webcam capture & FPS
│   ├── object_detection.py ← YOLOv8s (Ultralytics)
│   ├── tracking.py         ← ByteTrack tracker wrapper
│   ├── depth.py            ← MiDaS relative depth
│   ├── ocr.py              ← EasyOCR async text recognition
│   └── door_detection.py   ← Phase 11: Geometric + YOLO door detector
│
├── reasoning/
│   ├── spatial.py          ← left / center / right
│   ├── temporal.py         ← Bounded track history & temporal event reasoning
│   ├── decision.py         ← Actionable event extraction (obstacles, doors, OCR, temporal)
│   └── priority.py         ← Priority scoring & dedup cooldown
│
├── feedback/
│   └── audio.py            ← Non-blocking pyttsx3 TTS
│
├── models/
│   ├── yolov8s.pt          ← YOLOv8s weights (download separately)
│   ├── door_yolov8n.pt     ← (optional) fine-tuned door detector
│   └── (MiDaS downloaded via torch.hub on first run)
│
└── tests/
    ├── test_camera.py
    ├── test_object_detection.py
    ├── test_tracking.py
    ├── test_spatial.py
    ├── test_depth.py
    ├── test_temporal.py
    ├── test_decision.py
    ├── test_priority.py
    ├── test_audio.py
    ├── test_ocr.py
    └── test_door_detection.py
```

---

## Requirements

| Package | Purpose |
|---------|---------|
| `opencv-python` | Camera capture & display |
| `ultralytics` | YOLOv8s object detection & ByteTrack tracking |
| `torch` + `torchvision` | Model inference (CUDA recommended) |
| `timm` | MiDaS transform dependencies |
| `numpy` | Array operations |
| `pyttsx3` | Offline text-to-speech |
| `easyocr` | OCR text recognition |
| `pytest` | Tests |

Install dependencies:

```powershell
pip install -r ai_assistive_vision/requirements.txt
```

For CUDA (RTX 4060 / CUDA 12.8):

```powershell
pip install --upgrade torch==2.11.0+cu128 torchvision==0.26.0+cu128 --index-url https://download.pytorch.org/whl/cu128
```

---

## Run

From the project root:

```powershell
python -m ai_assistive_vision.app
```

### What to expect with a working webcam

- A live camera window opens titled **"AI Assistive Vision - Phase 11"**.
- FPS is displayed in the top-left corner.
- Detected objects show bounding boxes with `ID:# label confidence direction distance`.
- Temporal events (`approaching`, `moving`, `persistent_obstacle`, `disappearing`) are processed.
- **Door candidates** are highlighted with **green** bounding boxes labelled `DOOR (conf) direction [source]`.
- When an object approaches, presents an immediate obstacle, or a door is detected, you will **hear a short spoken alert** (e.g. *"Door ahead."*, *"Person approaching ahead."*).
- The same state event is suppressed until state changes or the 3-second cooldown expires.
- Press `q` or `Esc` to exit cleanly.

### Door Detection Notes

- By default, the **geometric contour detector** is active. It uses edge detection + aspect-ratio heuristics to find tall rectangular shapes consistent with doors.
- It will have **false positives** in complex scenes — suitable for research/demo use.
- For production accuracy, place a fine-tuned YOLOv8 door model at `ai_assistive_vision/models/door_yolov8n.pt` and set `DOOR_MODEL_PATH` in `config.py`.
- The detector automatically falls back to geometric mode if the model file is absent.

---

## Configuration

All tunable parameters are in [`config.py`](config.py):

```python
CAMERA_INDEX                    = 0
FRAME_WIDTH                     = 1280
FRAME_HEIGHT                    = 720
YOLO_MODEL                      = "ai_assistive_vision/models/yolov8s.pt"
YOLO_CONFIDENCE                 = 0.50
YOLO_DEVICE                     = "auto"
TRACKER_TYPE                    = "bytetrack.yaml"
TRACKER_ENABLED                 = True
TRACKER_PERSIST                 = True
DEPTH_MODEL                     = "MiDaS_small"
DEPTH_DEVICE                    = "auto"
TEMPORAL_HISTORY_LEN            = 10
TEMPORAL_MOTION_THRESHOLD       = 30
TEMPORAL_DEPTH_THRESHOLD        = 0.08
TEMPORAL_MISSING_TIMEOUT_FRAMES = 5
TEMPORAL_PERSISTENT_FRAMES      = 5
OCR_ENABLED                     = True
OCR_CONFIDENCE                  = 0.50
OCR_FRAME_INTERVAL              = 10
DOOR_ENABLED                    = True
DOOR_MODEL_PATH                 = ""    # Set to path of .pt file to use YOLO backend
DOOR_CONFIDENCE                 = 0.45
DOOR_DEVICE                     = "auto"
DOOR_FRAME_INTERVAL             = 3     # Runs every 3 frames to save compute
```

---

## Tests

```powershell
pytest -p no:cacheprovider ai_assistive_vision/tests
```

Expected result: **53 passed**.

---

## Environment

Tested on:

- **OS**: Windows 11
- **Python**: 3.13.9
- **GPU**: NVIDIA GeForce RTX 4060 Laptop GPU (8 GB VRAM)
- **OpenCV**: 5.0.0
- **Ultralytics**: 8.4.174
