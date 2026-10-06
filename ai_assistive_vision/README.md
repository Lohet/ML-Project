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
  └── Depth Estimation  (MiDaS)
  ↓
Spatial Reasoning       (left / center / right)
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
| 8 | Object tracking (ByteTrack / YOLO tracking) | 🔜 Next |
| 9 | Temporal reasoning | 🔜 Planned |
| 10 | OCR / text recognition | 🔜 Planned |
| 11–17 | Door/stair detection, haptics, optimization | 🔜 Planned |

---

## Detection Output Shape

```python
{
    "label":      "person",
    "confidence": 0.92,
    "bbox":       [x1, y1, x2, y2],
    "center":     [cx, cy],
    "direction":  "center",      # left / center / right
    "distance":   "very_close",  # far / medium / close / very_close
    "depth_score": 0.83,
}
```

### Audio output examples

```
"Person ahead."
"Car on the left."
"Obstacle on the right."
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
│   └── depth.py            ← MiDaS relative depth
│
├── reasoning/
│   ├── spatial.py          ← left / center / right
│   ├── decision.py         ← Obstacle event extraction
│   └── priority.py         ← Priority scoring & dedup cooldown
│
├── feedback/
│   └── audio.py            ← Non-blocking pyttsx3 TTS
│
├── models/
│   ├── yolov8s.pt          ← YOLOv8s weights (download separately)
│   └── (MiDaS downloaded via torch.hub on first run)
│
└── tests/
    ├── test_camera.py
    ├── test_object_detection.py
    ├── test_spatial.py
    ├── test_depth.py
    ├── test_decision.py
    ├── test_priority.py
    └── test_audio.py
```

---

## Requirements

| Package | Purpose |
|---------|---------|
| `opencv-python` | Camera capture & display |
| `ultralytics` | YOLOv8s object detection |
| `torch` + `torchvision` | Model inference (CUDA recommended) |
| `timm` | MiDaS transform dependencies |
| `numpy` | Array operations |
| `pyttsx3` | Offline text-to-speech |
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
C:\Users\Lohet.S.Ratnakar\anaconda3\python.exe -m ai_assistive_vision.app
```

### What to expect with a working webcam

- A live camera window opens titled **"AI Assistive Vision - Phase 7"**.
- FPS is displayed in the top-left corner.
- Detected objects show bounding boxes with `label confidence direction distance`.
- Obstacle events are overlaid on the frame in orange/red.
- When a close or very-close object is detected, you will **hear a short spoken alert** (e.g. *"Person ahead."*).
- The same event is suppressed for **3 seconds** to avoid repetition.
- Press `q` or `Esc` to exit cleanly.

### Troubleshooting

| Problem | Fix |
|---------|-----|
| No camera opens | Change `CAMERA_INDEX` in `config.py` |
| No GPU inference | Reinstall PyTorch CUDA wheel |
| No audio | Check speakers; ensure `pyttsx3` is installed |
| Slow FPS | Reduce `FRAME_WIDTH`/`FRAME_HEIGHT` in `config.py` |

---

## Configuration

All tunable parameters are in [`config.py`](config.py):

```python
CAMERA_INDEX    = 0
FRAME_WIDTH     = 1280
FRAME_HEIGHT    = 720
YOLO_MODEL      = "ai_assistive_vision/models/yolov8s.pt"
YOLO_CONFIDENCE = 0.50
YOLO_DEVICE     = "auto"   # auto → cuda:0 if available, else cpu
DEPTH_MODEL     = "MiDaS_small"
DEPTH_DEVICE    = "auto"
```

---

## Tests

```powershell
C:\Users\Lohet.S.Ratnakar\anaconda3\python.exe -m pytest -p no:cacheprovider ai_assistive_vision/tests
```

Expected result: **19 passed**.

---

## Environment

Tested on:

- **OS**: Windows 11
- **Python**: 3.13.9 (Anaconda)
- **GPU**: NVIDIA GeForce RTX 4060 Laptop GPU (8 GB VRAM)
- **CUDA**: 12.8 (PyTorch `torch==2.11.0+cu128`)
- **OpenCV**: 5.0.0
- **Ultralytics**: 8.4.173
