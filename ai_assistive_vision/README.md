# AI Assistive Vision

A modular, real-time AI assistive system for people with cortical blindness.

The system acts as a **"second pair of eyes"** by using a camera to understand the user's surroundings and converting important visual information into concise **multimodal audio and haptic feedback**.

> ⚠️ This is a **research/demo prototype**, not a certified medical device or guaranteed mobility aid.

---

## Architecture

```
Camera (1280x720 / 60 FPS target)
  ↓
AI Perception
  ├── Object Detection    (YOLOv8s)
  ├── Multi-Object Track  (ByteTrack via Ultralytics)
  ├── Depth Estimation    (MiDaS relative depth)
  ├── Environmental OCR   (EasyOCR background worker)
  ├── Door Detection      (Geometric contour / custom YOLO)
  └── Stair Detection     (Geometric Hough step clustering / custom YOLO)
  ↓
Reasoning & Intelligence
  ├── Spatial Reasoning   (horizontal sector: left / center / right)
  ├── Temporal Reasoning  (approaching, receding, persistent, disappearing)
  ├── Vehicle Awareness   (approach speed, lateral crossing, danger proximity)
  ├── Scene Understanding (indoor/outdoor classifier & mental map summary)
  └── Obstacle Decision   (close / very_close proximity events)
  ↓
Priority & Deduplication  (multi-tier score ranking & cooldown deduplication)
  ↓
Multimodal Feedback
  ├── Audio Speech (TTS)  (pyttsx3 offline text-to-speech engine)
  └── Haptic Vibration    (3-channel directional motors: left / center / right)
  ↓
Optimization & Metrics    (stage latency, end-to-end FPS, RAM & GPU VRAM profiler)
```

---

## Completed Phases

| Phase | Description | Status |
|---|---|---|
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
| 12 | Stair detection (geometric line clustering + custom YOLO support, uncertainty flag) | ✅ Done |
| 13 | Vehicle awareness (cars, buses, trucks, motorcycles, bicycles; trajectory vectors) | ✅ Done |
| 14 | Scene understanding (indoor/outdoor classifier + executive mental map summary) | ✅ Done |
| 15 | Haptic feedback (3-channel Left/Center/Right, distance-to-intensity mapping) | ✅ Done |
| 16 | Audio + Haptic fusion (synchronized multimodal safety hazards, channel throttling) | ✅ Done |
| 17 | Real-time optimization & latency profiling (FPS, stage timings, RAM & GPU VRAM) | ✅ Done |
| 18 | Master system evaluation & verification suite (105 passed tests) | ✅ Done |

---

## Detection & Event Schemas

### Standard Object Tracking

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

### Vehicle Awareness Event

```python
{
    "event":        "vehicle_approaching",
    "object":       "vehicle",
    "vehicle_type": "car",
    "direction":    "left",
    "distance":     "close",
    "priority":     "high",
    "is_fast":      False,
}
```

### Stair Hazard Event

```python
{
    "event":      "stairs",
    "object":     "stairs",
    "direction":  "center",
    "confidence": 0.40,
    "bbox":       [x1, y1, x2, y2],
    "priority":   "medium",
    "uncertain":  True,          # Speech states "Possible stairs" when True
    "source":     "geometric",
}
```

### Scene Understanding Summary

```python
{
    "event":       "scene_summary",
    "environment": "indoor",    # indoor / outdoor
    "text":        "Indoor environment. Person ahead. Door on the right. Chair on the left.",
    "priority":    "low",
}
```

### Multimodal Spoken & Haptic Examples

| Situation | Spoken Audio Output | Haptic Feedback |
|---|---|---|
| Car rapidly approaching from the left | *"Fast vehicle approaching from the left."* | Left motor → Rapid vibration |
| Obstacle ahead in close proximity | *"Obstacle ahead."* | Center motor → Strong vibration |
| Stairs detected ahead (uncertain) | *"Possible stairs ahead."* | Center motor → Moderate vibration |
| Door located on the right | *"Door on the right."* | Audio only (prevents tactile clutter) |
| High-priority emergency exit sign | *"Emergency Exit ahead."* | Audio only |
| Vehicle crossing field of view | *"Vehicle crossing from left to right."* | Audio only |

---

## Project Structure

```
ai_assistive_vision/
├── app.py                  ← Main entry point & real-time loop
├── config.py               ← Centralized parameters
├── requirements.txt
├── README.md
│
├── perception/
│   ├── camera.py           ← Webcam capture & FPS counter
│   ├── object_detection.py ← YOLOv8s (Ultralytics)
│   ├── tracking.py         ← ByteTrack tracker wrapper
│   ├── depth.py            ← MiDaS relative depth estimation
│   ├── ocr.py              ← EasyOCR async worker & keyword filtering
│   ├── door_detection.py   ← Phase 11: Geometric contour + YOLO door detector
│   └── stair_detection.py  ← Phase 12: Geometric Hough + YOLO stair detector
│
├── reasoning/
│   ├── spatial.py          ← Left / center / right sectoral mapping
│   ├── temporal.py         ← Multi-frame motion vectors & state reasoning
│   ├── vehicle_awareness.py← Phase 13: Vehicle trajectory & speed awareness
│   ├── scene_understanding.py ← Phase 14: Indoor/outdoor classifier & summarizer
│   ├── decision.py         ← Actionable event extraction
│   └── priority.py         ← Priority scoring & cooldown deduplication
│
├── feedback/
│   ├── audio.py            ← Offline pyttsx3 speech synthesizer
│   ├── haptic.py           ← Phase 15: 3-channel directional haptic motor interface
│   └── fusion.py           ← Phase 16: Multimodal audio + haptic fusion engine
│
├── optimization/
│   ├── __init__.py
│   └── profiler.py         ← Phase 17: Latency, FPS, RAM & VRAM profiler
│
├── models/
│   ├── yolov8s.pt          ← YOLOv8s weights
│   ├── door_yolov8n.pt     ← (optional) custom door weights
│   └── stair_yolov8n.pt    ← (optional) custom stair weights
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
    ├── test_door_detection.py
    ├── test_stair_detection.py
    ├── test_vehicle_awareness.py
    ├── test_scene_understanding.py
    ├── test_haptic.py
    ├── test_fusion.py
    ├── test_profiler.py
    └── test_phase18_system_evaluation.py
```

---

## Requirements & Installation

```powershell
pip install -r ai_assistive_vision/requirements.txt
```

For NVIDIA GPU acceleration (CUDA 12.8 / RTX 4060):

```powershell
pip install --upgrade torch==2.11.0+cu128 torchvision==0.26.0+cu128 --index-url https://download.pytorch.org/whl/cu128
```

---

## Run

From the project root:

```powershell
python -m ai_assistive_vision.app
```

Press `q` or `Esc` to cleanly shut down the application.

---

## Test Suite Execution

Run all automated unit, integration, and evaluation tests:

```powershell
pytest -p no:cacheprovider ai_assistive_vision/tests -v
```

Expected output: **105 passed** (100% test pass rate across all 18 phases).
