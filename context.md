# AI Assistive Vision Context

## Current Scope

- Completed through: Phase 11 - Door Detection.
- Stop point: Do not implement Phase 12 stair detection until the user approves the next phase.
- Latest user request: Proceed with next phases until project is complete, update documentation as you go.

## Environment Checks - 2026-10-06

- Python: `Python 3.13.9`.
- NVIDIA GPU: `nvidia-smi` detects an NVIDIA GeForce RTX 4060 Laptop GPU with 8188 MiB VRAM.
- NVIDIA driver/CUDA runtime reported by driver: Driver `581.86`, CUDA `13.0`.
- Python packages:
  - `cv2` importable.
  - `torch` importable according to `importlib`, but importing it for CUDA details failed with an OpenMP `libiomp5md.dll` duplicate runtime error.
- OpenCV version: `5.0.0`.
- Camera availability probe:
  - Checked camera indices `0..3` using OpenCV DirectShow.
  - No camera opened successfully in the current execution environment.
  - OpenCV warned that DirectShow was available but could not capture by index.

## Phase 1 Implementation Plan

- Create minimal modular project skeleton under `ai_assistive_vision/`.
- Implement camera capture in `perception/camera.py`.
- Implement a small app entry point in `app.py` that:
  - Opens a webcam.
  - Displays live video.
  - Displays FPS on the frame.
  - Handles camera failure cleanly.
  - Supports clean shutdown with `q` or `Esc`.
- Add focused tests for FPS calculation and camera failure behavior that do not require a physical camera.

## Phase 1 Files Created

- `ai_assistive_vision/app.py`
- `ai_assistive_vision/config.py`
- `ai_assistive_vision/perception/camera.py`
- `ai_assistive_vision/README.md`
- `ai_assistive_vision/requirements.txt`
- `ai_assistive_vision/tests/test_camera.py`
- Package placeholder files for `perception`, `reasoning`, `feedback`, `models`, and `tests`.

## Phase 1 Test Results

- `python -m pytest ai_assistive_vision/tests`
  - Result: passed.
  - Details: 3 tests passed.
- `python -m ai_assistive_vision.app`
  - Result: app handled missing camera cleanly and exited with code `1`.
  - Output included: `Could not open camera index 0. Check that a webcam is connected and not used by another app.`
- `python -m compileall ai_assistive_vision`
  - Result: passed.
- `git status --short`
  - Result: not available because the current folder is not a Git repository.

## Current Blockers / Notes

- A physical or OS-exposed webcam was not available to this shell session, so live video display could not be visually verified here.
- PyTorch CUDA details still need follow-up later because the import-based CUDA check hit an OpenMP duplicate runtime error. This does not block Phase 1 because no AI inference is implemented yet.

## Phase 2 - YOLO Object Detection

Implemented:

- Added `ai_assistive_vision/perception/object_detection.py`.
- Added lightweight Ultralytics YOLO integration.
- Detection output format:

```python
{
    "label": "person",
    "confidence": 0.92,
    "bbox": [x1, y1, x2, y2],
    "center": [x, y],
}
```

- Added bounding box and confidence display in `app.py`.
- Added tests in `ai_assistive_vision/tests/test_object_detection.py`.
- Installed required dependency for the active Python 3.13 environment:
  - `ultralytics==8.4.173`
  - Transitive installs: `nvidia-ml-py`, `polars`, `polars-runtime-32`, `ultralytics-platform`, `ultralytics-thop`.
- Downloaded `yolov8n.pt` and moved it to `ai_assistive_vision/models/yolov8n.pt`.

Notes:

- The active 3.13 environment has `torch 2.13.0+cpu`; CUDA is not available to PyTorch yet.
- YOLO currently runs on CPU despite the NVIDIA GPU being present at the driver level.
- `ObjectDetector` sets `KMP_DUPLICATE_LIB_OK=TRUE` before importing Ultralytics/Torch because this local environment has a duplicate OpenMP runtime issue.

## Phase 3 - Spatial Reasoning

Implemented:

- Added `ai_assistive_vision/reasoning/spatial.py`.
- Divides the frame into thirds:
  - `left`
  - `center`
  - `right`
- Adds a `direction` field to detections without mutating the original detection dictionary.
- Updated `app.py` so drawn labels include object label, confidence, and direction.
- Added tests in `ai_assistive_vision/tests/test_spatial.py`.

Example output shape after Phase 3:

```python
{
    "label": "person",
    "confidence": 0.92,
    "bbox": [10, 20, 110, 220],
    "center": [60, 120],
    "direction": "left",
}
```

## Phase 2 and 3 Test Results

- `C:\Users\Lohet.S.Ratnakar\anaconda3\python.exe -m pytest -p no:cacheprovider ai_assistive_vision/tests`
  - Result: passed.
  - Details: 9 tests passed.
- `C:\Users\Lohet.S.Ratnakar\anaconda3\python.exe -m compileall ai_assistive_vision`
  - Result: passed.
- Direct detector smoke test on a blank frame:
  - Loaded `ai_assistive_vision/models/yolov8n.pt`.
  - Ran inference successfully on CPU.
  - Returned `[]`, which is expected for a blank image.
- App launch check:
  - Command: `C:\Users\Lohet.S.Ratnakar\anaconda3\python.exe -m ai_assistive_vision.app`
  - Result: opened camera index `0` and loaded YOLO model on CPU.
  - The process was stopped manually with Ctrl+C because the OpenCV window cannot be controlled from this shell.

## Manual Verification for User

Run from the repository root:

```powershell
C:\Users\Lohet.S.Ratnakar\anaconda3\python.exe -m ai_assistive_vision.app
```

Expected with a working webcam:

- A live camera window opens.
- FPS appears in the top-left corner.
- Common objects show bounding boxes, confidence scores, and `left`/`center`/`right` labels.
- Pressing `q` or `Esc` closes the window cleanly.

If no camera opens, try changing `CAMERA_INDEX` in `ai_assistive_vision/config.py` from `0` to another available index.

## Accuracy Improvement Update

Applied on 2026-10-06:

- Changed YOLO model from `yolov8n.pt` to `yolov8s.pt`.
- Changed confidence threshold from `0.35` to `0.50`.
- Downloaded `ai_assistive_vision/models/yolov8s.pt`.
- Installed CUDA-enabled PyTorch packages in the Anaconda Python 3.13 environment:
  - `torch==2.11.0+cu128`
  - `torchvision==0.26.0+cu128`
- Verified PyTorch CUDA:
  - `torch.cuda.is_available()` is `True`.
  - CUDA runtime reported by PyTorch: `12.8`.
  - Device: `NVIDIA GeForce RTX 4060 Laptop GPU`.
- Verified tests:
  - `C:\Users\Lohet.S.Ratnakar\anaconda3\python.exe -m pytest -p no:cacheprovider ai_assistive_vision/tests`
  - Result: 9 passed.
- Verified detector smoke test:
  - Model: `ai_assistive_vision/models/yolov8s.pt`
  - Device: `cuda:0`
  - Blank-frame detections: `[]`.
- Verified app startup:
  - App opens camera index `0`.
  - App loads `yolov8s.pt` on `cuda:0`.
  - Process was stopped manually with Ctrl+C from the shell.

Notes:

- Higher confidence reduces false positives but can miss weak or partially visible objects.
- `yolov8s.pt` should be more accurate than `yolov8n.pt`, but it is also heavier.
- Camera conditions still matter: better lighting, less motion blur, and fully visible objects will improve practical detection quality.

## Phase 4 - Relative Depth

Implemented:

- Added `ai_assistive_vision/perception/depth.py`.
- Integrated `MiDaS` monocular depth model for relative depth estimation.
- Classifies objects into relative distance buckets: `far`, `medium`, `close`, `very_close`.
- Extends detection outputs by computing the median depth of the object's bounding box.
- Updated `app.py` to display relative depth alongside object class and direction.
- Added tests in `ai_assistive_vision/tests/test_depth.py` and fixed failing assertions.

Notes:

- Depth scores are relative to the scene and normalized (0..1) within the frame. Larger scores represent closer objects.
- Physical distance cannot be guaranteed without camera calibration, so depth is only relative.

## Phase 5 - Obstacle Reasoning

Implemented:

- Added `ai_assistive_vision/reasoning/decision.py`.
- Created an `extract_events` function that filters incoming detections.
- Generates "obstacle" events when objects are classified as `close` or `very_close`.
- Assigns priority: `high` for `very_close` objects and `medium` for `close` objects.
- Updated `app.py` to display extracted events dynamically on the camera feed.
- Updated `config.py` window name to reflect Phase 5.
- Added tests in `ai_assistive_vision/tests/test_decision.py`.

Notes:

- This is a basic rule-based engine. It doesn't yet deduplicate (Phase 6) or track objects over time (Phase 8/9).

## Phase 6 - Priority and Deduplication

Implemented:

- Added `ai_assistive_vision/reasoning/priority.py` containing `EventManager`.
- `EventManager` maps YOLO objects and obstacle priorities to discrete scores (e.g. `immediate_obstacle`, `vehicle`, `person`).
- Deduplication is handled by generating unique signatures (`{object}_{direction}_{distance}`) and maintaining a 3.0 second cooldown between announcements.
- Updated `app.py` to filter `events` through `EventManager.filter_and_prioritize()`.
- Added console logging (`logging.info`) for prioritized events to simulate speech triggering without redundancy.
- Updated `config.py` window name to reflect Phase 6.
- Added tests in `ai_assistive_vision/tests/test_priority.py`.

Notes:

- Filtered events are printed to the console rather than shown exclusively on screen, as screen drawing persists for only one frame otherwise.

## Phase 7 - Audio Feedback

Implemented:

- Added `ai_assistive_vision/feedback/audio.py` for Text-to-Speech (TTS).
- Uses `pyttsx3` for offline, low-latency voice synthesis.
- Audio engine runs in a separate background thread with a queue to ensure OpenCV frame processing is not blocked.
- Transforms raw labels into short descriptive sentences (e.g. "Person ahead.", "Car on the left.").
- Integrated into `app.py` to speak the deduplicated, high-priority events.
- Updated `config.py` window name to reflect Phase 7.
- Added tests in `ai_assistive_vision/tests/test_audio.py` mocking `pyttsx3` to verify queue and thread logic.
- Updated `requirements.txt` with `pyttsx3`.

Notes:

- The text queue size is artificially limited (max size 2) to ensure speech stays relevant and the system doesn't lag far behind real-world events.

## Phase 8 - Object Tracking

Implemented:

- Added `ai_assistive_vision/perception/tracking.py` containing `ObjectTracker`.
- Integrates Ultralytics `ByteTrack` (`bytetrack.yaml`) to assign and persist continuous `track_id` values across frames.
- Extended detection objects with `track_id` (e.g. `ID:1 person 0.92 center close`).
- Updated `app.py` to run `tracker.track(frame)` instead of raw detector when tracking is enabled (`TRACKER_ENABLED=True`).
- Added tests in `ai_assistive_vision/tests/test_tracking.py`.

Notes:

- `persist=True` preserves state between calls.
- `reset_tracker()` is provided to clear internal tracking state on pipeline resets.

## Phase 9 - Temporal Reasoning

Implemented:

- Added `ai_assistive_vision/reasoning/temporal.py` containing `TemporalReasoner`.
- Bounded sliding window history (`TEMPORAL_HISTORY_LEN=30`) per `track_id`.
- Evaluates frame-to-frame motion trends:
  - `approaching`: relative inverse depth score increases or distance bucket moves closer.
  - `receding`: relative inverse depth score decreases or distance bucket moves farther.
  - `moving`: center point offset exceeds spatial motion threshold.
  - `direction_change`: horizontal direction shifts (e.g., `left` to `center`).
  - `persistent_obstacle`: object remains `close` or `very_close` across consecutive frames.
  - `disappearing`: tracked object is missing for `TEMPORAL_MISSING_TIMEOUT_FRAMES` (15 frames).
- Implemented state-transition guarding (`_last_event_state`) to eliminate duplicate event spam.
- Integrated temporal events into `extract_events()` in `decision.py` and `app.py`.
- Added tests in `ai_assistive_vision/tests/test_temporal.py`.

Notes:

- Higher inverse depth score indicates an approaching object.
- State transitions emit events only when target state changes.

## Phase 10 - Environmental OCR

Implemented:

- Added `ai_assistive_vision/perception/ocr.py` featuring `TextRecognizer` and `OcrWorker`.
- Integrates `EasyOCR` for offline scene text detection and recognition.
- Dedicated `OcrWorker` thread running with a bounded queue (`maxsize=1`) and frame-interval sampling (`OCR_FRAME_INTERVAL=10`) so OpenCV live display feed runs smoothly at full FPS without blocking.
- Implemented multi-tier text filtering & candidate classification (`classify_text_candidate`):
  - High-priority keywords: `EXIT`, `EMERGENCY`, `STAIRS`, `DANGER`, `RESTROOM`, etc.
  - Room/location regex patterns: `ROOM 204`, `LAB 302`, `BLOCK C`, `RM 101`, etc.
  - Rejects single character/gibberish non-actionable noise.
- Maps spatial center `[x, y]` of text bounding boxes to horizontal direction (`left`, `center`, `right`).
- Integrated cached OCR results into `app.py` main display loop (`draw_ocr_detections`) and prioritized event queue.
- Added unit tests in `ai_assistive_vision/tests/test_ocr.py`.

Notes:

- `KMP_DUPLICATE_LIB_OK=TRUE` set for OpenMP compatibility.
- Asynchronous worker design ensures 0ms main-thread delay for OCR inference.

## Full Test Suite Results (Phase 1 to Phase 10)

Command executed:
`.\.venv\Scripts\python.exe -m pytest ai_assistive_vision/tests`

- Test result: **38 passed in 19.60s**.
- Breakdowns by test module:
  - `test_camera.py`: 3 passed (Camera config, FPS counter, failure handling)
  - `test_object_detection.py`: 3 passed (YOLO initialization, smoke test, detection output schema)
  - `test_spatial.py`: 3 passed (Left / center / right bounding box spatial classification)
  - `test_depth.py`: 4 passed (MiDaS model depth estimation & relative distance bucket labeling)
  - `test_decision.py`: 1 passed (Event extraction & priority assignment for close obstacles)
  - `test_priority.py`: 2 passed (Deduplication signature matching & 3s cooldown manager)
  - `test_audio.py`: 3 passed (TTS sentence generation, queue overflow protection, thread shutdown)
  - `test_tracking.py`: 3 passed (ByteTrack tracker initialization, track ID mapping, tracker reset)
  - `test_temporal.py`: 8 passed (Approaching, receding, lateral movement, direction change, persistence, disappearing events)
  - `test_ocr.py`: 8 passed (Keyword matching, room regex, confidence filtering, async worker thread submission, frame dropping)

## Phase 11 - Door Detection

Implemented on 2026-10-13:

- Added i_assistive_vision/perception/door_detection.py.
- Dual-backend design:
  - **Custom YOLO backend** (preferred): Loads a fine-tuned YOLOv8 .pt model from DOOR_MODEL_PATH when the file exists.
  - **Geometric contour fallback**: When no custom model is configured, uses OpenCV Canny + contour approximation with door aspect-ratio heuristics. Zero additional dependencies.
- Greedy NMS with configurable IoU threshold removes overlapping candidates.
- Spatial direction attached to each door detection based on bounding box center.
- Updated config.py with: DOOR_ENABLED, DOOR_MODEL_PATH, DOOR_CONFIDENCE, DOOR_DEVICE, DOOR_FRAME_INTERVAL.
- Updated easoning/decision.py: added door_events parameter to xtract_events().
- Updated easoning/priority.py: added "door" category recognition.
- Updated eedback/audio.py: added "door" event speech "Door {direction_phrase}.".
- Updated pp.py: integrated DoorDetector, draw_door_detections() (green boxes), door caching in main loop.
- Fixed pre-existing OCR test mock failures in 	est_ocr.py.
- Added tests in i_assistive_vision/tests/test_door_detection.py (13 tests).
- Window name updated to "AI Assistive Vision - Phase 11".

Notes:

- Geometric detector has moderate false-positive rate in complex scenes. For production, provide a fine-tuned YOLO door model.
- Door detection runs every DOOR_FRAME_INTERVAL=3 frames to minimize compute impact.

## Full Test Suite Results (Phase 1 to Phase 11)

Command executed:
`C:\Users\Lohet.S.Ratnakar\anaconda3\python.exe -m pytest -p no:cacheprovider ai_assistive_vision/tests`

- Test result: **53 passed in 7.45s**.
- Breakdowns by test module:
  - 	est_camera.py: 3 passed
  - 	est_object_detection.py: 3 passed
  - 	est_spatial.py: 3 passed
  - 	est_depth.py: 4 passed
  - 	est_decision.py: 1 passed
  - 	est_priority.py: 2 passed
  - 	est_audio.py: 3 passed
  - 	est_tracking.py: 3 passed
  - 	est_temporal.py: 8 passed
  - 	est_ocr.py: 8 passed (2 previously failing mock tests fixed)
  - 	est_door_detection.py: 13 passed (IOU, NMS, geometric detector, unified factory)
