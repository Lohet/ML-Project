# AI Assistive Vision --- Antigravity Master Build Specification

## 1. Project Goal

Build a modular, real-time AI assistive system for people with cortical
blindness.

The system should act as a **"second pair of eyes"** by using a camera
to understand the user's surroundings and converting important visual
information into simple **audio and/or haptic feedback**.

The system should eventually understand:

-   Obstacles
-   People
-   Vehicles
-   Doors
-   Stairs
-   Objects
-   Text/signs
-   Direction
-   Relative distance
-   Approaching objects
-   Basic scene context

The system is a **research/demo prototype**, not a certified medical
device or guaranteed mobility aid.

------------------------------------------------------------------------

# 2. Core Architecture

``` text
Camera
   ↓
AI Perception
   ├── Object Detection
   ├── Depth Estimation
   ├── OCR
   └── Tracking
   ↓
Spatial Reasoning
   ↓
Temporal Reasoning
   ↓
Priority / Decision Engine
   ↓
Feedback
   ├── Audio
   └── Haptic
```

The key design principle is:

> **Do not announce everything the AI detects. Announce what is
> important and actionable for the user.**

Example:

``` text
Person → CENTER → CLOSE

"Person ahead."
```

``` text
Car → LEFT → APPROACHING

"Vehicle approaching from the left."
```

------------------------------------------------------------------------

# 3. Development Environment

The primary development machine has:

-   NVIDIA RTX 4060 8 GB VRAM
-   24 GB RAM
-   Windows/Linux development environment as available

Use GPU acceleration where appropriate.

Initial software stack:

-   Python 3.11 preferred
-   OpenCV
-   PyTorch
-   Ultralytics YOLO
-   NumPy
-   Lightweight monocular depth model
-   PaddleOCR or EasyOCR
-   Local/offline TTS where practical

Do not introduce unnecessary frameworks.

------------------------------------------------------------------------

# 4. Project Structure

Use a modular structure:

``` text
ai_assistive_vision/
├── app.py
├── config.py
├── requirements.txt
├── README.md
│
├── perception/
│   ├── camera.py
│   ├── object_detection.py
│   ├── depth.py
│   ├── ocr.py
│   └── tracking.py
│
├── reasoning/
│   ├── spatial.py
│   ├── temporal.py
│   ├── priority.py
│   └── decision.py
│
├── feedback/
│   ├── audio.py
│   └── haptic.py
│
├── models/
└── tests/
```

Keep AI model implementations separate from reasoning and feedback
logic.

Use configuration variables rather than hard-coded values.

Add logging, error handling, and useful tests.

------------------------------------------------------------------------

# 5. IMPORTANT --- How Antigravity Must Develop the Project

**Do NOT build the entire project in one step.**

Develop it phase by phase.

For every phase:

1.  Explain what will be implemented.
2.  Inspect the existing project before changing it.
3.  Create or modify only the necessary files.
4.  Install only required dependencies.
5.  Run the implementation.
6.  Test it.
7.  Fix errors before continuing.
8.  Explain how the user can manually verify it.
9.  Update README/documentation when appropriate.
10. STOP and wait for approval before moving to the next phase.

Never rewrite the entire project unnecessarily.

Preserve working functionality when adding a new feature.

When modifying an existing module, make the smallest safe change
possible.

------------------------------------------------------------------------

# 6. ChatGPT + Antigravity Development Strategy

The project should use **ChatGPT for architecture, reasoning, difficult
debugging and technical decisions**, while **Antigravity performs
implementation, file editing, execution and testing**.

Recommended workflow:

``` text
                    USER
                     │
                     ↓
              CHATGPT 6 LUNA
       Architecture / hard decisions
       ML choices / difficult debugging
                     │
                     ↓
                ANTIGRAVITY
       Code / file changes / execution
                     │
                     ↓
              RTX 4060 MACHINE
             Real-time testing
                     │
                     ↓
              Errors / Results
                     │
                     └────→ CHATGPT
```

## Recommended model usage

### 6 Luna

Use for high-value reasoning:

-   Overall architecture
-   Computer vision architecture
-   ML model selection
-   Difficult bugs
-   CUDA/GPU problems
-   Performance bottlenecks
-   Dataset strategy
-   Custom model decisions
-   Door/stair detection strategy
-   Hardware/edge-device decisions
-   Final integration
-   Safety-critical reasoning

### 5.6 Luna

Use for most routine development:

-   Implementing modules
-   Adding features
-   Writing tests
-   Refactoring small sections
-   Documentation
-   README updates
-   Explaining normal errors
-   Routine debugging
-   Integration of already-decided designs

### 5.5

Use for simple questions:

-   Python syntax
-   Small fixes
-   Basic explanations
-   Package installation questions
-   Simple code transformations

### General rule

Do not spend the strongest model on trivial tasks.

Save 6 Luna for decisions where reasoning quality matters.

------------------------------------------------------------------------

# 7. Important Coding Rule

Do not repeatedly regenerate the whole project.

Bad approach:

``` text
"Here is my 2000-line project. Rewrite everything to add OCR."
```

Preferred approach:

``` text
"Here is ocr.py and the current app.py.
Add OCR while preserving the existing object detection,
depth and priority interfaces.
Modify only the files that need changes."
```

This keeps the project stable and reduces unnecessary context and token
usage.

------------------------------------------------------------------------

# 8. Development Phases

Follow this order.

## Phase 1 --- Camera

Implement:

``` text
Camera
 ↓
OpenCV
 ↓
Live video
 ↓
FPS
```

Requirements:

-   Open webcam
-   Display live feed
-   Display FPS
-   Handle camera failure
-   Clean shutdown

Do not implement AI yet.

------------------------------------------------------------------------

## Phase 2 --- YOLO Object Detection

Add a lightweight pretrained YOLO model.

Detect common objects.

Each detection should provide:

``` python
{
    "label": "person",
    "confidence": 0.92,
    "bbox": [x1, y1, x2, y2],
    "center": [x, y]
}
```

Display bounding boxes and confidence during development.

Do not implement speech yet.

------------------------------------------------------------------------

## Phase 3 --- Spatial Reasoning

Divide the camera view into:

``` text
+----------------+----------------+----------------+
|                |                |                |
|     LEFT       |     CENTER     |     RIGHT      |
|                |                |                |
+----------------+----------------+----------------+
```

For each object determine:

-   LEFT
-   CENTER
-   RIGHT

Example:

``` text
Person → CENTER
Car → LEFT
Chair → RIGHT
```

------------------------------------------------------------------------

## Phase 4 --- Relative Depth

Integrate a suitable monocular depth model.

Evaluate lightweight options such as:

-   MiDaS
-   Depth Anything
-   Depth Anything V2

Do not claim exact physical distance unless properly calibrated.

Initially classify:

``` text
FAR
MEDIUM
CLOSE
VERY_CLOSE
```

Example:

``` text
Person → CENTER → VERY_CLOSE
Car → LEFT → FAR
Chair → RIGHT → MEDIUM
```

------------------------------------------------------------------------

## Phase 5 --- Obstacle Reasoning

Combine:

``` text
Object
+
Direction
+
Relative depth
```

Example:

``` python
{
    "event": "obstacle",
    "object": "person",
    "direction": "center",
    "distance": "very_close",
    "priority": "high"
}
```

Do not generate speech yet.

------------------------------------------------------------------------

## Phase 6 --- Priority and Deduplication

The system must not announce the same event continuously.

Initial priority:

1.  Immediate obstacle
2.  Vehicle
3.  Person
4.  Stairs
5.  Door
6.  Important environmental object
7.  Text
8.  General scene information

Example:

``` text
Person → CENTER → CLOSE
```

should not become:

``` text
"Person."
"Person."
"Person."
"Person."
```

Instead announce once and repeat only if something important changes.

Possible triggers:

-   Distance changes significantly
-   Direction changes
-   Object disappears and reappears
-   Higher-priority event occurs
-   Object becomes dangerous/urgent

------------------------------------------------------------------------

## Phase 7 --- Audio Feedback

Convert high-priority events into short speech.

Examples:

``` text
"Person ahead."

"Obstacle on the left."

"Vehicle approaching from the right."

"Door ahead."

"Stairs ahead."
```

Speech must be:

-   Short
-   Clear
-   Low latency
-   Non-repetitive

Prefer local/offline TTS when practical.

------------------------------------------------------------------------

## Phase 8 --- Object Tracking

Introduce tracking using an appropriate tracker such as:

-   ByteTrack
-   BoT-SORT
-   YOLO tracking

Track objects across frames.

Example:

``` text
Car:
LEFT → CENTER
FAR → CLOSE
```

This allows the system to infer:

``` text
"Vehicle approaching."
```

rather than merely:

``` text
"Car."
```

------------------------------------------------------------------------

## Phase 9 --- Temporal Reasoning

Maintain a short history of detections.

Use it to understand:

-   Approaching objects
-   Moving vehicles
-   Moving people
-   Persistent obstacles
-   Disappearing objects
-   Direction changes

Do not make important decisions from one frame alone where temporal
information can improve reliability.

------------------------------------------------------------------------

## Phase 10 --- OCR

Add environmental text recognition.

Pipeline:

``` text
Camera
 ↓
Text detection
 ↓
OCR
 ↓
Priority
 ↓
Speech
```

Example:

``` text
Camera sees:

EXIT

↓

"Exit."
```

Do not automatically read every piece of text.

Prioritize useful environmental information such as:

-   Exit signs
-   Room numbers
-   Direction signs
-   Important labels

------------------------------------------------------------------------

## Phase 11 --- Door Detection

Investigate whether the selected pretrained detector can reliably detect
doors.

If not:

-   Find an appropriate dataset
-   Fine-tune a lightweight detector
-   Use a separate specialized model if necessary

Possible outputs:

``` text
"Door ahead."

"Door on the left."

"Door on the right."
```

Do not force a generic object detector to perform a class it was not
trained for.

------------------------------------------------------------------------

## Phase 12 --- Stair Detection

Treat stairs as a specialized safety-related problem.

Investigate:

-   Custom object detection
-   Segmentation
-   Geometric computer vision
-   Fine-tuned YOLO
-   Appropriate stair datasets

Support:

``` text
STAIRS UP
STAIRS DOWN
```

Example:

``` text
"Possible stairs ahead."
```

when confidence is insufficient for a definitive statement.

Do not claim this feature is safe for real-world navigation until
properly validated.

------------------------------------------------------------------------

## Phase 13 --- Vehicle Awareness

Detect:

-   Cars
-   Buses
-   Trucks
-   Motorcycles
-   Bicycles

Combine:

``` text
Object
+
Direction
+
Relative distance
+
Movement
```

Example:

``` text
"Vehicle approaching from the left."
```

Use tracking/temporal reasoning rather than relying on one frame.

------------------------------------------------------------------------

## Phase 14 --- Scene Understanding

Build a lightweight scene-understanding layer.

Example:

``` text
Indoor environment.
Person ahead.
Door on the right.
Chair on the left.
```

or:

``` text
Outdoor environment.
Road ahead.
Vehicle approaching from left.
Person on right.
```

The system should answer:

> "What does the user need to know right now?"

not:

> "What can the AI detect?"

------------------------------------------------------------------------

## Phase 15 --- Haptic Feedback

After audio works reliably, add optional haptic feedback.

Possible prototype:

``` text
ESP32
+
3 vibration motors
```

Mapping:

``` text
LEFT   → left motor
CENTER → center motor
RIGHT  → right motor
```

Distance can influence intensity:

``` text
FAR        → weak
MEDIUM     → moderate
CLOSE      → strong
VERY_CLOSE → strong/rapid
```

Keep the communication protocol simple.

------------------------------------------------------------------------

## Phase 16 --- Audio + Haptic Fusion

Combine both modalities intelligently.

Example:

``` text
Car
LEFT
CLOSE
APPROACHING
```

Output:

``` text
Audio:
"Vehicle approaching from the left."

Haptic:
Left motor → strong vibration
```

Do not provide redundant feedback for every minor detection.

------------------------------------------------------------------------

## Phase 17 --- Real-Time Optimization

Measure:

-   FPS
-   Inference latency
-   End-to-end latency
-   CPU usage
-   GPU usage
-   VRAM
-   RAM
-   TTS latency

Only optimize after functionality is correct.

Possible optimizations:

-   FP16
-   TensorRT
-   Smaller models
-   Frame skipping
-   Asynchronous inference
-   Quantization
-   Reduced image resolution
-   GPU acceleration

------------------------------------------------------------------------

## Phase 18 --- Testing

Create tests for:

### Object Detection

-   Person
-   Car
-   Chair
-   Bottle
-   Bag
-   Bicycle

### Spatial

-   Left
-   Center
-   Right

### Distance

-   Far
-   Medium
-   Close
-   Very close

### OCR

-   Exit
-   Room numbers
-   Signs
-   Labels

### Dynamic Objects

-   Moving person
-   Moving car
-   Approaching object
-   Object leaving

### Environment

-   Indoor
-   Outdoor
-   Corridor
-   Road
-   Room
-   Staircase

------------------------------------------------------------------------

# 9. Performance Evaluation

Track:

### Detection

-   Precision
-   Recall
-   mAP

### System

-   FPS
-   Inference latency
-   End-to-end latency
-   False alert rate
-   Missed important events

### User Experience

Where possible evaluate:

-   Number of unnecessary announcements
-   Number of missed important events
-   Time needed to understand a warning
-   Cognitive load
-   User preferences

------------------------------------------------------------------------

# 10. Privacy

Prefer local processing.

Avoid unnecessary cloud transmission of camera frames.

Do not store camera footage unless there is a clear reason.

If cloud processing is introduced later:

-   Explain it clearly
-   Encrypt communication
-   Minimize storage
-   Define retention rules

Do not implement face recognition in the initial system.

------------------------------------------------------------------------

# 11. Safety

This is an assistive research/demo prototype.

Never claim:

``` text
"This system guarantees safe navigation."
```

Use language such as:

``` text
"This system provides experimental environmental assistance."
```

For uncertain detections:

Bad:

``` text
"Stairs ahead."
```

Better:

``` text
"Possible stairs ahead."
```

The system should communicate uncertainty rather than confidently
presenting unreliable detections.

------------------------------------------------------------------------

# 12. MVP Milestone

The first major goal is NOT the complete wearable system.

First make this pipeline work reliably:

``` text
Camera
   ↓
YOLO
   ↓
Direction
   ↓
Depth
   ↓
Obstacle reasoning
   ↓
Priority
   ↓
Speech
```

Example:

``` text
Camera sees:

Person
CENTER
CLOSE

↓

"Person ahead."
```

Another:

``` text
Car
LEFT
CLOSE
APPROACHING

↓

"Vehicle approaching from the left."
```

Only after this MVP is stable should development proceed to OCR, doors,
stairs, advanced tracking and haptics.

------------------------------------------------------------------------

# 13. Hardware / Wearable Phase

Only after the laptop prototype works, investigate deployment options:

-   Smartphone
-   NVIDIA Jetson
-   Raspberry Pi + accelerator
-   Custom edge AI device
-   Camera glasses
-   External wearable camera
-   Bluetooth earbuds
-   ESP32 haptic device

Compare:

-   Latency
-   Battery life
-   Weight
-   Cost
-   Compute
-   VRAM/RAM
-   Portability
-   Camera quality
-   Connectivity

Do not select final hardware before validating the software
architecture.

------------------------------------------------------------------------

# 14. Final System

The target architecture is:

``` text
                    CAMERA
                       |
                       ↓
                FRAME PROCESSING
                       |
          +------------+-------------+
          |            |             |
          ↓            ↓             ↓
       OBJECT        DEPTH          OCR
       DETECTION     ESTIMATION     ENGINE
          |            |             |
          +------------+-------------+
                       |
                       ↓
                   TRACKING
                       |
                       ↓
              SPATIAL REASONING
                       |
                       ↓
              TEMPORAL REASONING
                       |
                       ↓
                 PRIORITY ENGINE
                       |
                       ↓
                DECISION ENGINE
                       |
                +------+------+
                |             |
                ↓             ↓
              AUDIO         HAPTIC
```

------------------------------------------------------------------------

# 15. FIRST TASK --- START HERE

**Do not implement the entire project immediately.**

Start ONLY with **Phase 1 --- Camera Input**.

Perform:

1.  Check Python version.
2.  Check NVIDIA GPU.
3.  Check CUDA/PyTorch.
4.  Check OpenCV.
5.  Check camera availability.
6.  Create the project directory.
7.  Create the initial files.
8.  Implement camera capture.
9.  Display live video.
10. Display FPS.
11. Test the application.
12. Fix any errors.
13. Explain how I can manually verify it.
14. STOP.

Do not implement YOLO until Phase 1 is working.

------------------------------------------------------------------------

# 16. Antigravity Operating Rules

-   Never implement multiple future phases without approval.
-   Never rewrite working code unnecessarily.
-   Inspect existing files before editing.
-   Keep modules independent.
-   Keep interfaces stable.
-   Prefer minimal changes.
-   Test after every meaningful change.
-   Report errors honestly.
-   Do not hide failed tests.
-   Do not claim hardware features were tested when they were not
    physically available.
-   Do not claim model accuracy without measurement.
-   Do not use cloud APIs when local processing is sufficient.
-   Keep the system real-time.
-   Optimize only after correctness.
-   Preserve backwards compatibility with previous phases.

------------------------------------------------------------------------

# 17. Expected Final Demonstration

The final demo should be able to show scenarios such as:

### Scenario 1

``` text
Person ahead
↓
"Person ahead."
```

### Scenario 2

``` text
Obstacle on right
↓
"Obstacle on the right."
+
Right vibration
```

### Scenario 3

``` text
Vehicle approaching from left
↓
"Vehicle approaching from the left."
+
Left vibration
```

### Scenario 4

``` text
EXIT sign
↓
"Exit."
```

### Scenario 5

``` text
Possible stairs
↓
"Possible stairs ahead."
```

The system should remain concise and prioritize the most important
information.

------------------------------------------------------------------------

# 18. Success Definition

The project is successful when it demonstrates:

``` text
Visual Environment
        ↓
AI Perception
        ↓
Spatial Understanding
        ↓
Temporal Understanding
        ↓
Priority Reasoning
        ↓
Decision
        ↓
Simple Sensory Feedback
```

The core innovation is not simply object detection.

It is:

> **Converting complex visual information into a small number of useful,
> understandable auditory and tactile signals for a person with impaired
> visual perception.**

Start with **Phase 1 only**.
