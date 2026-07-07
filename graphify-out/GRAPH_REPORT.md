# Graph Report - proj  (2026-07-07)

## Corpus Check
- 68 files · ~20,614 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 436 nodes · 1071 edges · 68 communities (16 shown, 52 thin omitted)
- Extraction: 87% EXTRACTED · 13% INFERRED · 0% AMBIGUOUS · INFERRED: 138 edges (avg confidence: 0.5)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `651c96df`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- [[_COMMUNITY_Core Pipeline & Camera|Core Pipeline & Camera]]
- [[_COMMUNITY_Framework & Context Core|Framework & Context Core]]
- [[_COMMUNITY_Motor & Gaze Detection|Motor & Gaze Detection]]
- [[_COMMUNITY_Detection Catalogue & Categories|Detection Catalogue & Categories]]
- [[_COMMUNITY_Longitudinal Behavior Storage|Longitudinal Behavior Storage]]
- [[_COMMUNITY_Context & Safety Modules|Context & Safety Modules]]
- [[_COMMUNITY_Skin Color Analysis|Skin Color Analysis]]
- [[_COMMUNITY_Landmark Pixel Helpers|Landmark Pixel Helpers]]
- [[_COMMUNITY_Emotion Recognition|Emotion Recognition]]
- [[_COMMUNITY_Drowsiness Concepts (EARPERCLOS)|Drowsiness Concepts (EAR/PERCLOS)]]
- [[_COMMUNITY_Pain Expression Module|Pain Expression Module]]
- [[_COMMUNITY_Age Estimation Module|Age Estimation Module]]
- [[_COMMUNITY_Sweating Module|Sweating Module]]
- [[_COMMUNITY_Detection Overlay|Detection Overlay]]
- [[_COMMUNITY_CLAUDE|CLAUDE.md]]
- [[_COMMUNITY___init__.py|__init__.py]]
- [[_COMMUNITY_age_estimation module|age_estimation module]]
- [[_COMMUNITY_emotion module|emotion module]]
- [[_COMMUNITY_facial_asymmetry module|facial_asymmetry module]]
- [[_COMMUNITY_fall module|fall module]]
- [[_COMMUNITY_gait module|gait module]]
- [[_COMMUNITY_heart_rate module|heart_rate module]]
- [[_COMMUNITY_presence module|presence module]]
- [[_COMMUNITY_respiration module|respiration module]]
- [[_COMMUNITY_skin_color module|skin_color module]]
- [[_COMMUNITY_tremor module|tremor module]]
- [[_COMMUNITY_Behavioral & Routine Patterns Category|Behavioral & Routine Patterns Category]]
- [[_COMMUNITY_Demographic  Contextual Estimation Category|Demographic / Contextual Estimation Category]]
- [[_COMMUNITY_Emotional & Cognitive State Category|Emotional & Cognitive State Category]]
- [[_COMMUNITY_Fatigue, Drowsiness & Consciousness Category|Fatigue, Drowsiness & Consciousness Category]]
- [[_COMMUNITY_Neurological  Motor Function Category|Neurological / Motor Function Category]]
- [[_COMMUNITY_Falls & Safety Events Category|Falls & Safety Events Category]]
- [[_COMMUNITY_Facial & Skin Analysis Category|Facial & Skin Analysis Category]]
- [[_COMMUNITY_Vital Signs Category|Vital Signs Category]]
- [[_COMMUNITY_Eye-Aspect-Ratio (EAR)|Eye-Aspect-Ratio (EAR)]]
- [[_COMMUNITY_MediaPipe FaceLandmarker (478 landmarks + iris)|MediaPipe FaceLandmarker (478 landmarks + iris)]]
- [[_COMMUNITY_Stroke FAST Protocol (facial droop)|Stroke FAST Protocol (facial droop)]]
- [[_COMMUNITY_FFT Dominant-Frequency Analysis|FFT Dominant-Frequency Analysis]]
- [[_COMMUNITY_Motion Extractor (frame-difference energy)|Motion Extractor (frame-difference energy)]]
- [[_COMMUNITY_PERCLOS|PERCLOS]]
- [[_COMMUNITY_Personal Baseline Calibration|Personal Baseline Calibration]]
- [[_COMMUNITY_MediaPipe PoseLandmarker (33 body landmarks)|MediaPipe PoseLandmarker (33 body landmarks)]]
- [[_COMMUNITY_rPPG Remote Photoplethysmography|rPPG Remote Photoplethysmography]]
- [[_COMMUNITY_Screening-Not-Diagnosis Safety Principle|Screening-Not-Diagnosis Safety Principle]]
- [[_COMMUNITY_Aggregator (latest signal per module,key)|Aggregator (latest signal per module,key)]]
- [[_COMMUNITY_Cadence-Aware Scheduler|Cadence-Aware Scheduler]]
- [[_COMMUNITY_Config-Driven Modules (modules.yaml)|Config-Driven Modules (modules.yaml)]]
- [[_COMMUNITY_DetectionModule base class|DetectionModule base class]]
- [[_COMMUNITY_FrameContext (per-frame feature cache)|FrameContext (per-frame feature cache)]]
- [[_COMMUNITY_Rule-Based Greeting Engine|Rule-Based Greeting Engine]]
- [[_COMMUNITY_Humanoid Camera Elderly-Care Detection Pipeline|Humanoid Camera Elderly-Care Detection Pipeline]]
- [[_COMMUNITY_Modular Pipeline Architecture|Modular Pipeline Architecture]]
- [[_COMMUNITY_Module Registry (@register self-registration)|Module Registry (@register self-registration)]]
- [[_COMMUNITY_Result Schema|Result Schema]]
- [[_COMMUNITY_Shared Extractors (once-per-frame)|Shared Extractors (once-per-frame)]]
- [[_COMMUNITY_SQLite Longitudinal History Store|SQLite Longitudinal History Store]]
- [[_COMMUNITY_DeepFace|DeepFace]]
- [[_COMMUNITY_Four Core Technology Pillars|Four Core Technology Pillars]]
- [[_COMMUNITY_Google MediaPipe (Face Mesh + Pose)|Google MediaPipe (Face Mesh + Pose)]]
- [[_COMMUNITY_Modular Pipeline Recommendation (don't chain 15 repos)|Modular Pipeline Recommendation (don't chain 15 repos)]]
- [[_COMMUNITY_Open-rPPG toolbox|Open-rPPG toolbox]]
- [[_COMMUNITY_OpenFace (Facial Action Units)|OpenFace (Facial Action Units)]]
- [[_COMMUNITY_Ultralytics YOLO (YOLOv8YOLOv11)|Ultralytics YOLO (YOLOv8/YOLOv11)]]

## God Nodes (most connected - your core abstractions)
1. `FrameContext` - 133 edges
2. `Severity` - 73 edges
3. `DetectionModule` - 69 edges
4. `TimedBuffer` - 53 edges
5. `register()` - 32 edges
6. `Result` - 23 edges
7. `bandpass()` - 16 edges
8. `Scheduler` - 15 edges
9. `dominant_frequency()` - 15 edges
10. `Aggregator` - 14 edges

## Surprising Connections (you probably didn't know these)
- `FaceExtractor` --uses--> `FrameContext`  [INFERRED]
  extractors/face.py → core/context.py
- `MotionExtractor` --uses--> `FrameContext`  [INFERRED]
  extractors/motion.py → core/context.py
- `PoseExtractor` --uses--> `FrameContext`  [INFERRED]
  extractors/pose.py → core/context.py
- `ActivityLevel` --uses--> `FrameContext`  [INFERRED]
  modules/activity_level.py → core/context.py
- `Agitation` --uses--> `FrameContext`  [INFERRED]
  modules/agitation.py → core/context.py

## Import Cycles
- None detected.

## Communities (68 total, 52 thin omitted)

### Community 0 - "Core Pipeline & Camera"
Cohesion: 0.06
Nodes (36): Any, FaceData, PoseData, A single detection outcome.      module:     registered module name (e.g. "heart, Result, all_registered(), build_enabled(), discover() (+28 more)

### Community 1 - "Framework & Context Core"
Cohesion: 0.10
Nodes (41): Per-frame shared state passed to every module.  Expensive extraction (face mesh,, Unified result schema emitted by every detection module., Severity, Module registry: modules self-register via decorator; the pipeline instantiates, Class decorator: @register("fall_detection")., register(), Enum, MediaPipe FaceMesh landmark indices shared across modules.  Index reference: htt (+33 more)

### Community 2 - "Motor & Gaze Detection"
Cohesion: 0.06
Nodes (17): Agitation, Balance, BodyEstimate, Bradykinesia, Drowsiness, _ear(), Fall, Gait (+9 more)

### Community 4 - "Longitudinal Behavior Storage"
Cohesion: 0.16
Nodes (4): ActivityLevel, Presence, HistoryStore, Persistent longitudinal store (SQLite).  Longitudinal modules (activity trends,

### Community 5 - "Context & Safety Modules"
Cohesion: 0.06
Nodes (18): Camera, Frame source abstraction: webcam index, video file, or RTSP URL., FrameContext, ndarray, Face landmarks in pixel coordinates, shape (478, 2)., Pose landmarks in pixel coordinates, shape (33, 2)., Pipeline, Pipeline: capture -> shared extractors -> scheduled modules -> aggregator. (+10 more)

### Community 6 - "Skin Color Analysis"
Cohesion: 0.10
Nodes (13): Bruise, DryLips, EyeRedness, Rash, _Baseline, _norm_chroma(), ndarray, Mean normalized (r,g,b) chromaticity of a set of BGR pixels. (+5 more)

### Community 7 - "Landmark Pixel Helpers"
Cohesion: 0.09
Nodes (21): Behavioral & Routine Patterns (longitudinal, needs history), Demographic / Contextual Estimation, Detection List, Emotional & Cognitive State, Facial & Skin Analysis, Falls & Safety Events, Fatigue, Drowsiness & Consciousness, Neurological / Motor Function (+13 more)

### Community 13 - "Pain Expression Module"
Cohesion: 0.18
Nodes (10): 1. Core Ecosystems (The Frameworks), 2. Specialized Repositories mapped to your Categories, 🎭 Emotional, Pain & Facial Analysis, 🚨 Falls & Safety Events, 🥱 Fatigue, Drowsiness & PERCLOS, Google MediaPipe, Quick Reference: Architecture Mapping, Technical Recommendation for Implementation (+2 more)

### Community 14 - "Age Estimation Module"
Cohesion: 0.47
Nodes (5): _parse_hr(), Standalone data window: renders all detections as readable text on a dark panel,, Collect heart_rate backend readings keyed by backend label.     Returns {label:, render(), _text()

### Community 22 - "Sweating Module"
Cohesion: 0.07
Nodes (24): ABC, EyeMovement, HeartRate, Remote photoplethysmography (rPPG) heart rate + HRV.  Runs one or more pluggable, Common interface for rPPG heart-rate backends.  A backend is fed one frame at a, Return a reading dict or None if not ready., RPPGBackend, ClassicalBackend (+16 more)

### Community 25 - "Detection Overlay"
Cohesion: 0.40
Nodes (3): draw_boxes(), Draw detection results onto the frame for a live debug view., Lightweight camera overlay: face/pose boxes + fps only. All textual     data liv

## Knowledge Gaps
- **68 isolated node(s):** `graphify`, `Heart rate: two backends, compared live`, `Architecture`, `Add or change a module`, `Test without a webcam` (+63 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **52 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `FrameContext` connect `Context & Safety Modules` to `Core Pipeline & Camera`, `Framework & Context Core`, `Motor & Gaze Detection`, `Longitudinal Behavior Storage`, `Skin Color Analysis`, `Emotion Recognition`, `Sweating Module`?**
  _High betweenness centrality (0.299) - this node is a cross-community bridge._
- **Why does `Severity` connect `Framework & Context Core` to `Core Pipeline & Camera`, `Motor & Gaze Detection`, `Longitudinal Behavior Storage`, `Context & Safety Modules`, `Skin Color Analysis`, `Emotion Recognition`, `Age Estimation Module`, `Sweating Module`, `Detection Overlay`?**
  _High betweenness centrality (0.082) - this node is a cross-community bridge._
- **Why does `TimedBuffer` connect `Motor & Gaze Detection` to `Framework & Context Core`, `Sweating Module`?**
  _High betweenness centrality (0.048) - this node is a cross-community bridge._
- **Are the 41 inferred relationships involving `FrameContext` (e.g. with `Camera` and `Pipeline`) actually correct?**
  _`FrameContext` has 41 INFERRED edges - model-reasoned connections that need verification._
- **Are the 34 inferred relationships involving `Severity` (e.g. with `ActivityLevel` and `AgeEstimation`) actually correct?**
  _`Severity` has 34 INFERRED edges - model-reasoned connections that need verification._
- **Are the 34 inferred relationships involving `DetectionModule` (e.g. with `ActivityLevel` and `AgeEstimation`) actually correct?**
  _`DetectionModule` has 34 INFERRED edges - model-reasoned connections that need verification._
- **Are the 15 inferred relationships involving `TimedBuffer` (e.g. with `Agitation` and `Balance`) actually correct?**
  _`TimedBuffer` has 15 INFERRED edges - model-reasoned connections that need verification._