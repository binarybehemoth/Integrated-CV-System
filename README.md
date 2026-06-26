# Integrated Computer-Vision System

A from-scratch, integrated computer-vision system that turns a single
camera frame into a structured, queryable **world model** and streams
the results live to a mobile browser over WebRTC.

This repository is the companion code for the book
*Building an Integrated Computer-Vision System*. It grows chapter by
chapter; this snapshot corresponds to the **complete book** (Parts I–X, Chapters 1–41, plus Appendices A–F): the perception engine and live web stack; the dense, recognition, structural, and 3D capabilities; the trainer with custom, continual, and active learning; the optimization, packaging, security, and scaling layers; and the frontier and capstone material: the perception engine, the world-model
data structures, the capability interface, the YOLOv8 detector, the
FastAPI inference server, live WebRTC streaming, the mobile frontend,
browser/server persistence, and instance segmentation.


## Live demo & studio features

- **index.html** is a capability tour: tap/click the video to cycle every
  capability in isolation (detection, segmentation, keypoints, tracking +
  trails, class hierarchy, parts, scene graph, depth, and a **WebGL 3D**
  reconstruction of geometric primitives/voxels via Three.js). A live **log
  panel** prints each capability's per-frame output as text. Boxes use the
  video's object-fit:cover transform so they bound objects **tightly**;
  **segmentation** fills each object's pixels a distinct colour; **tracking**
  shows **persistent IDs** + trails; the **3D** view is static and mirrors the
  scene layout. A **model menu** switches between default COCO and your
  trained model sets. On **landscape/PC** screens the controls sit beside the
  video (no scrolling).
- **FPS slider** on index.html lowers the transmitted camera frame rate to
  cut bandwidth (ideal for an ngrok tunnel on a phone). The engine also runs
  detection only every `detect_interval` frames and relies on tracking in
  between, minimizing model calls.
- **Capabilities & startup:** the server loads a light chain by default
  (detection, segmentation, pose, tracking, parts, scene graph, 3D
  primitives — all small YOLO weights). Monocular **depth is opt-in**
  because it downloads a ~470 MB DPT model: enable it with
  `CV_ENABLE_DEPTH=1`, pick a model with `CV_DEPTH_MODEL`, or list the
  exact capabilities in `CV_CAPS` (comma-separated). Without depth, the
  3D view still shows primitives but without near/far ordering.
- **All capabilities load by default** (depth, OCR, face included). Each is
  guarded, so any whose dependency/model is missing is skipped with a
  warning, and downloads happen in the background. Install the optional deps
  to use them: `pip install easyocr insightface` (depth uses
  transformers/timm, already in requirements). For a fast, no-download
  start use `CV_CAPS=light`; or list exactly what you want in `CV_CAPS`.
- **OCR** is also opt-in (it downloads EasyOCR models): enable with
  `CV_ENABLE_OCR=1`. Detected text appears as `text` objects and is shown
  on the video and in the capability log.
- **3D from parts:** when an object has detected parts (e.g. a person's
  head/torso/limbs from keypoints), the 3D view rebuilds it as a
  combination of part primitives (head sphere, torso box, limb cylinders).
- **Mobile testing:** `python scripts/tunnel.py` opens an HTTPS ngrok tunnel
  so the camera works on a phone (`pip install pyngrok`).
- **studio.html**: images fit the viewport (small images no longer tiny);
  classes are a **clickable expand/collapse hierarchy**; **zoom** (wheel) and
  **pan** (middle/Shift-drag) keep box coordinates correct at any zoom;
  **sub-parts and keypoints are nameable** (names show on index.html); each
  trained **model set saves under a folder you name** (models/<name>/).
- **Training** is wired end to end: the studio's *Start training* button
  builds a YOLO dataset from the annotations and trains in the background
  (polling shown live). If the class hierarchy groups leaves under parents,
  training runs **multi-phase** — a coarse parent model plus one small child
  model per parent group — which trains faster for many classes.
  The server now starts **immediately** with just the detector and loads
  the rest of the chain in a background thread, so a slow model download
  never blocks startup (detection works while the rest comes online).


## Capabilities (built across the book)

- Object **detection** (YOLOv8 + COCO) — *implemented*
- Instance **segmentation** — *implemented*
- **Keypoints** / pose — *implemented*
- Multi-object **tracking** — *implemented*
- **Properties** and **states** — *implemented*
- Reading text (**OCR**) — *implemented*
- **Biometrics**: face, fingerprint, gait — *implemented*
- **Class** and **part** hierarchy — *implemented*
- **Scene-graph** relations — *implemented*
- **Ground plane** and horizon — *implemented*
- **Geometric primitives** (box/cylinder/sphere) — *implemented*
- **Voxel reconstruction** (space carving) — *implemented*
- **Monocular depth** and 2D→3D — *implemented*
- **Ground-plane** and horizon estimation
- Geometric **primitives** and **voxel** reconstruction
- **Incremental / continual** learning via a browser **annotation studio**
- **Frontier**: foundation models, promptable/open-vocabulary vision,
  neural rendering, multimodal understanding

## Architecture

Three tiers:

- **engine** (`server/engine/`) — pure perception: a NumPy frame in,
  a `WorldModel` out. Knows nothing about the network.
- **server** (`server/`) — wraps the engine in FastAPI + WebRTC
  (added in Part II).
- **web** (`web/`) — a thin browser client: camera capture, canvas
  overlays, and the annotation studio (added in Parts II and VI).

The engine threads a single `WorldModel` through an ordered list of
`Capability` plug-ins. Adding a capability is appending one class.

## Quick start

```bash
python3.11 -m venv .venv
source .venv/bin/activate                       # Windows: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
# install the CUDA (or cpu) PyTorch wheel first — see requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# verify the environment
python scripts/verify_gpu.py
python scripts/verify_yolo.py

# run detection on an image (from the repo root)
python -m server.engine.detector path/to/photo.jpg

# run detection on a video
python scripts/detect_video.py path/to/input.mp4 out.mp4

# run the inference server (Chapter 6); add aiortc+av for streaming
pip install "fastapi[standard]" uvicorn opencv-python-headless aiortc av
uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

# test the HTTP API
curl -s -F "file=@photo.jpg" http://localhost:8000/detect | jq .
python scripts/test_detect.py photo.jpg

# then open the live mobile client (camera needs HTTPS or localhost):
#   http://localhost:8000/        on the same machine, or
#   tunnel the port to your phone over HTTPS (see Chapter 8)
```

## Repository layout

```
server/
  app.py         # FastAPI server: /detect, /snapshot, static mount (Ch 6, 9)
  webrtc.py      # aiortc signalling + server-side frame loop (Chapter 7)
  storage.py     # file-based snapshot store (Chapter 9)
  engine/        # perception: one module per capability
    world_model.py   # shared data structures (the spine)
    capability.py    # the plug-in interface
    engine.py        # threads a WorldModel through capabilities
    config.py        # tunable settings
    detector.py      # YOLOv8 detection (Chapter 5)
    segmenter.py     # instance segmentation (Chapter 10)
    pose.py          # keypoints / pose (Chapter 11)
    tracker.py       # multi-object tracking (Chapter 12)
    state.py         # properties + finite-state machines (Chapter 13)
    ocr.py           # text detection + recognition (Chapter 14)
    face.py          # facial recognition (Chapter 15)
    fingerprint.py   # minutiae extraction + matching (Chapter 16)
    gait.py          # posture / movement / gait (Chapter 17)
    hierarchy.py     # class taxonomy / inheritance (Chapter 18)
    parts.py         # part decomposition (Chapter 19)
    scene_graph.py   # relations between objects (Chapter 20)
    ground.py        # ground plane / depth proxy (Chapter 21)
    geometry3d.py    # geometric primitives (Chapter 22)
    voxel.py         # voxel grid + space carving (Chapter 23)
    depth.py         # monocular depth + back-projection (Chapter 24)
    draw.py          # debug visualiser
    geometry.py      # IoU, NMS helpers
  training/      # custom training + continual learning (Parts VII)
    jobs.py          # training-job submission from the studio (Ch 25)
    openimages.py    # Open Images subset download via FiftyOne (Ch 26)
    orchestrator.py  # real Ultralytics training run (Ch 27)
    continual.py     # rehearsal/replay dataset merge (Ch 28)
    active.py        # active-learning uncertainty ranking (Ch 29)
    dataset.py       # build a YOLO dataset from a studio payload
    hierarchical.py  # multi-phase training: parent + per-parent child models
  security.py      # API-key auth + rate limiting (Ch 34)
  engine/hier_detector.py  # multi-phase hierarchical inference
  store/         # files: snapshots, models, datasets, jobs
web/             # browser viewer + annotation studio (Parts II, VII)
  index.html         # live camera + canvas overlay (Chapter 8)
  studio.html        # annotation studio UI (Chapter 25)
  js/rtc.js          # WebRTC client + render loop (Chapter 8)
  js/overlay.js      # draws boxes, masks, skeletons, relations (Ch 8, 10–20)
  js/studio.js       # studio: boxes, sub-parts, keypoints, YOLO export (Ch 25)
  js/db.js           # IndexedDB wrapper + studio Store (Chapters 9, 25)
models/          # exported weights (.pt / .onnx / .engine)
scripts/         # helper CLIs:
  verify_gpu.py / verify_yolo.py / smoke_detect.* / test_detect.py
  detect_video.py      # batch-process a video file (Ch 5)
  export_model.py      # export to ONNX/TensorRT/FP16 (Ch 30)
  profile_pipeline.py  # per-capability timing (Ch 32)
  tunnel.py            # ngrok HTTPS tunnel for mobile testing
Dockerfile         # GPU container image for the server (Ch 33)
```

## License

Source code in this repository is released under the MIT License
(see the book's copyright page). **Note:** it depends on Ultralytics
YOLOv8, which is licensed under **AGPL-3.0**; if you run this as a
network service you must comply with that license. See Chapter 3 of
the book for a plain-language explanation.
