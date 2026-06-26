"""FastAPI inference server over the perception Engine.

Loads one warmed Engine at startup (lifespan), exposes a small JSON
API, and keeps the event loop responsive by offloading the blocking
forward pass to a worker thread. See Chapter 6.
"""
from __future__ import annotations
import asyncio
import json
import os
import time
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from .engine.config import EngineConfig
from .engine.engine import Engine
from .engine.detector import DetectorCapability
from . import storage

# One process-wide home for the shared engine and its config.
STATE: dict = {"engine": None, "config": None}


def build_engine(cfg: EngineConfig, detector=None, only=None) -> Engine:
    """Assemble the capability chain. Each capability is loaded in order
    but guarded: a missing dependency or a load error disables just that
    capability with a warning, so the server always starts.

    The set is configurable so startup stays fast. By DEFAULT the heavy
    monocular-depth model (a ~470 MB DPT download) is left OUT -- enable
    it with CV_ENABLE_DEPTH=1, or list capabilities explicitly in CV_CAPS
    (comma-separated). Override the depth model with CV_DEPTH_MODEL.

    Pass ``detector`` to use a custom first stage (the multi-phase
    HierarchicalDetector) in place of the default DetectorCapability."""
    import logging
    import os
    log = logging.getLogger("uvicorn.error")

    # A small depth model by default so it downloads fast even on a
    # rate-limited (unauthenticated) Hugging Face connection. Override with
    # CV_DEPTH_MODEL=Intel/dpt-hybrid-midas (better, ~470 MB) if you prefer.
    depth_model = os.environ.get("CV_DEPTH_MODEL", "Intel/dpt-swinv2-tiny-256")
    # name -> (module, class, constructor args)
    registry = {
        "detector":    ("detector", "DetectorCapability", (cfg,)),
        "segmenter":   ("segmenter", "SegmenterCapability", (cfg,)),
        "pose":        ("pose", "PoseCapability", (cfg,)),
        "tracker":     ("tracker", "TrackerCapability", ()),
        "state":       ("state", "StateCapability", (cfg,)),
        "gait":        ("gait", "PostureGaitCapability", (cfg,)),
        "hierarchy":   ("hierarchy", "HierarchyCapability", ()),
        "parts":       ("parts", "PartsCapability", (cfg,)),
        "scene_graph": ("scene_graph", "SceneGraphCapability", (cfg,)),
        "ground":      ("ground", "GroundPlaneCapability", (cfg,)),
        "geometry3d":  ("geometry3d", "GeometryCapability", (cfg,)),
        "face":        ("face", "FaceCapability", (cfg,)),
        "depth":       ("depth", "DepthCapability", (cfg, depth_model)),
        "ocr":         ("ocr", "OcrCapability", (cfg,)),
    }
    # Canonical processing order (detector is added first, separately).
    order = ["detector", "segmenter", "pose", "tracker", "state", "gait",
             "hierarchy", "parts", "scene_graph", "ground", "geometry3d",
             "face", "depth", "ocr"]
    # The "light" set (CV_CAPS=light): everything that needs no large
    # download. By default ALL capabilities load; depth/ocr/face pull big
    # models (DPT / EasyOCR / InsightFace) but load in the background and
    # are skipped if their dependency is missing.
    default = ["segmenter", "pose", "tracker", "state", "gait", "hierarchy",
               "parts", "scene_graph", "ground", "geometry3d"]
    env = os.environ.get("CV_CAPS")
    if only is not None:
        names = list(only)                       # explicit override (fast start)
    elif env and env.strip().lower() == "light":
        names = list(default)                    # light set: no large downloads
    elif env and env.strip().lower() == "all":
        names = list(order)                      # everything (explicit)
    elif env:
        names = [n.strip() for n in env.split(",") if n.strip()]
    else:
        names = list(order)                      # all capabilities by default
    # Additive opt-ins (handy alongside CV_CAPS=light).
    for flag, cap in (("CV_ENABLE_DEPTH", "depth"),
                      ("CV_ENABLE_OCR", "ocr"),
                      ("CV_ENABLE_FACE", "face")):
        if os.environ.get(flag) and cap not in names:
            names.append(cap)

    # Always load in canonical order so dependencies (e.g. tracker before
    # gait, pose before parts) are respected regardless of CV_CAPS order.
    names = ([n for n in order if n in names]
             + [n for n in names if n not in order])

    caps = []

    def add_name(name):
        spec = registry.get(name)
        if not spec:
            log.warning("unknown capability '%s' (ignored)", name)
            return
        module, cls, args = spec
        try:
            mod = __import__("server.engine." + module, fromlist=[cls])
            cap = getattr(mod, cls)(*args)
            cap.setup()                          # load this model now
            caps.append(cap)
        except Exception as exc:                 # missing dep / load error
            log.warning("capability %s disabled: %s", cls, exc)

    # First stage: custom detector if given, else the default detector.
    if detector is not None:
        try:
            detector.setup()
            caps.append(detector)
        except Exception as exc:
            log.warning("custom detector failed: %s", exc)
    else:
        add_name("detector")

    for n in names:
        if n == "detector":
            continue                             # already added as stage 1
        add_name(n)

    if not caps:
        raise RuntimeError("no capabilities could be loaded")
    engine = Engine(caps)
    engine._ready = True                         # each cap already set up
    log.warning("engine ready with capabilities: %s",
                ", ".join(c.name for c in caps))
    return engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    import logging
    import threading
    log = logging.getLogger("uvicorn.error")
    cfg = EngineConfig()
    try:                           # fall back to CPU if no GPU
        import torch
        if not torch.cuda.is_available():
            cfg.device, cfg.half = "cpu", False
    except Exception:
        cfg.device, cfg.half = "cpu", False
    STATE["config"] = cfg

    # Bring the server up FAST with just the detector (a tiny download),
    # then assemble the rest of the chain in a background thread and swap
    # it in when ready. This way a slow/large model download (e.g. depth)
    # can never block startup -- the server stays responsive throughout.
    try:
        STATE["engine"] = build_engine(cfg, only=["detector"])
        STATE["model_set"] = "default"
    except Exception as exc:
        log.warning("detector load failed: %s", exc)
        STATE["engine"] = None

    def _load_full():
        try:
            STATE["engine"] = build_engine(cfg)
            log.warning("full capability chain is now active")
        except Exception as exc:
            log.warning("full chain load failed (detector still active): %s", exc)
    threading.Thread(target=_load_full, daemon=True).start()

    yield                          # server serves traffic here
    STATE["engine"] = None         # release on shutdown


app = FastAPI(title="Integrated CV System",
              version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],           # tighten in production (Ch 34)
    allow_methods=["*"],
    allow_headers=["*"],
)


def _decode(raw: bytes) -> np.ndarray:
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)      # BGR ndarray
    if img is None:
        raise HTTPException(400, "Not a decodable image")
    return img


@app.post("/detect")
async def detect(file: UploadFile = File(...)) -> JSONResponse:
    engine: Engine = STATE["engine"]
    if engine is None:
        raise HTTPException(503, "Engine not ready")

    raw = await file.read()                        # async I/O
    frame = _decode(raw)

    t0 = time.perf_counter()
    loop = asyncio.get_running_loop()
    # Push the blocking forward pass onto a worker thread so the
    # event loop stays free to accept other connections.
    world = await loop.run_in_executor(None, engine.process, frame)
    infer_ms = (time.perf_counter() - t0) * 1000.0

    payload = world.to_dict()                      # lossless dict
    payload["timing_ms"] = round(infer_ms, 1)
    payload["image"] = {"width": frame.shape[1],
                        "height": frame.shape[0]}
    return JSONResponse(payload)


# --- Open-vocabulary detection (YOLOE) ------------------------------------
# A single, lazily-loaded YOLOE wrapper shared across requests. It sits beside
# the main engine chain rather than inside it because it is prompt-driven: the
# live page sends a text prompt and YOLOE returns boxes/masks for exactly those
# named classes, with no training. See server/engine/open_vocab.py.
_OPEN_VOCAB = None


def _open_vocab():
    global _OPEN_VOCAB
    if _OPEN_VOCAB is None:
        from .engine.open_vocab import OpenVocabDetector
        _OPEN_VOCAB = OpenVocabDetector()
    return _OPEN_VOCAB


@app.post("/detect_open")
async def detect_open(file: UploadFile = File(...),
                      prompt: str = Form("")) -> JSONResponse:
    """Open-vocabulary detection with YOLOE. Detects the comma-separated
    classes in `prompt` (e.g. "zebra, traffic cone, skateboard") with no
    training. Powers the live page's 'YOLOE (open-vocab)' mode."""
    raw = await file.read()
    frame = _decode(raw)
    base = {"image": {"width": frame.shape[1], "height": frame.shape[0]}}
    if not prompt.strip():
        return JSONResponse({"objects": [], "graph": {"edges": []},
                             "note": "empty prompt", **base})
    t0 = time.perf_counter()
    loop = asyncio.get_running_loop()
    try:
        world = await loop.run_in_executor(
            None, lambda: _open_vocab().detect(frame, prompt))
    except Exception as exc:                        # model not installed, etc.
        raise HTTPException(503, f"YOLOE unavailable: {exc}")
    infer_ms = (time.perf_counter() - t0) * 1000.0
    payload = world.to_dict()
    payload["timing_ms"] = round(infer_ms, 1)
    payload["prompt"] = prompt
    payload.update(base)
    return JSONResponse(payload)


_CASCADE = None


def _cascade():
    global _CASCADE
    if _CASCADE is None:
        from .engine.cascade import CascadeDetector
        models_dir = os.environ.get("CV_MODELS_DIR", "models")
        _CASCADE = CascadeDetector(_open_vocab(), models_dir)
    return _CASCADE


@app.post("/detect_cascade")
async def detect_cascade(file: UploadFile = File(...),
                         prompt: str = Form("")) -> JSONResponse:
    """Two-phase cascade. Phase 1: YOLOE detects the fixed level-1 vocabulary
    (its built-in ~4585 classes, prompt-free; a prompt narrows it). Phase 2:
    every detection whose class has a trained level-2 model is cropped and
    refined into a custom leaf class with parts and keypoints. Trained level-2
    models register via models/<name>/level2.json and activate automatically."""
    raw = await file.read()
    frame = _decode(raw)
    base = {"image": {"width": frame.shape[1], "height": frame.shape[0]}}
    t0 = time.perf_counter()
    loop = asyncio.get_running_loop()
    try:
        world = await loop.run_in_executor(
            None, lambda: _cascade().detect(frame, prompt))
    except Exception as exc:
        raise HTTPException(503, f"cascade unavailable: {exc}")

    # Enrich the two-phase detections with the SAME non-detection capabilities
    # the flat /detect path uses: tracker -> state/properties -> gait -> face
    # identity -> depth -> 3D -> scene graph. The cascade owns detection,
    # segmentation, parts and keypoints, so those capabilities are skipped.
    eng = STATE.get("engine")
    enriched = []
    if eng is not None and getattr(eng, "_ready", False):
        skip = {"detector", "segmenter", "pose", "parts", "hierarchy"}

        def _enrich():
            for cap in eng.capabilities:
                if getattr(cap, "name", "") in skip:
                    continue
                try:
                    cap.apply(world, frame)
                    enriched.append(getattr(cap, "name", "?"))
                except Exception:
                    pass
        await loop.run_in_executor(None, _enrich)

    infer_ms = (time.perf_counter() - t0) * 1000.0
    payload = world.to_dict()
    payload["timing_ms"] = round(infer_ms, 1)
    payload["levels"] = 2
    payload["enriched"] = enriched
    payload["registered_parents"] = sorted(_cascade().registry().keys())
    payload.update(base)
    return JSONResponse(payload)


# ---- Named Persons: face-identity enrollment & gallery -------------------
_FACE = None


def _gallery_path() -> str:
    return os.environ.get("CV_FACE_GALLERY", "models/face_gallery.json")


def _gallery_names() -> list:
    """Read enrolled identity names straight from disk (no model load)."""
    path = _gallery_path()
    if not os.path.exists(path):
        return []
    try:
        with open(path) as fh:
            return sorted(json.load(fh).keys())
    except Exception:
        return []


def _face():
    global _FACE
    if _FACE is None:
        from .engine.face import FaceCapability
        fc = FaceCapability(gallery_path=_gallery_path())
        fc.setup()                                  # loads InsightFace + gallery
        _FACE = fc
    return _FACE


@app.post("/enroll_face")
async def enroll_face(name: str = Form(...),
                      file: UploadFile = File(...)) -> JSONResponse:
    """Enroll a named identity from a clear face photo into the persistent
    gallery (the 'Named Persons' category). Running inference picks it up live
    via the gallery file's mtime, so no restart is needed."""
    name = name.strip()
    if not name:
        raise HTTPException(400, "name is required")
    raw = await file.read()
    frame = _decode(raw)
    loop = asyncio.get_running_loop()
    try:
        ok = await loop.run_in_executor(
            None, lambda: _face().enroll(name, frame))
    except Exception as exc:
        raise HTTPException(503, f"face recognition unavailable: {exc}")
    if not ok:
        return JSONResponse(
            {"enrolled": False, "error": "no face found in image"},
            status_code=422)
    return JSONResponse({"enrolled": True, "name": name,
                         "identities": sorted(_face().gallery.keys())})


@app.get("/faces")
async def list_faces() -> JSONResponse:
    """List enrolled identities. Reads the gallery file directly so it does not
    load the recognition model just to enumerate names."""
    names = _gallery_names()
    return JSONResponse({"identities": names, "count": len(names)})


@app.post("/faces/delete")
async def delete_face(name: str = Form(...)) -> JSONResponse:
    """Remove an enrolled identity from the gallery file."""
    path = _gallery_path()
    removed = False
    if os.path.exists(path):
        try:
            with open(path) as fh:
                data = json.load(fh)
            if name in data:
                del data[name]
                with open(path, "w") as fh:
                    json.dump(data, fh)
                removed = True
        except Exception as exc:
            raise HTTPException(500, f"could not update gallery: {exc}")
    if _FACE is not None:
        _FACE.gallery.pop(name, None)
    return JSONResponse({"removed": removed, "name": name,
                         "identities": _gallery_names()})


# ---- Open Images: bootstrap level-2 training data -----------------------
@app.post("/fetch_open_images")
async def fetch_open_images(class_name: str = Form(...),
                            n: int = Form(10),
                            exclude: str = Form("[]")) -> JSONResponse:
    """Download up to `n` Open Images V7 images containing `class_name`, each
    with bounding boxes, to bootstrap level-2 annotation. Only that class is
    fetched (via FiftyOne, which downloads just the matching images) -- nothing
    from Open Images is bundled with the project. `exclude` is a JSON array of
    already-imported source ids to skip, so repeated calls return new images.
    See `license` in the response for attribution terms."""
    n = max(1, min(int(n), 50))
    try:
        skip = json.loads(exclude) if exclude else []
        if not isinstance(skip, list):
            skip = []
    except Exception:
        skip = []
    loop = asyncio.get_running_loop()
    try:
        from .training.open_images import fetch, OPEN_IMAGES_LICENSE
        images = await loop.run_in_executor(
            None, lambda: fetch(class_name, n, skip))
    except Exception as exc:
        raise HTTPException(503, str(exc))
    return JSONResponse({"class": class_name, "count": len(images),
                         "images": images, "license": OPEN_IMAGES_LICENSE})


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "ready": STATE["engine"] is not None}


@app.get("/info")
async def info() -> dict:
    cfg: EngineConfig = STATE["config"]
    engine: Engine = STATE["engine"]
    classes: list[str] = []
    if engine is not None and engine.capabilities:
        det = engine.capabilities[0]
        classes = sorted(getattr(det, "names", {}).values())
    return {
        "model": cfg.detect_weights if cfg else None,
        "device": cfg.device if cfg else None,
        "img_size": cfg.img_size if cfg else None,
        "num_classes": len(classes),
        "classes": classes,
    }


@app.post("/snapshot")
async def snapshot(file: UploadFile = File(...)) -> JSONResponse:
    """Run detection on an image and persist the result + frame."""
    engine: Engine = STATE["engine"]
    if engine is None:
        raise HTTPException(503, "Engine not ready")
    raw = await file.read()
    frame = _decode(raw)
    loop = asyncio.get_running_loop()
    world = await loop.run_in_executor(None, engine.process, frame)
    payload = world.to_dict()
    payload["image"] = {"width": frame.shape[1],
                        "height": frame.shape[0]}
    sid = storage.save_snapshot(payload, raw)
    return JSONResponse({"id": sid, "world": payload})


@app.get("/snapshots")
async def snapshots() -> dict:
    return {"ids": storage.list_snapshots()}


@app.get("/snapshots/{sid}")
async def get_snapshot(sid: str) -> JSONResponse:
    try:
        return JSONResponse(storage.load_snapshot(sid))
    except FileNotFoundError:
        raise HTTPException(404, "No such snapshot")


@app.post("/train")
async def train(payload: dict, background_tasks: BackgroundTasks) -> dict:
    """Accept an annotation payload from the studio and train on it.

    Chapter 25 submits the job; Chapter 27 runs it. If the payload
    carries image pixels, we build a YOLO dataset and launch training in
    the background (FastAPI runs the sync job in a threadpool, so the
    event loop is not blocked). The studio polls /train/{job_id}.
    """
    from .training import jobs, dataset
    if not payload.get("images"):
        raise HTTPException(400, "No annotated images in payload")
    job_id = jobs.submit(payload)
    has_pixels = any(im.get("data") or im.get("dataUrl")
                     for im in payload["images"])
    if has_pixels:
        ds_dir = os.path.join("data", "datasets", job_id)
        try:
            dataset.build_from_payload(payload, ds_dir)
            if jobs.has_keypoints(payload):
                jobs.set_status(job_id, {"status": "queued", "dataset": ds_dir,
                                         "mode": "pose"})
                background_tasks.add_task(jobs.run_pose, job_id, ds_dir)
            elif jobs.is_hierarchical(payload):
                jobs.set_status(job_id, {"status": "queued", "dataset": ds_dir,
                                         "mode": "hierarchical"})
                background_tasks.add_task(jobs.run_hierarchical, job_id, ds_dir)
            else:
                jobs.set_status(job_id, {"status": "queued", "dataset": ds_dir})
                background_tasks.add_task(jobs.run, job_id, ds_dir)
        except Exception as exc:
            jobs.set_status(job_id, {"status": "failed", "error": str(exc)})
    else:
        jobs.set_status(job_id, {"status": "submitted",
            "note": "No image data in payload; training not auto-started."})
    return jobs.status(job_id)


@app.get("/train/{job_id}")
async def train_status(job_id: str) -> dict:
    from .training import jobs
    st = jobs.status(job_id)
    if st is None:
        raise HTTPException(404, "No such job")
    return st


@app.get("/models")
async def list_models() -> dict:
    """List selectable model sets: the default COCO model plus any folder
    under models/ that holds trained weights or a hierarchy manifest."""
    sets = [{"name": "default", "label": "Default \u2014 YOLO26 (COCO 80)",
             "type": "coco"}]
    root = "models"
    if os.path.isdir(root):
        for name in sorted(os.listdir(root)):
            d = os.path.join(root, name)
            if not os.path.isdir(d):
                continue
            if os.path.exists(os.path.join(d, "hierarchy.json")):
                sets.append({"name": name, "label": name + " (multi-phase)",
                             "type": "hierarchical"})
            elif os.path.exists(os.path.join(d, "weights", "best.pt")):
                sets.append({"name": name, "label": name, "type": "flat"})
    return {"models": sets, "active": STATE.get("model_set", "default")}


def _build_engine_for(name: str) -> Engine:
    """Build an engine for a model set: COCO, a flat best.pt, or a
    multi-phase hierarchy manifest. Inherits device/precision from the
    running config so it works on CPU-only machines too."""
    base = STATE.get("config") or EngineConfig()
    dev, half = base.device, base.half
    if not name or name == "default":
        return build_engine(EngineConfig(device=dev, half=half))
    d = os.path.join("models", name)
    manifest = os.path.join(d, "hierarchy.json")
    if os.path.exists(manifest):
        from .engine.hier_detector import HierarchicalDetector
        det = HierarchicalDetector.from_manifest(
            manifest, device=dev, imgsz=base.img_size,
            conf=base.conf_threshold)
        return build_engine(EngineConfig(device=dev, half=half), detector=det)
    best = os.path.join(d, "weights", "best.pt")
    if os.path.exists(best):
        return build_engine(EngineConfig(weights=best, device=dev, half=half))
    raise HTTPException(404, f"model set '{name}' has no usable weights")


@app.post("/models/select")
async def select_model(payload: dict) -> dict:
    """Switch the engine to a chosen model set (affects all clients)."""
    name = (payload or {}).get("name", "default")
    try:
        STATE["engine"] = _build_engine_for(name)
        STATE["model_set"] = name
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, f"failed to load model set: {exc}")
    return {"active": name, "ok": True}


# The live mobile path posts camera frames to /detect over HTTPS (see the
# frontend in web/js/rtc.js and Appendix G.4). No WebRTC peer is involved,
# which is also why the live view works through an ngrok tunnel.

# Serve the web frontend (Chapter 8) when the folder exists.
WEB_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                       "web")
if os.path.isdir(WEB_DIR):
    app.mount("/", StaticFiles(directory=WEB_DIR, html=True),
              name="web")
