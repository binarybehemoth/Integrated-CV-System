"""Open-vocabulary detection with YOLOE (Real-Time Seeing Anything).

YOLOE (Wang et al., ICCV 2025; github.com/THU-MIG/yoloe) detects and segments
object classes that are named at *runtime* by a text prompt, with no per-class
training. It is shipped inside Ultralytics, so we load it through the same
package as the rest of our detectors and drive it with set_classes():

    model = YOLOE("yoloe-11s-seg.pt")
    model.set_classes(names, model.get_text_pe(names))   # encode the prompt
    results = model.predict(frame)

We wrap that behind a small, thread-safe cache so the model loads once and the
text embeddings are recomputed only when the prompt actually changes. The model
name is configurable with CV_YOLOE_MODEL (default yoloe-11s-seg.pt); set it to
a yoloe-26*-seg.pt checkpoint to use the newest generation.
docs.ultralytics.com/models/yoloe
"""
from __future__ import annotations

import os
import threading

from .world_model import WorldModel, Object, BBox

_MODEL_NAME = os.environ.get("CV_YOLOE_MODEL", "yoloe-11s-seg.pt")


class OpenVocabDetector:
    """Lazily-loaded YOLOE wrapper. One model; the prompt is cached so the
    text encoder runs only when the requested class list changes."""

    def __init__(self, model_name: str = _MODEL_NAME):
        self.model_name = model_name
        self._model = None
        self._classes: list[str] = []
        self._lock = threading.Lock()

    def _ensure_model(self) -> None:
        if self._model is None:
            from ultralytics import YOLOE          # imported lazily
            self._model = YOLOE(self.model_name)

    def _ensure_classes(self, names: list[str]) -> None:
        if names != self._classes:                 # re-encode only on change
            self._model.set_classes(names, self._model.get_text_pe(names))
            self._classes = list(names)

    def detect(self, frame, prompt: str, conf: float = 0.25) -> WorldModel:
        """Run YOLOE on ``frame`` for the comma-separated ``prompt`` classes
        and return a WorldModel of the matches (boxes, masks, confidences)."""
        names = [p.strip() for p in prompt.split(",") if p.strip()]
        world = WorldModel()
        if not names:
            return world

        with self._lock:
            self._ensure_model()
            self._ensure_classes(names)
            results = self._model.predict(frame, conf=conf, verbose=False)

        if not results:
            return world
        r = results[0]
        boxes = getattr(r, "boxes", None)
        if boxes is None or boxes.xyxy is None:
            return world

        xyxy = boxes.xyxy.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int)
        confs = boxes.conf.cpu().numpy()
        masks = getattr(r, "masks", None)
        polys = masks.xy if masks is not None else None

        for i in range(len(xyxy)):
            x1, y1, x2, y2 = (float(v) for v in xyxy[i])
            ci = int(cls[i])
            name = names[ci] if 0 <= ci < len(names) else str(ci)
            obj = Object(id=i, cls=name, box=BBox(x1, y1, x2, y2),
                         confidence=float(confs[i]))
            if polys is not None and i < len(polys):
                # Down-sample the contour to keep the JSON small.
                obj.mask = [[float(x), float(y)] for x, y in polys[i][::2]]
            world.objects.append(obj)
        return world

    def _ensure_pf_model(self):
        """Lazily load a *prompt-free* YOLOE checkpoint (its own built-in
        4585-class vocabulary). Separate from the prompted model so both can
        coexist. Override with CV_YOLOE_PF_MODEL."""
        if getattr(self, "_pf", None) is None:
            from ultralytics import YOLO
            name = os.environ.get("CV_YOLOE_PF_MODEL", "yoloe-11s-seg-pf.pt")
            self._pf = YOLO(name)
        return self._pf

    def detect_prompt_free(self, frame, conf: float = 0.25) -> WorldModel:
        """Phase 1 of the cascade: YOLOE prompt-free over its built-in
        vocabulary. Class names come from the model's own ``names`` map, so no
        prompt is supplied and up to ~4585 categories can be returned."""
        world = WorldModel()
        with self._lock:
            model = self._ensure_pf_model()
            results = model.predict(frame, conf=conf, verbose=False)
        if not results:
            return world
        r = results[0]
        boxes = getattr(r, "boxes", None)
        if boxes is None or boxes.xyxy is None:
            return world
        names = getattr(r, "names", {}) or {}
        xyxy = boxes.xyxy.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int)
        confs = boxes.conf.cpu().numpy()
        masks = getattr(r, "masks", None)
        polys = masks.xy if masks is not None else None
        for i in range(len(xyxy)):
            x1, y1, x2, y2 = (float(v) for v in xyxy[i])
            obj = Object(id=i, cls=names.get(int(cls[i]), str(int(cls[i]))),
                         box=BBox(x1, y1, x2, y2), confidence=float(confs[i]))
            if polys is not None and i < len(polys):
                obj.mask = [[float(x), float(y)] for x, y in polys[i][::2]]
            world.objects.append(obj)
        return world
