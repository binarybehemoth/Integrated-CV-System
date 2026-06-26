"""Two-phase cascade detection (the fixed level-1 / custom level-2 design).

Phase 1  YOLOE detects the fixed level-1 vocabulary (its built-in ~4585
         classes), prompt-free.
Phase 2  For each phase-1 box whose class has a trained level-2 model, the box
         is cropped and the level-2 model runs on that crop to (a) refine the
         class to a custom leaf, (b) add parts, and (c) add keypoints. Results
         are mapped back into full-frame coordinates and merged into the same
         object, with ``parent_class`` recording the level-1 class it came from.

Level-2 models register themselves with a tiny manifest written at training
time:  models/<name>/level2.json = {"parent_class": str, "weights": path,
"task": "detect"|"pose"|"segment"}. The cascade scans for these lazily, so
training a new level-2 model makes it active on the next request with no wiring.
"""
from __future__ import annotations

import glob
import json
import os

from .world_model import WorldModel, Object, BBox, Keypoint


class CascadeDetector:
    def __init__(self, open_vocab, models_dir: str = "models"):
        self.open_vocab = open_vocab          # OpenVocabDetector (YOLOE)
        self.models_dir = models_dir
        self._models: dict = {}               # weights path -> loaded YOLO
        self._reg: dict | None = None         # parent_class -> [manifest, ...]

    # ---- level-2 registry -------------------------------------------------
    def registry(self, refresh: bool = False) -> dict:
        if self._reg is None or refresh:
            reg: dict = {}
            pattern = os.path.join(self.models_dir, "*", "level2.json")
            for manifest_path in glob.glob(pattern):
                try:
                    with open(manifest_path) as fh:
                        man = json.load(fh)
                except Exception:
                    continue
                parent = man.get("parent_class")
                if not parent:
                    continue
                man.setdefault("weights", os.path.join(
                    os.path.dirname(manifest_path), "weights", "best.pt"))
                man.setdefault("task", "detect")
                # part-primitive definitions (shape + rotation + axis) live
                # beside the manifest; attach them so detected parts render in 3D.
                prim_path = os.path.join(os.path.dirname(manifest_path),
                                         "primitives.json")
                man["primitives"] = {}
                if os.path.exists(prim_path):
                    try:
                        with open(prim_path) as pf:
                            man["primitives"] = json.load(pf)
                    except Exception:
                        man["primitives"] = {}
                # trained orientation / 6-DoF head, if present
                pose_path = os.path.join(os.path.dirname(manifest_path),
                                         "pose_head.pt")
                man["pose"] = None
                if os.path.exists(pose_path):
                    try:
                        from ..training.pose_head import PosePredictor
                        man["pose"] = PosePredictor(pose_path)
                    except Exception:
                        man["pose"] = None
                reg.setdefault(parent, []).append(man)
            self._reg = reg
        return self._reg

    def _model(self, weights: str):
        if weights not in self._models:
            from ultralytics import YOLO
            self._models[weights] = YOLO(weights)
        return self._models[weights]

    # ---- inference --------------------------------------------------------
    def detect(self, frame, prompt: str = "", conf: float = 0.25) -> WorldModel:
        # Phase 1: level-1 detection. A prompt narrows it; otherwise the full
        # built-in vocabulary is used.
        if prompt and prompt.strip():
            world = self.open_vocab.detect(frame, prompt, conf)
        else:
            world = self.open_vocab.detect_prompt_free(frame, conf)

        reg = self.registry()
        if not reg:
            return world

        h, w = frame.shape[:2]
        for obj in list(world.objects):
            # Level-2: refine each level-1 box with its registered specialist
            # (reclassify into the custom class + attach parts/keypoints/pose/
            # primitives/properties). One pass; depth lives in the class names.
            children = reg.get(obj.cls)
            if not children:
                continue
            x1 = max(0, int(obj.box.x1)); y1 = max(0, int(obj.box.y1))
            x2 = min(w, int(obj.box.x2)); y2 = min(h, int(obj.box.y2))
            if x2 - x1 < 4 or y2 - y1 < 4:
                continue
            crop = frame[y1:y2, x1:x2]
            for man in children:
                weights = man.get("weights")
                if not weights or not os.path.exists(weights):
                    continue
                try:
                    results = self._model(weights).predict(
                        crop, conf=conf, verbose=False)
                except Exception:
                    continue
                self._merge(obj, results, x1, y1, man.get("task", "detect"),
                            man.get("primitives", {}))
                pose = man.get("pose")
                if pose is not None and getattr(pose, "ok", False):
                    self._apply_pose(obj, frame, pose)
        return world

    def _apply_pose(self, parent: Object, frame, pose) -> None:
        """Predict each part's rotation from its image crop and write it into
        the part primitive (overriding the static annotation value)."""
        h, w = frame.shape[:2]
        for part in parent.parts:
            prim = part.properties.get("primitive")
            if not prim:
                continue
            b = part.box
            x1 = max(0, int(b.x1)); y1 = max(0, int(b.y1))
            x2 = min(w, int(b.x2)); y2 = min(h, int(b.y2))
            if x2 - x1 < 6 or y2 - y1 < 6:
                continue
            pred = pose.predict(frame[y1:y2, x1:x2])
            if not pred:
                continue
            angle, axis = pred
            prim = dict(prim)                 # don't mutate the shared manifest
            prim["rotation"] = round(angle, 1)
            prim["axisX"], prim["axisY"], prim["axisZ"] = [round(a, 3) for a in axis]
            prim["pose_predicted"] = True
            part.properties["primitive"] = prim

    # ---- merge a level-2 result back into its parent object ---------------
    def _merge(self, parent: Object, results, ox: int, oy: int, task: str,
               prims: dict = None):
        prims = prims or {}
        if not results:
            return
        r = results[0]
        boxes = getattr(r, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return
        names = getattr(r, "names", {}) or {}
        confs = boxes.conf.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int)
        xyxy = boxes.xyxy.cpu().numpy()

        # The most confident child detection refines the parent's class.
        best = int(confs.argmax())
        leaf = names.get(int(cls[best]), str(int(cls[best])))
        parent.parent_class = parent.cls
        parent.cls = leaf
        parent.confidence = float(confs[best])

        # Remaining child detections become parts nested under the parent, each
        # carrying its trained primitive (shape + rotation + axis) so the 3D/
        # WebGL view can render it.
        for i in range(len(xyxy)):
            if i == best:
                continue
            px1, py1, px2, py2 = (float(v) for v in xyxy[i])
            pcls = names.get(int(cls[i]), str(int(cls[i])))
            part = Object(
                id=1000 + len(parent.parts),
                cls=pcls,
                box=BBox(px1 + ox, py1 + oy, px2 + ox, py2 + oy),
                confidence=float(confs[i]),
                parent_class=parent.cls)
            if pcls in prims:
                part.properties["primitive"] = prims[pcls]
            parent.parts.append(part)

        # Keypoints from a pose level-2 model, mapped to full-frame coords.
        kp = getattr(r, "keypoints", None)
        if kp is not None and getattr(kp, "xy", None) is not None \
                and len(kp.xy) > best:
            pts = kp.xy[best].cpu().numpy()
            for j, (kx, ky) in enumerate(pts):
                if kx == 0 and ky == 0:        # absent keypoint
                    continue
                parent.keypoints.append(Keypoint(
                    name=f"kp{j}", x=float(kx) + ox, y=float(ky) + oy))
