"""Facial recognition: detect faces, embed them, and match against a
known gallery, attaching an identity to the matching person object.

Pipeline: detect + align (RetinaFace), embed (ArcFace), match by
cosine similarity. InsightFace backend. Use responsibly and lawfully
-- see the privacy guidance in Chapter 15.
"""
from __future__ import annotations
import json
import os

import numpy as np

from .capability import Capability
from .world_model import WorldModel, Object, BBox
from .config import EngineConfig
from .geometry import iou


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return float(np.dot(a, b) / denom)


def _contains(outer, inner) -> bool:
    cx = (inner[0] + inner[2]) / 2.0
    cy = (inner[1] + inner[3]) / 2.0
    return outer[0] <= cx <= outer[2] and outer[1] <= cy <= outer[3]


class FaceCapability(Capability):
    name = "face"

    def __init__(self, config: EngineConfig | None = None,
                 gallery: dict | None = None, threshold: float = 0.5,
                 gallery_path: str | None = None):
        self.cfg = config or EngineConfig()
        self.gallery = gallery or {}      # name -> 512-d embedding
        self.threshold = threshold
        self.app = None
        self.gallery_path = gallery_path or os.environ.get(
            "CV_FACE_GALLERY", "models/face_gallery.json")
        self._mtime = 0.0
        self._person_set = None           # classes under "People & roles"

    def setup(self) -> None:
        from insightface.app import FaceAnalysis
        self.app = FaceAnalysis(name="buffalo_l")
        ctx = 0 if str(self.cfg.device).startswith("cuda") else -1
        self.app.prepare(ctx_id=ctx)
        self.load_gallery()

    # ---- persistent identity gallery -------------------------------------
    def load_gallery(self) -> None:
        """Load enrolled identities (name -> embedding) from disk, if present."""
        path = self.gallery_path
        if not path or not os.path.exists(path):
            return
        try:
            with open(path) as fh:
                data = json.load(fh)
            self.gallery = {n: np.asarray(v, dtype=np.float32)
                            for n, v in data.items()}
            self._mtime = os.path.getmtime(path)
        except Exception:
            pass

    def save_gallery(self) -> None:
        path = self.gallery_path
        if not path:
            return
        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            data = {n: np.asarray(e, dtype=np.float32).tolist()
                    for n, e in self.gallery.items()}
            with open(path, "w") as fh:
                json.dump(data, fh)
            self._mtime = os.path.getmtime(path)
        except Exception:
            pass

    def _maybe_reload(self) -> None:
        """Pick up enrollments made by another process (the /enroll_face route)
        without a restart, by reloading when the gallery file changes."""
        path = self.gallery_path
        if path and os.path.exists(path):
            try:
                if os.path.getmtime(path) > self._mtime:
                    self.load_gallery()
            except Exception:
                pass

    def remove(self, name: str) -> bool:
        if name in self.gallery:
            del self.gallery[name]
            self.save_gallery()
            return True
        return False

    def enroll(self, name: str, image) -> bool:
        """Add a person to the gallery from a clear photo of their face."""
        faces = self.app.get(image)
        if not faces:
            return False
        self.gallery[name] = faces[0].normed_embedding
        self.save_gallery()
        return True

    def identify(self, emb) -> tuple[str, float]:
        best, best_sim = "unknown", self.threshold
        for name, ref in self.gallery.items():
            sim = cosine(emb, ref)
            if sim > best_sim:
                best, best_sim = name, sim
        return best, best_sim

    # ---- "People & roles" gating -----------------------------------------
    _FALLBACK_PERSON = {"person", "man", "woman", "boy", "girl", "people",
                        "child", "kid", "athlete", "player", "pedestrian",
                        "human", "face", "rider", "dancer"}

    def _load_person_set(self) -> None:
        """Collect every class whose root supercategory is 'People & roles'
        from the vocabulary hierarchy, so recognition runs only on those boxes."""
        if self._person_set is not None:
            return
        self._person_set = set()
        path = os.environ.get("CV_VOCAB_PATH", "web/data/yoloe_vocab.json")
        try:
            with open(path) as fh:
                hier = json.load(fh).get("hierarchy", {})
            for cls in hier:
                cur, hops = cls, 0
                while hier.get(cur) and hops < 8:   # walk to the root supercat
                    cur = hier[cur]
                    hops += 1
                if cur == "People & roles":
                    self._person_set.add(cls)
        except Exception:
            pass

    def _is_person(self, cls: str, parent: str = "") -> bool:
        self._load_person_set()
        if cls in self._person_set or parent in self._person_set:
            return True
        return cls.lower() in self._FALLBACK_PERSON

    def apply(self, world: WorldModel, frame) -> None:
        """Face recognition strictly WITHIN 'person' bounding boxes. For each
        detection whose class is exactly 'person', crop its box, detect the
        face inside that crop, recognise it against the gallery, and attach the
        identity to that person. There is no whole-frame face sweep, and no
        other class (man, woman, rider, ...) is processed."""
        if self.app is None:
            return
        self._maybe_reload()
        h, w = frame.shape[:2]
        for obj in world.objects:
            if obj.cls != "person":              # ONLY the 'person' class
                continue
            x1 = max(0, int(obj.box.x1)); y1 = max(0, int(obj.box.y1))
            x2 = min(w, int(obj.box.x2)); y2 = min(h, int(obj.box.y2))
            if x2 - x1 < 12 or y2 - y1 < 12:
                continue
            crop = frame[y1:y2, x1:x2]           # detect within the box only
            try:
                faces = self.app.get(crop)
            except Exception:
                continue
            if not faces:
                continue
            # the largest face inside the person box is that person's own face
            f = max(faces, key=lambda ff: (ff.bbox[2] - ff.bbox[0]) *
                    (ff.bbox[3] - ff.bbox[1]))
            name, sim = self.identify(f.normed_embedding)
            if name != "unknown":
                obj.properties["identity"] = name
                obj.properties["identity_score"] = round(float(sim), 3)
            fx1, fy1, fx2, fy2 = (float(v) for v in f.bbox)   # map back to frame
            obj.properties["face_box"] = [round(fx1 + x1, 1), round(fy1 + y1, 1),
                                          round(fx2 + x1, 1), round(fy2 + y1, 1)]
