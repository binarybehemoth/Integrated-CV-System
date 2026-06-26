"""Reading text in the scene (OCR): detect text regions and recognise
their characters, adding them to the world model as text objects.

Backend-agnostic with an EasyOCR default (CRAFT detector + CRNN/CTC
recogniser). See Chapter 14.
"""
from __future__ import annotations

from .capability import Capability
from .world_model import WorldModel, Object, BBox
from .config import EngineConfig


class OcrCapability(Capability):
    name = "ocr"

    def __init__(self, config: EngineConfig | None = None,
                 languages=("en",), min_conf: float = 0.3):
        self.cfg = config or EngineConfig()
        self.languages = list(languages)
        self.min_conf = min_conf
        self.reader = None

    def setup(self) -> None:
        import easyocr
        gpu = str(self.cfg.device).startswith("cuda")
        self.reader = easyocr.Reader(self.languages, gpu=gpu)

    def apply(self, world: WorldModel, frame) -> None:
        if self.reader is None:
            return
        results = self.reader.readtext(frame)   # [(quad, text, conf), ...]
        next_id = max((o.id for o in world.objects), default=-1) + 1
        for quad, text, conf in results:
            if conf < self.min_conf or not text.strip():
                continue
            xs = [float(p[0]) for p in quad]
            ys = [float(p[1]) for p in quad]
            box = BBox(x1=min(xs), y1=min(ys), x2=max(xs), y2=max(ys))
            obj = Object(id=next_id, cls="text", box=box,
                         confidence=float(conf))
            obj.properties["text"] = text.strip()
            world.objects.append(obj)
            next_id += 1
