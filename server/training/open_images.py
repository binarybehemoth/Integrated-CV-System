"""Download Open Images V7 images for a SINGLE class on demand and return them
with bounding boxes in pixel coordinates -- the exact shape the annotation
studio consumes. Only the
requested class is fetched: FiftyOne downloads just the images whose detections
include that class, so nothing is pulled in bulk and nothing from Open Images is
bundled with the project.

We use FiftyOne (``pip install fiftyone``), which wraps the official Open Images
download and lets you restrict by class in one call. The OIDv4_ToolKit CLI is an
equivalent standalone alternative; this module standardises on FiftyOne for a
clean in-process integration.

Licensing: Open Images *annotations* are CC BY 4.0, and each image keeps its own
licence (most are CC BY 2.0). Attribute the original authors when you use them.
"""
from __future__ import annotations

import base64
import io
import os
import time

OPEN_IMAGES_LICENSE = (
    "Open Images annotations are CC BY 4.0; each image keeps its own licence "
    "(mostly CC BY 2.0). Attribute the original authors when you use them."
)

_DATASET = "open-images-v7"
_SPLIT = "validation"            # far smaller than train; ample for bootstrapping


def _resolve_class(name: str):
    """Case-insensitively match ``name`` to an Open Images detection class.

    The studio's vocabulary is lower-case and may use underscores; Open Images
    uses capitalised display names with spaces (e.g. ``"Coffee cup"``)."""
    from fiftyone.utils.openimages import get_classes

    def norm(s: str) -> str:
        return s.strip().lower().replace("_", " ")

    table = {norm(c): c for c in get_classes()}
    return table.get(norm(name))


def fetch(class_name: str, n: int = 10, exclude=None) -> list:
    """Download up to ``n`` Open Images samples containing ``class_name`` and
    return a list of dicts shaped like::

        {name, width, height, source_id, data (base64 PNG), boxes:[{box:[...]}]}

    ``source_id`` is the stable Open Images image id; pass already-imported ids
    in ``exclude`` and they are skipped, so repeated calls return NEW images.
    Boxes are in pixel coordinates and restricted to the requested class.
    Raises RuntimeError if FiftyOne/Pillow are missing or the class is unknown.
    """
    try:
        import fiftyone.zoo as foz
        from fiftyone.core.labels import Detections
        from PIL import Image
    except Exception as exc:  # pragma: no cover - depends on optional install
        raise RuntimeError(
            "FiftyOne and Pillow are required for Open Images downloads: "
            "pip install fiftyone pillow") from exc

    oi_class = _resolve_class(class_name)
    if not oi_class:
        raise RuntimeError(
            f"'{class_name}' is not an Open Images class. Open Images V7 covers "
            "~600 boxable categories; choose a class whose name matches one "
            "(matching is case-insensitive and ignores underscores).")

    n = max(1, int(n))
    exset = {str(x) for x in (exclude or [])}
    # Pull a deep enough pool that, after skipping the already-imported ids, we
    # can still return n fresh images.
    pool = n + len(exset) + 5

    # Open Images is served over HTTPS from cloud buckets. On a flaky link, a
    # VPN, or a TLS-intercepting firewall/antivirus, individual records can fail
    # to decrypt ("bad record mac"). Such failures are usually intermittent, and
    # because FiftyOne caches whatever it has already pulled, simply retrying
    # resumes the download. Persistent failures here are environmental (allow
    # Python through the OS firewall), not a code bug.
    def _transient(e):
        s = str(e).lower()
        return any(k in s for k in (
            "ssl", "record mac", "decryption", "eof occurred", "timed out",
            "connection reset", "connection aborted", "broken pipe",
            "max retries", "temporarily unavailable", "remote end closed"))

    dataset = None
    last = None
    for attempt in range(4):
        try:
            dataset = foz.load_zoo_dataset(
                _DATASET,
                split=_SPLIT,
                label_types=["detections"],
                classes=[oi_class],
                max_samples=pool,
            )
            break
        except Exception as exc:           # re-raised below if it is not transient
            if not _transient(exc):
                raise
            last = exc
            time.sleep(1.5 * (attempt + 1))
    if dataset is None:
        raise RuntimeError(
            "Network/TLS error while downloading from Open Images after several "
            f"attempts ({last}). This is a transport-layer failure \u2014 commonly "
            "a firewall, antivirus, VPN, or proxy corrupting TLS. On Windows, "
            "allow python.exe through Windows Defender Firewall (or disable it for "
            "the download) and turn off any HTTPS/SSL scanning, then fetch again.")

    want = class_name.strip().lower().replace(" ", "_")
    target = oi_class.lower()
    out: list = []
    for sample in dataset:
        # stable Open Images image id (filename without extension)
        src = os.path.splitext(os.path.basename(sample.filepath))[0]
        if src in exset:
            continue                          # already imported for this class
        dets = None
        for fname in sample.field_names:
            try:
                value = sample[fname]
            except Exception:
                continue
            if isinstance(value, Detections):
                dets = value
                break
        if dets is None:
            continue
        try:
            img = Image.open(sample.filepath).convert("RGB")
        except Exception:
            continue
        w, h = img.size
        boxes = []
        for d in dets.detections:
            if str(getattr(d, "label", "")).lower() != target:
                continue
            bx, by, bw, bh = d.bounding_box       # normalised [x, y, w, h]
            x1, y1 = bx * w, by * h
            x2, y2 = (bx + bw) * w, (by + bh) * h
            boxes.append({"box": [float(x1), float(y1), float(x2), float(y2)]})
        if not boxes:
            continue
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        data = ("data:image/png;base64,"
                + base64.b64encode(buf.getvalue()).decode("ascii"))
        out.append({"name": f"oi_{want}_{src}.png", "source_id": src,
                    "width": w, "height": h, "data": data, "boxes": boxes})
        if len(out) >= n:
            break
    return out
