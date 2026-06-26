"""Build a YOLO dataset on disk from a studio annotation payload.

The studio (Chapter 25) sends classes, per-image annotations, and the
image data URLs. This decodes the images, writes normalized YOLO label
files, lays them out as images/{train,val} + labels/{train,val}, and
returns the dataset directory for the orchestrator (Chapter 27).
"""
from __future__ import annotations
import base64
import os
import random
import shutil


def _decode_data_url(data_url: str) -> bytes:
    """'data:image/png;base64,AAAA...' -> raw image bytes."""
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]
    return base64.b64decode(data_url)


def _yolo_lines(objects, classes, width, height) -> list[str]:
    """Convert pixel-space object and part boxes to normalized YOLO lines.
    Studio-annotated sub-parts are emitted alongside their parent object so
    the detector learns to find parts as well."""
    idx = {c: i for i, c in enumerate(classes)}
    lines = []
    if not width or not height:
        return lines

    def emit(cls, box):
        if cls not in idx or not box or len(box) != 4:
            return
        x1, y1, x2, y2 = box
        cx = (x1 + x2) / 2 / width
        cy = (y1 + y2) / 2 / height
        w = abs(x2 - x1) / width
        h = abs(y2 - y1) / height
        lines.append(f"{idx[cls]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    for o in objects:
        emit(o.get("cls"), o.get("box"))
        for p in o.get("parts", []):           # studio-annotated sub-parts
            emit(p.get("cls"), p.get("box"))
    return lines


def _canonical_keypoints(images) -> list[str]:
    """First-seen union of keypoint names across all annotated objects. This
    fixed order defines the columns of every pose label and the kpt_shape."""
    names: list[str] = []
    for im in images:
        for o in im.get("objects", []):
            for kp in o.get("keypoints", []):
                n = kp.get("name")
                if n and n not in names:
                    names.append(n)
    return names


def _pose_lines(objects, classes, kp_names, width, height) -> list[str]:
    """YOLO pose labels: 'cls cx cy w h (kx ky v)*K' with keypoints written in
    the canonical order. Absent keypoints are written as '0 0 0' (v=0)."""
    idx = {c: i for i, c in enumerate(classes)}
    lines = []
    if not width or not height:
        return lines
    for o in objects:
        cls, box = o.get("cls"), o.get("box")
        if cls not in idx or not box or len(box) != 4:
            continue
        x1, y1, x2, y2 = box
        cx = (x1 + x2) / 2 / width
        cy = (y1 + y2) / 2 / height
        w = abs(x2 - x1) / width
        h = abs(y2 - y1) / height
        cols = [f"{idx[cls]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"]
        have = {kp.get("name"): kp for kp in o.get("keypoints", [])}
        for name in kp_names:
            kp = have.get(name)
            if kp and kp.get("x") is not None and kp.get("y") is not None:
                kx = float(kp["x"]) / width
                ky = float(kp["y"]) / height
                v = 0 if kp.get("visible") is False else 2
                cols.append(f"{kx:.6f} {ky:.6f} {v}")
            else:
                cols.append("0 0 0")
        lines.append(" ".join(cols))
    return lines


def build_from_payload(payload: dict, out_dir: str,
                       val_fraction: float = 0.2, seed: int = 0) -> str:
    """Materialize the payload into a YOLO dataset and return out_dir."""
    classes = list(payload.get("classes", []))
    images = payload.get("images", [])
    # If any object carries keypoints, this is a POSE dataset (train a pose
    # model on objects + keypoints). Otherwise it is a detection dataset, and
    # we additionally train studio-annotated sub-parts as detectable classes.
    kp_names = _canonical_keypoints(images)
    pose = len(kp_names) > 0
    primitives = {}                          # part class -> primitive (w/ rotation)
    if not pose:
        for im in images:
            for o in im.get("objects", []):
                for p in o.get("parts", []):
                    pc = p.get("cls")
                    if pc and pc not in classes:
                        classes.append(pc)
                    if pc and p.get("primitive"):
                        primitives[pc] = p["primitive"]
    for split in ("train", "val"):
        os.makedirs(os.path.join(out_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(out_dir, "labels", split), exist_ok=True)

    # Decide which images go to validation.
    order = list(range(len(images)))
    random.Random(seed).shuffle(order)
    n_val = max(1, int(len(images) * val_fraction)) if len(images) > 1 else 0
    val_set = set(order[:n_val])

    written = 0
    for i, im in enumerate(images):
        data = im.get("data") or im.get("dataUrl")
        if not data:
            continue                         # no pixels -> cannot train on it
        try:
            raw = _decode_data_url(data)
        except Exception:
            continue
        split = "val" if i in val_set else "train"
        name = im.get("name") or f"img_{i}.jpg"
        stem, ext = os.path.splitext(name)
        if not ext:
            ext, name = ".jpg", stem + ".jpg"
        with open(os.path.join(out_dir, "images", split, name), "wb") as fh:
            fh.write(raw)
        w_ = im.get("width") or im.get("w")
        h_ = im.get("height") or im.get("h")
        lines = (_pose_lines(im.get("objects", []), classes, kp_names, w_, h_)
                 if pose else
                 _yolo_lines(im.get("objects", []), classes, w_, h_))
        with open(os.path.join(out_dir, "labels", split, stem + ".txt"),
                  "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines))
        written += 1

    if written == 0:
        raise ValueError("No usable images in payload (missing image data).")

    # Tiny datasets may end up with an empty val split; mirror train so
    # the trainer can still validate.
    _ensure_val(out_dir)

    # Write data.yaml here, with the parts-augmented class list, so it is the
    # single source of truth; the orchestrator reuses this file if present and
    # the label indices above always match its names.
    names = "\n".join(f"  {i}: {c}" for i, c in enumerate(classes))
    root = os.path.abspath(out_dir).replace("\\", "/")
    pose_hdr = ""
    if pose:
        pose_hdr = (f"kpt_shape: [{len(kp_names)}, 3]\n"
                    f"flip_idx: {list(range(len(kp_names)))}\n")
    with open(os.path.join(out_dir, "data.yaml"), "w", encoding="utf-8") as fh:
        fh.write(f"path: {root}\ntrain: images/train\nval: images/val\n"
                 f"{pose_hdr}names:\n{names}\n")
    # Persist part-primitive definitions (shape + rotation + axis) so the
    # cascade can reattach them to detected parts at inference time.
    if primitives:
        import json as _json
        with open(os.path.join(out_dir, "primitives.json"), "w",
                  encoding="utf-8") as fh:
            _json.dump(primitives, fh)
    # Save image crops + rotation labels for the trainable pose/6-DoF head.
    try:
        from . import pose_head
        pose_head.save_pose_samples(payload, out_dir)
    except Exception:
        pass
    return out_dir


def _ensure_val(out_dir: str) -> None:
    val_img = os.path.join(out_dir, "images", "val")
    if os.listdir(val_img):
        return
    for sub in ("images", "labels"):
        src = os.path.join(out_dir, sub, "train")
        dst = os.path.join(out_dir, sub, "val")
        for f in os.listdir(src):
            shutil.copy2(os.path.join(src, f), os.path.join(dst, f))
