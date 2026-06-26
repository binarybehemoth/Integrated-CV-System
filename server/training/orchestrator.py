"""Run a real training job.

Consumes a dataset (from the studio export of Chapter 25 or the Open
Images subset of Chapter 26) and fine-tunes a YOLO26 model with
Ultralytics. Chapter 27. Designed so the heavy import is lazy and the
progress is reported back into the in-memory job registry.
"""
from __future__ import annotations
import json
import os
import time
from typing import Callable, Optional


def _dataset_yaml(dataset_dir: str, classes: list[str]) -> str:
    """Write a YOLO data.yaml pointing at images/ and labels/. If the dataset
    builder already wrote one (with parts-augmented classes), reuse it so the
    label indices and names always agree."""
    path = os.path.join(dataset_dir, "data.yaml")
    if os.path.exists(path):
        return path
    names = "\n".join(f"  {i}: {c}" for i, c in enumerate(classes))
    # Forward slashes are safe in YAML on every OS; a raw Windows path
    # with backslashes can be misread.
    root = os.path.abspath(dataset_dir).replace("\\", "/")
    text = (
        f"path: {root}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"names:\n{names}\n"
    )
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    return path


def train_model(dataset_dir: str, classes: list[str],
                base_weights: str = "yolo26n.pt", epochs: int = 50,
                imgsz: int = 640, project: str = "runs/train",
                name: str = "custom",
                on_epoch: Optional[Callable[[int, int], None]] = None) -> str:
    """Fine-tune ``base_weights`` on the dataset and return the path to
    the best weights. Requires ultralytics and a prepared dataset.
    """
    from ultralytics import YOLO

    data_yaml = _dataset_yaml(dataset_dir, classes)
    # Drop any stale label caches so a previous failed verification can't
    # poison this run.
    import glob
    for cache in glob.glob(os.path.join(dataset_dir, "labels", "*.cache")):
        try:
            os.remove(cache)
        except Exception:
            pass
    model = YOLO(base_weights)             # start from pretrained weights

    if on_epoch is not None:
        def _cb(trainer):                  # Ultralytics callback hook
            on_epoch(trainer.epoch + 1, epochs)
        model.add_callback("on_train_epoch_end", _cb)

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        project=project,
        name=name,
        exist_ok=True,
    )
    # Ultralytics saves weights/best.pt under the run directory.
    save_dir = getattr(results, "save_dir", os.path.join(project, name))
    best = os.path.join(str(save_dir), "weights", "best.pt")
    return best


def _write_level2_manifest(weights: str, parent_class: str, task: str) -> None:
    """Register a freshly-trained model as a level-2 model under its parent
    class, so the two-phase cascade (engine/cascade.py) picks it up on the next
    request. ``weights`` is .../models/<name>/weights/best.pt; the manifest goes
    beside it at .../models/<name>/level2.json."""
    if not parent_class or not weights:
        return
    model_dir = os.path.dirname(os.path.dirname(str(weights)))
    try:
        os.makedirs(model_dir, exist_ok=True)
        with open(os.path.join(model_dir, "level2.json"), "w") as fh:
            json.dump({"parent_class": parent_class, "weights": str(weights),
                       "task": task}, fh)
    except Exception:
        pass


def run_job(job: dict, dataset_dir: str,
            set_status: Callable[[str, dict], None]) -> None:
    """Execute a persisted job end to end, updating its status.

    ``set_status(job_id, patch)`` merges fields into the job record so
    the /train/{job_id} endpoint reflects live progress.
    """
    job_id = job["job_id"]
    classes = job.get("classes", [])
    cfg = job.get("config", {})
    set_status(job_id, {"status": "running", "started": time.time(),
                        "progress": 0.0})
    try:
        def _progress(done: int, total: int) -> None:
            set_status(job_id, {"progress": round(done / max(total, 1), 3),
                                "epoch": done, "epochs": total})

        best = train_model(
            dataset_dir, classes,
            base_weights=cfg.get("model", "yolo26n.pt"),
            epochs=int(cfg.get("epochs", 50)),
            imgsz=int(cfg.get("imgsz", 640)),
            project="models",
            name=cfg.get("name") or job_id,
            on_epoch=_progress,
        )
        set_status(job_id, {"status": "done", "weights": best,
                            "model_set": cfg.get("name") or job_id,
                            "finished": time.time(), "progress": 1.0})
        _write_level2_manifest(best, cfg.get("parent_class", ""), "detect")
        # Carry the part-primitive manifest (shape + rotation + axis) into the
        # model dir so the cascade can reattach primitives to detected parts.
        src = os.path.join(dataset_dir, "primitives.json")
        if os.path.exists(src):
            try:
                model_dir = os.path.dirname(os.path.dirname(str(best)))
                with open(src) as _f:
                    _data = _f.read()
                with open(os.path.join(model_dir, "primitives.json"), "w") as _o:
                    _o.write(_data)
            except Exception:
                pass
        # Train the orientation / 6-DoF pose head on the annotated rotations,
        # so rotation is predicted from pixels at inference (not just copied).
        try:
            from . import pose_head
            model_dir = os.path.dirname(os.path.dirname(str(best)))
            pose_dir = os.path.join(dataset_dir, "pose")
            if os.path.exists(os.path.join(pose_dir, "labels.json")):
                set_status(job_id, {"status": "training pose head"})
                wp = pose_head.train_pose_head(
                    pose_dir, os.path.join(model_dir, "pose_head.pt"))
                set_status(job_id, {"status": "done",
                                    "pose_head": bool(wp)})
        except Exception:
            pass
    except Exception as exc:               # pragma: no cover - runtime guard
        set_status(job_id, {"status": "failed", "error": str(exc),
                            "finished": time.time()})


def run_pose_job(job: dict, dataset_dir: str,
                 set_status: Callable[[str, dict], None]) -> None:
    """Train a YOLO pose model (object boxes + keypoints). The dataset's
    data.yaml carries kpt_shape, and the base is a -pose checkpoint, so
    Ultralytics trains the pose task."""
    job_id = job["job_id"]
    classes = job.get("classes", [])
    cfg = job.get("config", {})
    set_status(job_id, {"status": "running", "started": time.time(),
                        "progress": 0.0})
    try:
        def _progress(done: int, total: int) -> None:
            set_status(job_id, {"progress": round(done / max(total, 1), 3),
                                "epoch": done, "epochs": total})

        # A pose dataset MUST be trained with a -pose checkpoint, otherwise
        # Ultralytics reads the keypoint columns as segmentation polygons and
        # fails ("cannot reshape ... into shape (2)"). Ignore a non-pose model
        # sent by the client and force a pose base.
        pose_base = cfg.get("model") or "yolo26n-pose.pt"
        if "pose" not in str(pose_base):
            pose_base = "yolo26n-pose.pt"
        best = train_model(
            dataset_dir, classes,
            base_weights=pose_base,
            epochs=int(cfg.get("epochs", 50)),
            imgsz=int(cfg.get("imgsz", 640)),
            project="models",
            name=cfg.get("name") or job_id,
            on_epoch=_progress,
        )
        set_status(job_id, {"status": "done", "weights": best, "task": "pose",
                            "model_set": cfg.get("name") or job_id,
                            "finished": time.time(), "progress": 1.0})
        _write_level2_manifest(best, cfg.get("parent_class", ""), "pose")
    except Exception as exc:               # pragma: no cover - runtime guard
        set_status(job_id, {"status": "failed", "error": str(exc),
                            "finished": time.time()})


def run_hierarchical_job(job: dict, dataset_dir: str,
                         set_status: Callable[[str, dict], None]) -> None:
    """Multi-phase training: a parent model + one child model per group.

    Used when the job's hierarchy groups leaf classes under parents, so
    many classes train as several small models instead of one big one.
    """
    from . import hierarchical
    job_id = job["job_id"]
    classes = job.get("classes", [])
    hierarchy = job.get("hierarchy", {})
    cfg = job.get("config", {})
    name = cfg.get("name") or job_id
    out_dir = os.path.join("models", name)
    set_status(job_id, {"status": "running", "mode": "hierarchical",
                        "started": time.time(), "progress": 0.0})
    try:
        manifest = hierarchical.train_hierarchical(
            dataset_dir, classes, hierarchy, out_dir,
            epochs=int(cfg.get("epochs", 50)),
            imgsz=int(cfg.get("imgsz", 640)),
            base=cfg.get("model", "yolo26n.pt"))
        set_status(job_id, {"status": "done", "progress": 1.0,
                            "manifest": os.path.join(out_dir, "hierarchy.json"),
                            "parent_model": manifest.get("parent_model"),
                            "child_models": manifest.get("child_models"),
                            "finished": time.time()})
    except Exception as exc:               # pragma: no cover - runtime guard
        set_status(job_id, {"status": "failed", "error": str(exc),
                            "finished": time.time()})
