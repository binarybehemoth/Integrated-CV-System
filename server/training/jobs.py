"""Training job submission.

For now a job is persisted to disk and registered in memory; Chapter 27
replaces ``_run`` with a real Ultralytics training run. The studio's
"Start training" button posts a payload that lands here.
"""
from __future__ import annotations
import json
import os
import time
import uuid

JOBS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "data", "training_jobs")

# In-memory registry: job_id -> status dict.
JOBS: dict[str, dict] = {}


def submit(payload: dict) -> str:
    """Persist an annotation payload and register a pending job."""
    os.makedirs(JOBS_DIR, exist_ok=True)
    job_id = "job_" + uuid.uuid4().hex[:10]
    path = os.path.join(JOBS_DIR, job_id + ".json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh)
    n_imgs = len(payload.get("images", []))
    n_objs = sum(len(im.get("objects", [])) for im in payload.get("images", []))
    JOBS[job_id] = {
        "job_id": job_id,
        "status": "pending",
        "submitted": time.time(),
        "images": n_imgs,
        "objects": n_objs,
        "classes": payload.get("classes", []),
        "hierarchy": payload.get("hierarchy", {}),
        "config": payload.get("config", {}),
        "path": path,
    }
    return job_id


def status(job_id: str) -> dict | None:
    return JOBS.get(job_id)


def set_status(job_id: str, patch: dict) -> None:
    """Merge fields into a job record (used by the orchestrator)."""
    if job_id in JOBS:
        JOBS[job_id].update(patch)


def run(job_id: str, dataset_dir: str) -> None:
    """Run a registered job against a prepared YOLO dataset directory."""
    from . import orchestrator
    job = JOBS.get(job_id)
    if job is None:
        raise KeyError(job_id)
    orchestrator.run_job(job, dataset_dir, set_status)


def run_hierarchical(job_id: str, dataset_dir: str) -> None:
    """Run multi-phase (parent + per-parent child) training for a job."""
    from . import orchestrator
    job = JOBS.get(job_id)
    if job is None:
        raise KeyError(job_id)
    orchestrator.run_hierarchical_job(job, dataset_dir, set_status)


def run_pose(job_id: str, dataset_dir: str) -> None:
    """Run pose training (object boxes + keypoints) for a job."""
    from . import orchestrator
    job = JOBS.get(job_id)
    if job is None:
        raise KeyError(job_id)
    orchestrator.run_pose_job(job, dataset_dir, set_status)


def has_keypoints(payload: dict) -> bool:
    """True if any annotated object carries keypoints (-> pose training)."""
    return any(o.get("keypoints")
               for im in payload.get("images", [])
               for o in im.get("objects", []))


def is_hierarchical(payload: dict) -> bool:
    """True if the payload's hierarchy groups any leaf under a parent."""
    from . import hierarchical
    classes = payload.get("classes", [])
    hierarchy = payload.get("hierarchy", {}) or {}
    return any(hierarchical.parent_of(hierarchy, c) != c for c in classes)
