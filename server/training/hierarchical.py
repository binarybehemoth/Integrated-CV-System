"""Multi-phase hierarchical training.

Instead of one flat detector with many classes, train a coarse parent
model plus one child model per parent group. Each model sees only a few
classes, so the groups train independently and far faster than a single
large model. The class hierarchy comes from the studio (child -> parent).
"""
from __future__ import annotations
import json
import os
import shutil

from . import orchestrator


def _safe(name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in str(name))


def parent_of(hierarchy: dict, cls: str) -> str:
    """Climb to the top-level ancestor of a class."""
    seen = set()
    while cls in hierarchy and hierarchy[cls] and cls not in seen:
        seen.add(cls)
        cls = hierarchy[cls]
    return cls


def groups_from_hierarchy(classes: list[str], hierarchy: dict,
                          cut_level: int = 3) -> dict:
    """Two-level split for staged detection. Phase 1 detects the ancestor
    at depth ``cut_level`` (root = level 1; e.g. level 3 = "car"); phase 2
    refines to the leaf class within that box (e.g. "sedan"). Returns
    {phase1_class: [leaf classes]}. Classes shallower than cut_level group
    under themselves."""
    groups: dict[str, list[str]] = {}
    for c in classes:
        groups.setdefault(ancestor_at_level(hierarchy, c, cut_level), []).append(c)
    return groups


def _chain_to_root(hierarchy: dict, cls: str) -> list[str]:
    """Ancestor chain [root, ..., cls] (root first)."""
    chain = [cls]
    seen = {cls}
    cur = cls
    while cur in hierarchy and hierarchy[cur] and hierarchy[cur] not in seen:
        cur = hierarchy[cur]
        chain.append(cur)
        seen.add(cur)
    chain.reverse()
    return chain


def ancestor_at_level(hierarchy: dict, cls: str, level: int) -> str:
    """The ancestor at 1-based depth ``level`` (root = 1). If the class is
    shallower than ``level``, return the deepest available (the class)."""
    chain = _chain_to_root(hierarchy, cls)
    return chain[min(level, len(chain)) - 1]


def _relabel_dataset(src_dir: str, out_dir: str,
                     keep_to_newidx: dict[int, int]) -> int:
    """Copy a YOLO dataset, remapping class indices via keep_to_newidx
    (old_idx -> new_idx). Boxes whose class is not kept are dropped, and
    images left with no boxes are skipped. Returns images written."""
    written = 0
    for split in ("train", "val"):
        si = os.path.join(src_dir, "images", split)
        sl = os.path.join(src_dir, "labels", split)
        oi = os.path.join(out_dir, "images", split)
        ol = os.path.join(out_dir, "labels", split)
        os.makedirs(oi, exist_ok=True)
        os.makedirs(ol, exist_ok=True)
        if not os.path.isdir(si):
            continue
        for fn in os.listdir(si):
            stem = os.path.splitext(fn)[0]
            lbl = os.path.join(sl, stem + ".txt")
            out_lines = []
            if os.path.exists(lbl):
                with open(lbl) as fh:
                    for line in fh:
                        parts = line.split()
                        if len(parts) < 5:
                            continue
                        old = int(float(parts[0]))
                        if old in keep_to_newidx:
                            out_lines.append(" ".join(
                                [str(keep_to_newidx[old])] + parts[1:]))
            if not out_lines:
                continue
            shutil.copy2(os.path.join(si, fn), os.path.join(oi, fn))
            with open(os.path.join(ol, stem + ".txt"), "w") as fh:
                fh.write("\n".join(out_lines))
            written += 1
    # A relabel can drop every val image (its boxes weren't in this group),
    # leaving an empty val that Ultralytics refuses to load. Mirror train
    # into val so the split is always valid -- same guard as the builder.
    _mirror_train_to_val(out_dir)
    return written


def _mirror_train_to_val(out_dir: str) -> None:
    """If images/val is empty, copy images/train (and labels) into it."""
    val_img = os.path.join(out_dir, "images", "val")
    train_img = os.path.join(out_dir, "images", "train")
    if not os.path.isdir(train_img) or not os.listdir(train_img):
        return
    if os.path.isdir(val_img) and os.listdir(val_img):
        return
    for sub in ("images", "labels"):
        s = os.path.join(out_dir, sub, "train")
        dvl = os.path.join(out_dir, sub, "val")
        os.makedirs(dvl, exist_ok=True)
        if os.path.isdir(s):
            for f in os.listdir(s):
                shutil.copy2(os.path.join(s, f), os.path.join(dvl, f))


def train_hierarchical(dataset_dir: str, classes: list[str],
                       hierarchy: dict, out_dir: str, epochs: int = 50,
                       imgsz: int = 640, base: str = "yolo26n.pt") -> dict:
    """Train the parent model and the per-parent child models.

    Returns a manifest with the parent model path, the child model paths
    keyed by parent, and the groups. Persisted as out_dir/hierarchy.json.
    """
    groups = groups_from_hierarchy(classes, hierarchy)
    parents = list(groups.keys())
    cls_idx = {c: i for i, c in enumerate(classes)}
    parent_idx = {p: i for i, p in enumerate(parents)}
    manifest = {"groups": groups, "parents": parents,
                "parent_model": None, "child_models": {}}

    # Phase 1: coarse parent model -- every leaf relabelled to its phase-1
    # ancestor (the same key used to form the groups, so they always agree).
    leaf_parent = {c: p for p, leaves in groups.items() for c in leaves}
    keep = {cls_idx[c]: parent_idx[leaf_parent[c]] for c in classes}
    pdir = os.path.join(out_dir, "_parent_ds")
    if _relabel_dataset(dataset_dir, pdir, keep):
        manifest["parent_model"] = orchestrator.train_model(
            pdir, parents, base_weights=base, epochs=epochs, imgsz=imgsz,
            project=out_dir, name="parent")

    # Phase 2: one child model per parent group that has >1 leaf.
    for p, leaves in groups.items():
        if len(leaves) < 2:
            continue                       # nothing to disambiguate
        local = {c: i for i, c in enumerate(leaves)}
        keep = {cls_idx[c]: local[c] for c in leaves}
        cdir = os.path.join(out_dir, "_child_" + _safe(p) + "_ds")
        if _relabel_dataset(dataset_dir, cdir, keep):
            manifest["child_models"][p] = orchestrator.train_model(
                cdir, leaves, base_weights=base, epochs=epochs, imgsz=imgsz,
                project=out_dir,
                name="child_" + _safe(p))

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "hierarchy.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    return manifest
