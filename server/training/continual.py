"""Continual learning helpers.

Adding new classes over time risks catastrophic forgetting if you train
on only the new data. The simplest robust defence is rehearsal: mix a
sample of old data back in. These helpers assemble a combined dataset
from an old dataset and new data with a configurable replay fraction.
See Chapter 28.
"""
from __future__ import annotations
import os
import random
import shutil


def _list_images(images_dir: str) -> list[str]:
    if not os.path.isdir(images_dir):
        return []
    exts = (".jpg", ".jpeg", ".png", ".bmp")
    return [f for f in os.listdir(images_dir) if f.lower().endswith(exts)]


def build_replay_dataset(old_dir: str, new_dir: str, out_dir: str,
                         replay_fraction: float = 0.3,
                         split: str = "train") -> str:
    """Build a combined dataset: all of ``new_dir`` plus a random
    ``replay_fraction`` sample of ``old_dir``, copied into ``out_dir``.

    Each input is expected in YOLO layout (images/<split>, labels/<split>).
    The label file for each image is copied alongside it.
    """
    img_out = os.path.join(out_dir, "images", split)
    lbl_out = os.path.join(out_dir, "labels", split)
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(lbl_out, exist_ok=True)

    def _copy(src_root: str, names: list[str]) -> None:
        si = os.path.join(src_root, "images", split)
        sl = os.path.join(src_root, "labels", split)
        for n in names:
            stem = os.path.splitext(n)[0]
            shutil.copy2(os.path.join(si, n), os.path.join(img_out, n))
            lbl = stem + ".txt"
            src_lbl = os.path.join(sl, lbl)
            if os.path.exists(src_lbl):
                shutil.copy2(src_lbl, os.path.join(lbl_out, lbl))

    new_imgs = _list_images(os.path.join(new_dir, "images", split))
    _copy(new_dir, new_imgs)

    old_imgs = _list_images(os.path.join(old_dir, "images", split))
    k = int(len(old_imgs) * max(0.0, min(1.0, replay_fraction)))
    replay = random.sample(old_imgs, k) if k else []
    _copy(old_dir, replay)

    return out_dir
