"""Source training images from Google Open Images at scale.

Uses FiftyOne to download a subset for chosen classes and export it as
a YOLO dataset the trainer (Chapter 27) consumes -- so you are not
drawing every box by hand. See Chapter 26.
"""
from __future__ import annotations
import os


def download_subset(classes: list[str], max_samples: int = 200,
                    split: str = "train",
                    export_dir: str = "data/datasets/oi_subset") -> str:
    """Download up to ``max_samples`` Open Images samples containing the
    given classes and export them in YOLO format.

    Requires the optional dependency ``fiftyone``. Returns the export
    directory, which contains images/, labels/, and a dataset.yaml.
    """
    import fiftyone as fo
    import fiftyone.zoo as foz

    dataset = foz.load_zoo_dataset(
        "open-images-v7",
        split=split,
        label_types=["detections"],
        classes=classes,
        max_samples=max_samples,
        only_matching=True,        # keep just the requested classes
    )
    os.makedirs(export_dir, exist_ok=True)
    dataset.export(
        export_dir=export_dir,
        dataset_type=fo.types.YOLOv5Dataset,
        label_field="ground_truth",
        classes=classes,
    )
    return export_dir


def merge_class_lists(studio_classes: list[str],
                      oi_classes: list[str]) -> list[str]:
    """Union of studio-defined and Open Images classes, order-stable."""
    out = list(studio_classes)
    for c in oi_classes:
        if c not in out:
            out.append(c)
    return out
