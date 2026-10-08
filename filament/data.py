"""MAGFiLO annotations, images and the day-grouped cross-validation split.

Ground truth is a COCO-style JSON:
  images[]       id = "<record>-<stem>" (e.g. "010401-20140609195854Bh"),
                 file_name = "<stem>.jpeg"
  annotations[]  image_id -> that id; one polygon per filament plus bbox/area.

A *record* is one annotator's complete labelling of one physical image. 296 of
the 707 training images carry two or three independent records, and the metric
scores predictions against every record separately, so the record (not the
image) is the unit of ground truth throughout this package.

    python -m filament.data           # regenerate data/cv_folds.json
    python -m filament.data --check   # verify the committed file
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

IMG_SIZE = 2048  # every GONG frame in the challenge is 2048 x 2048
ROOT = Path(__file__).resolve().parent.parent
FOLDS_PATH = ROOT / "data" / "cv_folds.json"
TRAIN_JSON = "train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"
TRAIN_IMAGES = "train/train_images"
TEST_IMAGES = "test/test_images"


@dataclass
class Filament:
    """One annotated filament: a thin dark ribbon, typically 3-12 px wide."""

    polygon: list[float]  # flat [x0, y0, x1, y1, ...] in pixels
    bbox: list[float]  # COCO [x, y, w, h]
    area: float
    category_id: int  # chirality: 1 left, 2 right, 3 unidentifiable, 4 ambiguous


@dataclass
class Record:
    """One annotator's labelling of one physical image."""

    record_id: str  # "010401-20140609195854Bh"
    stem: str  # "20140609195854Bh"
    file_name: str  # "20140609195854Bh.jpeg"
    filaments: list[Filament] = field(default_factory=list)

    @property
    def day(self) -> str:
        """Observation day (YYYYMMDD): consecutive frames of one day look alike."""
        return self.stem[:8]


def data_root() -> Path:
    """Competition data directory: $MAGFILO_DIR, else data/MAGFiLO_1.0_Kaggle_2026."""
    env = os.environ.get("MAGFILO_DIR")
    return Path(env) if env else ROOT / "data" / "MAGFiLO_1.0_Kaggle_2026"


def load_records(json_path: str | Path | None = None) -> list[Record]:
    """Parse the COCO JSON into records, in file order (default: training JSON)."""
    with open(json_path or data_root() / TRAIN_JSON) as f:
        coco = json.load(f)
    by_id = {
        i["id"]: Record(record_id=i["id"], stem=i["file_name"].rsplit(".", 1)[0], file_name=i["file_name"])
        for i in coco["images"]
    }
    for a in coco["annotations"]:
        by_id[a["image_id"]].filaments.append(
            Filament(polygon=a["segmentation"][0], bbox=a["bbox"], area=a["area"], category_id=a["category_id"])
        )
    return list(by_id.values())


def load_image_gray(images_dir: Path, file_name: str) -> np.ndarray:
    """Read an H-alpha JPEG as uint8 grayscale at native 2048 px resolution.

    Never downsample: filaments are a few pixels wide and vanish at half size.
    """
    img = cv2.imread(str(Path(images_dir) / file_name), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"missing image {file_name} in {images_dir}")
    return img


def gt_rows_for(records: list[Record]) -> list[tuple[str, str]]:
    """Ground truth in the scorer's format: ("<record_id>_<k>", COCO RLE counts).

    The record id prefix is what makes the metric score each annotator separately.
    """
    from pycocotools import mask as mask_util

    rows: list[tuple[str, str]] = []
    for rec in records:
        for k, f in enumerate(rec.filaments, start=1):
            rle = mask_util.frPyObjects([f.polygon], IMG_SIZE, IMG_SIZE)[0]
            rows.append((f"{rec.record_id}_{k}", rle["counts"].decode("utf-8")))
    return rows


def split_stems(records: list[Record], n_folds: int = 5, seed: int = 0) -> list[set[str]]:
    """Deal observing days into n_folds image-balanced folds.

    Grouping by day (not by image, not by record) keeps all records of one image
    and all near-duplicate frames of one day in the same fold; anything finer
    leaks identical pixels between train and validation. Days are shuffled,
    then each goes to the fold with the fewest images so far.
    """
    stem_day = {rec.stem: rec.day for rec in records}
    groups: dict[str, list[str]] = defaultdict(list)
    for s in sorted(stem_day):
        groups[stem_day[s]].append(s)

    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    folds: list[set[str]] = [set() for _ in range(n_folds)]
    counts = [0] * n_folds
    for k in keys:
        i = counts.index(min(counts))
        folds[i].update(groups[k])
        counts[i] += len(groups[k])
    return folds


def fold_stems(fold: int, path: Path = FOLDS_PATH) -> set[str]:
    """Validation image stems of one fold, from the committed data/cv_folds.json."""
    return set(json.loads(Path(path).read_text())["folds"][fold])


def main() -> None:
    ap = argparse.ArgumentParser(description="Regenerate (or --check) data/cv_folds.json.")
    ap.add_argument("--check", action="store_true", help="only verify the committed file")
    args = ap.parse_args()
    folds = split_stems(load_records(), n_folds=5, seed=0)
    text = json.dumps({"folds": [sorted(f) for f in folds]}, indent=0)
    if args.check:
        if FOLDS_PATH.read_text() != text:
            raise SystemExit(f"{FOLDS_PATH} differs from the regenerated split")
        print(f"{FOLDS_PATH} matches the regenerated split")
    else:
        FOLDS_PATH.write_text(text)
        print(f"wrote {FOLDS_PATH}")
    print("images per fold:", [len(f) for f in folds])


if __name__ == "__main__":
    main()
