"""Submission format and the pre-submit validator.

One CSV row per predicted filament: filament_id = "<image stem>_<k>",
segmentation_rle = COCO RLE counts string of a 2048 x 2048 mask. The host
rejects a file whose instances overlap within an image, so every submission
written by this package is validated before it is reported as done.

    python -m filament.submission submission.csv [--test]
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from .data import IMG_SIZE, TEST_IMAGES, data_root

COLUMNS = ["filament_id", "segmentation_rle"]


def layers_to_rles(layers: list[np.ndarray] | np.ndarray) -> list[str]:
    """Encode binary (H, W) masks as COCO RLE counts strings."""
    from pycocotools import mask as mask_util

    rles = []
    for layer in layers:
        rle = mask_util.encode(np.asfortranarray(layer.astype(np.uint8)))
        rles.append(rle["counts"].decode("utf-8"))
    return rles


def rles_to_layers(rles: list[str], height: int = IMG_SIZE, width: int = IMG_SIZE) -> np.ndarray:
    """Decode COCO RLE counts strings into an (n, H, W) float32 mask stack.

    Counts strings carry no size; like the organisers' scorer, the frame size is
    supplied explicitly.
    """
    from pycocotools import mask as mask_util

    if not rles:
        return np.zeros((0, height, width), dtype=np.float32)
    rle_dicts = []
    for rle in rles:
        counts = rle if isinstance(rle, bytes) else str(rle).encode("utf-8")
        rle_dicts.append({"size": [height, width], "counts": counts})
    masks = mask_util.decode(rle_dicts)
    return masks.transpose(2, 0, 1).astype(np.float32)


def write_submission(rows: list[tuple[str, str]], path: str | Path) -> None:
    """Write (filament_id, rle) rows as a submission CSV."""
    pd.DataFrame(rows, columns=COLUMNS).to_csv(path, index=False)


def validate(csv_path: str | Path, test_dir: Path | None = None) -> str:
    """Raise ValueError unless the CSV is a well-formed, overlap-free submission.

    With test_dir, also require every image stem to be a test image. Returns a
    one-line summary.
    """
    from pycocotools import mask as mask_util

    df = pd.read_csv(csv_path)
    if list(df.columns) != COLUMNS:
        raise ValueError(f"columns must be {COLUMNS}, got {list(df.columns)}")
    if len(df) == 0:
        raise ValueError("empty submission")
    if not df.filament_id.is_unique:
        raise ValueError("duplicate filament_id")
    if df.segmentation_rle.isna().any():
        raise ValueError("NaN rle")
    if df.segmentation_rle.astype(str).str.len().eq(0).any():
        raise ValueError("empty rle")

    by_img: dict[str, list[str]] = defaultdict(list)
    for fid, rle in zip(df.filament_id, df.segmentation_rle.astype(str)):
        by_img[fid.rsplit("_", 1)[0]].append(rle)

    for stem, rles in by_img.items():
        occupied = np.zeros((IMG_SIZE, IMG_SIZE), bool)
        for rle in rles:
            mask = mask_util.decode({"size": [IMG_SIZE, IMG_SIZE], "counts": rle.encode()}).astype(bool)
            if not mask.any():
                raise ValueError(f"{stem}: empty instance mask")
            overlap_px = int((occupied & mask).sum())
            if overlap_px:
                raise ValueError(f"{stem}: {overlap_px} overlapping pixels across instances")
            occupied |= mask
    summary = f"OK: {len(df)} rows / {len(by_img)} images, no overlaps"

    if test_dir is not None:
        test = {p.stem for p in Path(test_dir).glob("*.jpeg")}
        if not test:
            raise ValueError(f"no test images found under {test_dir}")
        unknown = set(by_img) - test
        if unknown:
            raise ValueError(f"stems not in test set: {sorted(unknown)[:5]}")
        summary += f"; all stems are test images ({len(test) - len(by_img)} test images without predictions)"
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate a submission CSV (format + no overlaps).")
    ap.add_argument("csv")
    ap.add_argument("--test", action="store_true", help="also check stems against the test image set")
    args = ap.parse_args()
    print(validate(args.csv, data_root() / TEST_IMAGES if args.test else None))


if __name__ == "__main__":
    main()
