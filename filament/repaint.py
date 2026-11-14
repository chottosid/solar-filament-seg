"""Redraw a candidate pool's masks with another refiner set, keeping the clusters.

Detector outputs can change slightly between machines, so the detectors are
run once and only the masks are recomputed.

    python -m filament.repaint --weights weights --pool test_pool.csv.gz --out test_pool_extent.csv.gz
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data import TEST_IMAGES, TRAIN_IMAGES, data_root, load_image_gray
from .ensemble import EXTENT_REFINER_FILES, REFINER_FILES, EnsembleConfig, ring_contrast, seed_masks
from .inference import load_refiner
from .submission import layers_to_rles
from .weights import ensure

MODES = ("nearest", "bilinear", "pmean")


def repaint_pool(pool: pd.DataFrame, refiners: list[torch.nn.Module], load_image: Callable[[str], np.ndarray],
                 device: torch.device, cfg: EnsembleConfig) -> pd.DataFrame:
    """Copy of pool whose refiner-derived columns come from `refiners` (seed boxes unchanged)."""
    out = pool.copy()
    img, current = None, None
    for i, row in pool.iterrows():
        if row.stem != current:
            current, img = row.stem, load_image(row.stem)
        agree: list[float] = []
        masks = seed_masks(refiners, img, (row.x1, row.y1, row.x2, row.y2), device, cfg, MODES, agree)
        out.at[i, "refiner_agree"] = agree[0]
        for mode, mask in masks.items():
            out.at[i, f"area_{mode}"] = int(mask.sum())
            out.at[i, f"contrast_{mode}"] = ring_contrast(mask, img) if mask.any() else 0.0
            out.at[i, f"rle_{mode}"] = layers_to_rles([mask])[0]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Repaint a candidate pool with another refiner set.",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--pool", type=Path, required=True, help="pool written by python -m filament.ensemble --pool")
    ap.add_argument("--out", type=Path, required=True, help="repainted pool (.csv.gz)")
    ap.add_argument("--refiner-set", choices=["production", "extent"], default="extent")
    ap.add_argument("--split", choices=["test", "train"], default="test", help="which image folder the pool's stems are in")
    args = ap.parse_args()
    names = EXTENT_REFINER_FILES if args.refiner_set == "extent" else REFINER_FILES
    ensure(args.weights, names)
    cfg = EnsembleConfig()
    if args.refiner_set == "extent":
        cfg = replace(cfg, context=2.25)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    refiners = [load_refiner(args.weights / n, device) for n in names]
    image_dir = data_root() / (TEST_IMAGES if args.split == "test" else TRAIN_IMAGES)
    pool = pd.read_csv(args.pool)
    painted = repaint_pool(pool, refiners, lambda stem: load_image_gray(image_dir, stem + ".jpeg"), device, cfg)
    painted.to_csv(args.out, index=False)
    print(f"wrote {args.out}: {len(painted)} candidates repainted with {len(refiners)} {args.refiner_set} refiners")


if __name__ == "__main__":
    main()
