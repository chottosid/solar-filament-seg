"""The submitted pipeline: 7 detectors vote, 5 refiners draw, greedy painting.

Per test image:
  1. Each of 7 YOLO voters proposes boxes (conf >= 0.20) at its own image size.
  2. Boxes of different voters with box IoU > 0.30 are linked (union-find).
     A cluster survives if >= 5 voters have their own conf >= 0.40 in it.
     Agreement buys precision: the metric charges a spurious instance once
     per annotator record of the image, and looser votes that gained recall
     in cross-validation lost on the leaderboard.
  3. The highest-confidence box of a cluster seeds every refiner. Each of the
     5 fold refiners sees 3 views (identity, vertical flip, horizontal flip).
     Final ("pmean") mask: the 15 probability maps, bilinearly resized to the
     frame, are averaged and thresholded at 0.5. Nested cross-validation (the
     whole ensemble retrained without each outer fold) puts this +0.004 PQ
     above the binary majority ("nearest"/"bilinear": a pixel kept by a
     refiner if >= 2 of 3 views agree, by the ensemble if >= 3 of 5 refiners
     do) in every fold, with the same instances kept.
  4. Masks < 200 px are dropped, the rest painted greedily by seed confidence
     without overlap (the host rejects overlapping instances); painted
     fragments < 200 px and masks with ring contrast < 0.05 (not darker than
     their surroundings) are dropped.

    python -m filament.ensemble --weights weights --out submission.csv
    python -m filament.ensemble --weights weights --fold 0 --out oof_f0.csv   # leak-free fold check
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field, fields, replace
from pathlib import Path

import cv2
import numpy as np
import torch

from .data import TEST_IMAGES, TRAIN_IMAGES, data_root, fold_stems, load_image_gray
from .inference import detect_boxes, load_refiner, paste_full, refine_prob
from .submission import layers_to_rles, validate, write_submission

# Weights: 5 fold detectors (YOLO11s @1280), YOLO11s @1536 and YOLO11m @1280
# (both on the fold-0 split), then the 5 fold refiners. Order matters: the
# detector order matches EnsembleConfig.sizes.
DET_FILES = ("det_f0.pt", "det_f1.pt", "det_f2.pt", "det_f3.pt", "det_f4.pt", "det_s1536_f0.pt", "det_m1280_f0.pt")
REFINER_FILES = ("refiner_f0.pt", "refiner_f1.pt", "refiner_f2.pt", "refiner_f3.pt", "refiner_f4.pt")
WEIGHTS_URL = "https://example.com/REPLACE-ME/solar-filament-seg-weights.zip"  # placeholder: 12 files above


def _setting(default, doc: str):
    return field(default=default, metadata={"help": doc})


@dataclass(frozen=True)
class EnsembleConfig:
    """Inference settings; the defaults are the submitted (LB 0.40) configuration."""

    sizes: tuple[int, ...] = _setting((1280, 1280, 1280, 1280, 1280, 1536, 1280), "YOLO imgsz per detector")
    conf: float = _setting(0.20, "detection floor; low-confidence boxes still link clusters")
    iou: float = _setting(0.50, "YOLO NMS IoU")
    max_det: int = _setting(300, "YOLO max boxes per image")
    vote_iou: float = _setting(0.30, "box IoU that links boxes of different voters")
    vote_conf: float = _setting(0.40, "a voter supports a cluster only with its own conf >= this")
    min_votes: int = _setting(5, "supporting voters a cluster needs")
    crop: int = _setting(512, "refiner input side (px)")
    jitter: int = _setting(8, "seed-jitter TTA shift (px)")
    thr: float = _setting(0.50, "refiner probability threshold")
    min_area: int = _setting(200, "drop masks smaller than this before painting (px)")
    post_min_area: int = _setting(200, "drop painted fragments smaller than this (px)")
    min_contrast: float = _setting(0.05, "drop painted masks with lower ring contrast")


@dataclass
class Cluster:
    """Boxes of different voters that overlap: at most one (best) box per voter."""

    members: list[tuple[int, float, np.ndarray]] = field(default_factory=list)  # (voter, conf, xyxy)


def box_iou(a, b) -> float:
    """IoU of two xyxy boxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / max(ua, 1e-9)


def cluster_boxes(per_voter: list[list[tuple[float, np.ndarray]]], vote_iou: float) -> list[Cluster]:
    """Union-find over boxes of DIFFERENT voters linked at box IoU > vote_iou.

    Linking is transitive, so a long filament split into pieces by one voter
    still joins the cluster of another voter's single box. Boxes of the same
    voter are never linked directly (YOLO NMS already separated them). Each
    cluster keeps the highest-confidence box per voter.
    """
    items = [(vi, c, b) for vi, its in enumerate(per_voter) for c, b in its]
    parent = list(range(len(items)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(len(items)):
        vi, _, bi = items[i]
        for j in range(i + 1, len(items)):
            vj, _, bj = items[j]
            if vi == vj:
                continue
            if box_iou(bi, bj) > vote_iou:
                ra, rb = find(i), find(j)
                if ra != rb:
                    parent[ra] = rb

    groups: dict[int, list[int]] = {}
    for i in range(len(items)):
        groups.setdefault(find(i), []).append(i)

    clusters = []
    for idxs in groups.values():
        best_per_voter: dict[int, tuple[float, np.ndarray]] = {}
        for i in idxs:
            vi, c, b = items[i]
            if vi not in best_per_voter or c > best_per_voter[vi][0]:
                best_per_voter[vi] = (c, b)
        clusters.append(Cluster(members=[(vi, c, b) for vi, (c, b) in best_per_voter.items()]))
    return clusters


def seed_masks(refiners: list[torch.nn.Module], img: np.ndarray, box, device: torch.device,
               cfg: EnsembleConfig, modes: tuple[str, ...]) -> dict[str, np.ndarray]:
    """Full-frame mask for one seed box, per mode.

    "nearest"/"bilinear": binary majority over 3 views per refiner, then over
    refiners. "pmean": mean of all refiner x view probability maps > thr.
    The flipped views flip the whole image (and the box with it), so each view
    is a genuinely different crop context for the network.
    """
    if len(refiners) % 2 != 1:
        raise ValueError("refiner majority requires an odd number of models")
    h, w = img.shape
    x1, y1, x2, y2 = box
    views = [(img, box, None),
             (np.flip(img, 0).copy(), (x1, h - y2, x2, h - y1), 0),
             (np.flip(img, 1).copy(), (w - x2, y1, w - x1, y2), 1)]
    binary = tuple(mode for mode in modes if mode != "pmean")
    fold_counts = {mode: np.zeros(img.shape, np.uint8) for mode in binary}
    prob_sum = np.zeros(img.shape, np.float32)
    for refiner in refiners:
        flip_counts = {mode: np.zeros(img.shape, np.uint8) for mode in binary}
        for view, view_box, axis in views:
            prob, win = refine_prob(refiner, view, view_box, device, cfg.crop, cfg.jitter)
            if "pmean" in modes:
                x0, y0, x1, y1 = win
                full = np.zeros(img.shape, np.float32)
                full[y0:y1, x0:x1] = cv2.resize(prob, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
                prob_sum += full if axis is None else np.flip(full, axis)
            for mode in binary:
                mask = paste_full(prob, win, img.shape, cfg.thr, mode)
                if axis is not None:
                    mask = np.flip(mask, axis)
                flip_counts[mode] += mask
        for mode in binary:
            fold_counts[mode] += (flip_counts[mode] >= 2).astype(np.uint8)
    majority = len(refiners) // 2 + 1
    masks = {mode: (count >= majority).astype(np.uint8) for mode, count in fold_counts.items()}
    if "pmean" in modes:
        masks["pmean"] = (prob_sum / (len(views) * len(refiners)) > cfg.thr).astype(np.uint8)
    return {mode: masks[mode] for mode in modes}


def resolve_overlaps(scored: list[tuple[float, np.ndarray]], post_min_area: int = 0) -> list[np.ndarray]:
    """Paint masks greedily by descending score; each pixel goes to the first mask that claims it.

    Fragments smaller than post_min_area are dropped from the output but keep
    their pixels, so removing a sliver never hands its pixels to a weaker mask.
    """
    out = []
    occupied = np.zeros_like(scored[0][1], bool) if scored else None
    for _, m in sorted(scored, key=lambda t: -t[0]):
        m = m.copy()
        m[occupied] = 0
        if m.sum() == 0:
            continue
        occupied |= m.astype(bool)
        out.append(m)
    return [m for m in out if m.sum() >= post_min_area]


def ring_contrast(m: np.ndarray, img: np.ndarray, pad: int = 24) -> float:
    """(ring mean - mask mean) / ring mean, ring = mask bbox +-pad minus the mask.

    Filaments are darker than the surrounding chromosphere; masks that are not
    (plage edges, limb artefacts, specks) score ~0 or below.
    """
    ys, xs = np.nonzero(m)
    if len(ys) == 0:
        return 0.0
    y0, y1 = max(0, ys.min() - pad), min(m.shape[0], ys.max() + pad)
    x0, x1 = max(0, xs.min() - pad), min(m.shape[1], xs.max() + pad)
    ring = ~m[y0:y1, x0:x1].astype(bool)  # bool first: ~ on uint8 is bitwise-not, not negation
    if not ring.any():
        return 0.0
    rm = float(img[y0:y1, x0:x1][ring].mean())
    return (rm - float(img[m.astype(bool)].mean())) / max(rm, 1.0)


def image_rows(dets: list, refiners: list[torch.nn.Module], stem: str, img: np.ndarray, device: torch.device,
               cfg: EnsembleConfig = EnsembleConfig(),
               modes: tuple[str, ...] = ("pmean",)) -> dict[str, list[tuple[str, str]]]:
    """Submission rows ("<stem>_<k>", rle) for one image, per paste mode.

    Several paste modes share every network pass, so comparing them costs no
    extra GPU time.
    """
    if len(dets) != len(cfg.sizes):
        raise ValueError("one detector image size is required per voter")
    bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    per_voter = [detect_boxes(det, bgr, size, cfg.conf, cfg.iou, cfg.max_det) for det, size in zip(dets, cfg.sizes)]
    scored: dict[str, list[tuple[float, np.ndarray]]] = {mode: [] for mode in modes}
    for cluster in cluster_boxes(per_voter, cfg.vote_iou):
        support = sum(conf >= cfg.vote_conf for _, conf, _ in cluster.members)
        if support < cfg.min_votes:
            continue
        _, seed_conf, seed_box = max(cluster.members, key=lambda member: member[1])
        for mode, mask in seed_masks(refiners, img, seed_box, device, cfg, modes).items():
            if int(mask.sum()) >= cfg.min_area:
                scored[mode].append((seed_conf, mask))
    rows = {}
    for mode in modes:
        painted = resolve_overlaps(scored[mode], cfg.post_min_area)
        layers = [mask for mask in painted if ring_contrast(mask, img) >= cfg.min_contrast]
        rows[mode] = [(f"{stem}_{k}", rle) for k, rle in enumerate(layers_to_rles(layers), 1)]
    return rows


def predict(dets: list, refiners: list[torch.nn.Module], image_dir: Path, stems: list[str], device: torch.device,
            cfg: EnsembleConfig = EnsembleConfig(), paste: str = "pmean") -> list[tuple[str, str]]:
    """Run image_rows over a list of image stems."""
    rows: list[tuple[str, str]] = []
    start = time.time()
    for i, stem in enumerate(stems, 1):
        rows.extend(image_rows(dets, refiners, stem, load_image_gray(image_dir, stem + ".jpeg"), device, cfg,
                               (paste,))[paste])
        if i % 10 == 0 or i == len(stems):
            print(f"{i}/{len(stems)} images, {len(rows)} instances, {time.time() - start:.0f}s", flush=True)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Predict a submission CSV with the 7-detector / 5-refiner ensemble.",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--weights", type=Path, required=True, help=f"dir with {', '.join(DET_FILES + REFINER_FILES)}")
    ap.add_argument("--out", type=Path, default=Path("submission.csv"), help="output CSV")
    ap.add_argument("--fold", type=int, default=None,
                    help="instead of the test set, predict fold K's validation images with fold K's own "
                         "detector + refiner only (min votes 1): the leak-free CV check")
    ap.add_argument("--paste", choices=["pmean", "nearest", "bilinear"], default="pmean",
                    help="mask fusion: pmean = mean of 15 probability maps (final); nearest = the Oct-6 LB-0.40 file")
    for f in fields(EnsembleConfig):
        parse = (lambda s: tuple(int(v) for v in s.split(","))) if f.name == "sizes" else type(f.default)
        ap.add_argument("--" + f.name.replace("_", "-"), type=parse, default=f.default, help=f.metadata["help"])
    args = ap.parse_args()
    cfg = EnsembleConfig(**{f.name: getattr(args, f.name) for f in fields(EnsembleConfig)})

    if args.fold is None:
        det_files, ref_files = DET_FILES, REFINER_FILES
        image_dir = data_root() / TEST_IMAGES
        stems = sorted(p.stem for p in image_dir.glob("*.jpeg"))
    else:
        det_files, ref_files = (f"det_f{args.fold}.pt",), (f"refiner_f{args.fold}.pt",)
        cfg = replace(cfg, sizes=(1280,), min_votes=1)
        image_dir = data_root() / TRAIN_IMAGES
        stems = sorted(fold_stems(args.fold))
    missing = [name for name in det_files + ref_files if not (args.weights / name).exists()]
    if missing:
        ap.error(f"missing weights in {args.weights}: {missing} (download: {WEIGHTS_URL})")
    if not stems:
        ap.error(f"no input images found in {image_dir}")
    if len(cfg.sizes) != len(det_files):
        ap.error(f"--sizes needs one image size per detector ({len(det_files)})")
    if not 1 <= cfg.min_votes <= len(det_files) or cfg.conf > cfg.vote_conf:
        ap.error("invalid vote count or detection floor above the vote confidence")
    print(cfg, flush=True)

    from ultralytics import YOLO

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dets = [YOLO(str(args.weights / name)) for name in det_files]
    refiners = [load_refiner(args.weights / name, device) for name in ref_files]
    rows = predict(dets, refiners, image_dir, stems, device, cfg, args.paste)
    if not rows:
        raise SystemExit("no instances predicted")
    write_submission(rows, args.out)
    print(f"wrote {args.out}: {len(rows)} rows over {len(stems)} images")
    print(validate(args.out, image_dir if args.fold is None else None))


if __name__ == "__main__":
    main()
