"""Panoptic Quality exactly as the organisers compute it.

Ported from the organisers' self-evaluation notebook. Their PQ differs from
the textbook definition, and every quirk changes which predictions pay off:

* Pooled: TP IoUs, FP and FN are summed over the whole set, then
  PQ = sum(TP IoU) / (TP + FP/2 + FN/2). It is not a mean of per-image PQs.
* Per annotator record: ground truth is keyed by record ("010401-<stem>"),
  predictions by image ("<stem>"). An image with three records scores the same
  predictions three times, so a filament only one annotator drew costs a FN
  against the other two records, and a spurious prediction costs one FP per record.
* No one-to-one matching: every (GT, prediction) pair with IoU > 0.5 is a TP.
  FP = predictions that hit no GT, FN = GT hit by no prediction.
* Predictions on an image without any record count entirely as FP.

`pq_one_image` is the organisers' dense implementation and the reference;
`pq_one_image_rles` reproduces it on RLEs without building 2048^2 mask stacks.

    python -m filament.metric --pred oof_f0.csv --fold 0
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data import IMG_SIZE, fold_stems, gt_rows_for, load_records
from .submission import rles_to_layers

IOU_THRESHOLD = 0.5


def _as_float_layers(masks: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(masks, dtype=np.float32))


def get_overlap_matrices(gt_layers: torch.Tensor, pred_layers: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Pairwise IoU and Dice matrices, shape (n_gt, n_pred) (organisers' code)."""
    n_gt, height, width = gt_layers.shape
    n_pred = pred_layers.shape[0]

    gt_flat = gt_layers.reshape(n_gt, height * width)
    pred_flat = pred_layers.reshape(n_pred, height * width)

    intersection = torch.matmul(gt_flat, pred_flat.t())

    gt_areas = gt_flat.sum(dim=1).view(-1, 1)
    pred_areas = pred_flat.sum(dim=1).view(1, -1)

    union = gt_areas + pred_areas - intersection

    iou_matrix = torch.where(
        union == 0,
        torch.tensor(0.0, device=gt_layers.device),
        intersection / union,
    )
    dice_matrix = torch.where(
        union == 0,
        torch.tensor(0.0, device=gt_layers.device),
        2 * intersection / (gt_areas + pred_areas),
    )
    return iou_matrix, dice_matrix


@torch.no_grad()
def pq_one_image(gt_layers: torch.Tensor, pred_layers: torch.Tensor) -> tuple[list[float], int, int]:
    """TP IoU scores, FP count, FN count for one record (dense reference)."""
    n_gt, n_pred = gt_layers.shape[0], pred_layers.shape[0]
    if n_gt == 0:
        return [], n_pred, 0
    if n_pred == 0:
        return [], 0, n_gt

    iou_matrix, _ = get_overlap_matrices(gt_layers, pred_layers)
    hit_matrix = iou_matrix > IOU_THRESHOLD

    tp_iou_scores = iou_matrix[hit_matrix].tolist()
    fp_count = (hit_matrix.sum(dim=0) == 0).sum().item()
    fn_count = (hit_matrix.sum(dim=1) == 0).sum().item()
    return tp_iou_scores, fp_count, fn_count


def pq_one_image_rles(
    gt_rles: list[str], pred_rles: list[str], height: int = IMG_SIZE, width: int = IMG_SIZE
) -> tuple[list[float], int, int]:
    """Same hit-matrix accounting as pq_one_image, computed on RLEs.

    pycocotools' non-crowd IoU is intersection/union; casting to float32 before
    the strict > 0.5 test reproduces the dense float32 path, including pairs
    sitting exactly on the threshold.
    """
    from pycocotools import mask as mask_util

    if not gt_rles:
        return [], len(pred_rles), 0
    if not pred_rles:
        return [], 0, len(gt_rles)

    def packed(rles):
        return [{"size": [height, width], "counts": r if isinstance(r, bytes) else str(r).encode("utf-8")}
                for r in rles]

    iou = mask_util.iou(packed(gt_rles), packed(pred_rles), [0] * len(pred_rles)).astype(np.float32)
    hits = iou > IOU_THRESHOLD
    return iou[hits].tolist(), int((~hits.any(axis=0)).sum()), int((~hits.any(axis=1)).sum())


def pq_totals(tp_iou_scores: list[float], fp_count: int, fn_count: int) -> dict:
    """Pooled PQ = SQ x RQ from summed TP IoUs and FP/FN counts."""
    tp = len(tp_iou_scores)
    denom = tp + 0.5 * fp_count + 0.5 * fn_count
    pq = sum(tp_iou_scores) / denom if denom > 0 else 0.0
    return {
        "pq": pq,
        "tp": tp,
        "fp": fp_count,
        "fn": fn_count,
        "mean_iou_of_tps": float(np.mean(tp_iou_scores)) if tp else 0.0,
        "sq": float(np.mean(tp_iou_scores)) if tp else 0.0,  # segmentation quality
        "rq": tp / denom if denom > 0 else 0.0,  # recognition quality
    }


def pq_from_rles(gt_rows: list[tuple[str, str]], pred_rows: list[tuple[str, str]], *, backend: str = "rle") -> dict:
    """Score predictions against per-record ground truth, pooled over all records.

    gt_rows ids look like "010401-20140609195854Bh_3" (record id before "_");
    pred_rows ids look like "20140609195854Bh_3" (image stem before "_").
    """
    if backend not in {"rle", "dense"}:
        raise ValueError(f"unknown PQ backend: {backend}")

    gt_by_record = defaultdict(list)
    for fid, rle in gt_rows:
        gt_by_record[fid.split("_", 1)[0]].append(rle)

    pred_by_img = defaultdict(list)
    for fid, rle in pred_rows:
        pred_by_img[fid.split("_", 1)[0]].append(rle)

    all_tp: list[float] = []
    fp = fn = 0
    for record_id, gt_rles in gt_by_record.items():
        pred_rles = pred_by_img.get(record_id.split("-", 1)[1], [])
        if backend == "dense":
            tps, f, n = pq_one_image(
                _as_float_layers(rles_to_layers(gt_rles)),
                _as_float_layers(rles_to_layers(pred_rles)),
            )
        else:
            tps, f, n = pq_one_image_rles(gt_rles, pred_rles)
        all_tp.extend(tps)
        fp += f
        fn += n

    # predictions on images with no record at all are pure FP
    gt_image_ids = {rid.split("-", 1)[1] for rid in gt_by_record}
    for image_id, pred_rles in pred_by_img.items():
        if image_id not in gt_image_ids:
            fp += len(pred_rles)

    return pq_totals(all_tp, fp, fn)


def evaluate_fold(pred_csv: str | Path, fold: int, backend: str = "rle") -> dict:
    """Exact PQ of a prediction CSV on one CV fold's validation images.

    Predictions for images outside the fold are ignored (they would all be FP).
    """
    val_stems = fold_stems(fold)
    gt = gt_rows_for([rec for rec in load_records() if rec.stem in val_stems])
    pred = pd.read_csv(pred_csv)
    pred_rows = list(zip(pred["filament_id"], pred["segmentation_rle"].astype(str)))
    pred_rows = [r for r in pred_rows if r[0].split("_", 1)[0] in val_stems]
    return pq_from_rles(gt, pred_rows, backend=backend)


def main() -> None:
    ap = argparse.ArgumentParser(description="Score a prediction CSV on a CV fold with the organisers' exact PQ.",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--pred", required=True, help="prediction CSV (submission format)")
    ap.add_argument("--fold", type=int, default=0, help="CV fold whose validation images are scored")
    ap.add_argument("--backend", choices=["rle", "dense"], default="rle",
                    help="rle is fast; dense is the organisers' reference implementation")
    args = ap.parse_args()
    res = evaluate_fold(args.pred, args.fold, args.backend)
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items()})


if __name__ == "__main__":
    main()
