"""Single-model primitives: detector boxes, refiner probability maps, paste-back.

The ensemble (ensemble.py) is built only from these four calls, so each one
behaves exactly as it did when the submitted predictions were produced.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

from .refiner import CONTEXT, PAD_RATIO, REFINE_CROP, Window, build_refiner, refiner_input, seed_crop, square_bounds


def detect_boxes(det, bgr: np.ndarray, imgsz: int, conf: float = 0.20, iou: float = 0.50,
                 max_det: int = 300) -> list[tuple[float, np.ndarray]]:
    """(confidence, xyxy float32 box) pairs from one YOLO model, highest confidence first.

    The 2048 px frame is letterboxed to imgsz inside YOLO; the boxes come back
    in full-resolution pixels, which is where the refiner crops.
    """
    res = det.predict(source=bgr, conf=conf, iou=iou, imgsz=imgsz, verbose=False, augment=False, max_det=max_det,
                      device=0 if torch.cuda.is_available() else "cpu")[0]
    if res.boxes is None or not len(res.boxes):
        return []
    xyxy = res.boxes.xyxy.cpu().numpy()
    confs = res.boxes.conf.cpu().numpy()
    return [(float(confs[i]), xyxy[i]) for i in np.argsort(-confs)]


def load_refiner(ckpt: str | Path, device: torch.device) -> torch.nn.Module:
    """Refiner U-Net in eval mode with weights from a state-dict checkpoint."""
    model = build_refiner(pretrained=False)
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    return model.to(device).eval()


@torch.no_grad()
def refine_prob(refiner: torch.nn.Module, img: np.ndarray, box_xyxy, device: torch.device,
                crop: int = REFINE_CROP, jitter_px: int = 8, context: float = CONTEXT) -> tuple[np.ndarray, Window]:
    """Filament probability inside the box's crop window, at crop resolution.

    Seed-jitter TTA: the seed box is shifted by +-jitter_px along x and y and
    the five sigmoid maps are averaged, so the mask does not hinge on the exact
    detector box edges. context = window side / box long side (1.5 for the
    submitted refiners, INFER_CONTEXT = 2.25 for extent-robust ones).
    Returns (prob of shape (crop, crop), window).
    """
    h, w = img.shape
    x1, y1, x2, y2 = box_xyxy
    bbox = [float(x1), float(y1), max(1.0, float(x2 - x1)), max(1.0, float(y2 - y1))]
    win = square_bounds(bbox, h, w, context=context)
    x0, y0, x1b, y1b = win
    img_c = cv2.resize(img[y0:y1b, x0:x1b], (crop, crop), interpolation=cv2.INTER_LINEAR)
    shifts = [(0, 0)] if jitter_px <= 0 else [(0, 0), (jitter_px, 0), (-jitter_px, 0), (0, jitter_px), (0, -jitter_px)]
    batch = [refiner_input(img_c, seed_crop([bbox[0] + dx, bbox[1] + dy, bbox[2], bbox[3]], PAD_RATIO, win, crop))
             for dx, dy in shifts]
    x = torch.from_numpy(np.stack(batch)).to(device)
    with torch.autocast("cuda", enabled=device.type == "cuda"):
        logits = refiner(x)
    return torch.sigmoid(logits.float()).mean(dim=0)[0].cpu().numpy(), win


def paste_full(prob: np.ndarray, win: Window, shape: tuple[int, int], thr: float,
               mode: str = "nearest") -> np.ndarray:
    """Threshold a crop probability map and paste it into a full-frame uint8 mask.

    "nearest": threshold at crop resolution, nearest-resize the binary mask to
    the window (used by the submitted LB-0.40 predictions).
    "bilinear": resize the probability to the window, then threshold; smoother
    on small windows, kept as the documented alternative.
    """
    x0, y0, x1, y1 = win
    full = np.zeros(shape, np.uint8)
    if mode == "bilinear":
        prob_win = cv2.resize(prob, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
        full[y0:y1, x0:x1] = (prob_win > thr).astype(np.uint8)
    elif mode == "nearest":
        full[y0:y1, x0:x1] = cv2.resize(
            (prob > thr).astype(np.uint8), (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST,
        )
    else:
        raise ValueError(f"unknown paste mode: {mode}")
    return full
