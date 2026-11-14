"""U-Net refiner (ResNet-34 encoder) that segments a filament inside a crop around its box.

The crop is 1.5x the box (at least 96 px), resized to 512 x 512. Input channels:
image, CLAHE(image), filled box. Training boxes are jittered ground-truth boxes.

--extent-robust trains on boxes that are too short or too long, in a crop
centred on the box, so the network does not copy the box's length.

    python -m filament.refiner --fold 0 --out weights/refiner_f0.pt
    python -m filament.refiner --fold 0 --extent-robust --epochs 48 --out weights/refiner_extent_f0.pt
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .data import TRAIN_IMAGES, Filament, Record, data_root, fold_stems, load_image_gray, load_records

REFINE_CROP = 512  # network input side
CONTEXT = 1.5  # window side = CONTEXT x longer box side ...
MIN_CROP = 96  # ... but never smaller than this (keeps background context)
PAD_RATIO = 0.40  # seed box inflation per side

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)[:, None, None]
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)[:, None, None]

Window = tuple[int, int, int, int]  # x0, y0, x1, y1 in full-image pixels


def square_bounds(bbox: list[float], h: int, w: int, context: float = CONTEXT, minimum: int = MIN_CROP) -> Window:
    """Square crop window around an xywh box, shifted (not shrunk) to stay inside the image."""
    x, y, bw, bh = bbox
    cx, cy = x + bw / 2, y + bh / 2
    side = max(int(math.ceil(max(bw, bh) * context)), minimum)
    x0 = int(round(cx - side / 2))
    y0 = int(round(cy - side / 2))
    x1, y1 = x0 + side, y0 + side
    if x0 < 0:
        x1 -= x0
        x0 = 0
    if y0 < 0:
        y1 -= y0
        y0 = 0
    if x1 > w:
        x0 -= x1 - w
        x1 = w
    if y1 > h:
        y0 -= y1 - h
        y1 = h
    return max(0, x0), max(0, y0), x1, y1


def padded_box(bbox: list[float], pad_ratio: float, h: int, w: int) -> Window:
    """Seed rectangle: the xywh box inflated by pad_ratio per side, clamped to h x w."""
    x, y, bw, bh = bbox
    px, py = bw * pad_ratio, bh * pad_ratio
    return max(0, int(x - px)), max(0, int(y - py)), min(w, int(x + bw + px)), min(h, int(y + bh + py))


def seed_crop(bbox: list[float], pad_ratio: float, win: Window, crop: int) -> np.ndarray:
    """Filled padded seed box drawn directly at crop scale (box shifted into window coordinates)."""
    x0, y0, x1, y1 = win
    local = [bbox[0] - x0, bbox[1] - y0, bbox[2], bbox[3]]
    sx0, sy0, sx1, sy1 = padded_box(local, pad_ratio, y1 - y0, x1 - x0)
    s = crop / (x1 - x0)
    seed = np.zeros((crop, crop), np.uint8)
    seed[max(0, int(sy0 * s)):int(sy1 * s), max(0, int(sx0 * s)):int(sx1 * s)] = 1
    return seed


def jitter_bbox(bbox: list[float], h: int, w: int, rng: random.Random) -> tuple[list[float], float]:
    """Perturb a ground-truth box (shift +-15 %, scale 0.85-1.25, pad 0.2-0.6) to mimic a detector box."""
    x, y, bw, bh = map(float, bbox)
    cx = x + bw / 2 + rng.uniform(-0.15, 0.15) * bw
    cy = y + bh / 2 + rng.uniform(-0.15, 0.15) * bh
    bw = max(8.0, bw * rng.uniform(0.85, 1.25))
    bh = max(8.0, bh * rng.uniform(0.85, 1.25))
    x = float(np.clip(cx - bw / 2, 0, w - 2))
    y = float(np.clip(cy - bh / 2, 0, h - 2))
    bw = float(min(bw, w - x))
    bh = float(min(bh, h - y))
    pad = rng.uniform(PAD_RATIO * 0.5, PAD_RATIO * 1.5)
    return [x, y, bw, bh], pad


def rasterize_in_window(poly: list[float], x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
    """Rasterize a polygon only inside the crop window (no 2048^2 mask per sample)."""
    from pycocotools import mask as mask_util

    shifted = [c - (x0 if i % 2 == 0 else y0) for i, c in enumerate(poly)]
    rle = mask_util.frPyObjects([shifted], y1 - y0, x1 - x0)[0]
    return mask_util.decode(rle)


def refiner_input(img_c: np.ndarray, seed_c: np.ndarray) -> np.ndarray:
    """(3, H, W) float32 network input [image, CLAHE, seed], ImageNet-normalised.

    CLAHE is computed on the crop, identically at training and inference.
    """
    clahe_c = cv2.createCLAHE(2.0, (8, 8)).apply(img_c)
    x = np.stack([img_c, clahe_c, seed_c * 255], axis=0).astype(np.float32) / 255.0
    return (x - _IMAGENET_MEAN) / _IMAGENET_STD


class RefinerDataset(Dataset):
    """One sample per (annotator record, filament): jittered seed box -> that record's polygon.

    The crop window comes from the un-jittered ground-truth box, the seed from
    the jittered one. Randomness is a pure function of (index, epoch), but the
    training DataLoader's persistent workers keep the copy of the dataset they
    started with (epoch 0): in the submitted runs every sample kept one jitter
    and flip draw for all epochs. ExtentRobustDataset redraws every epoch.
    """

    def __init__(self, records: list[Record], images_dir: Path, crop: int = REFINE_CROP, augment: bool = True):
        self.items: list[tuple[Record, Filament]] = [(rec, f) for rec in records for f in rec.filaments]
        self.images_dir = images_dir
        self.crop = crop
        self.augment = augment
        self.epoch = 0  # set by the training loop

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        rng = random.Random(idx * 1_000_003 + 7 + self.epoch * 1_000_000_007)
        rec, f = self.items[idx]
        img = load_image_gray(self.images_dir, rec.file_name)
        h, w = img.shape

        if self.augment:
            seed_bbox, pad = jitter_bbox(f.bbox, h, w, rng)
        else:
            seed_bbox, pad = f.bbox, PAD_RATIO
        win = square_bounds(f.bbox, h, w)
        x0, y0, x1, y1 = win

        img_c = cv2.resize(img[y0:y1, x0:x1], (self.crop, self.crop), interpolation=cv2.INTER_LINEAR)
        tgt_c = cv2.resize(
            rasterize_in_window(f.polygon, x0, y0, x1, y1), (self.crop, self.crop), interpolation=cv2.INTER_NEAREST
        ).astype(np.float32)
        seed_c = seed_crop(seed_bbox, pad, win, self.crop)

        if self.augment and rng.random() < 0.5:
            img_c, seed_c, tgt_c = np.fliplr(img_c), np.fliplr(seed_c), np.fliplr(tgt_c)
        if self.augment and rng.random() < 0.5:
            img_c, seed_c, tgt_c = np.flipud(img_c), np.flipud(seed_c), np.flipud(tgt_c)
        x = refiner_input(np.ascontiguousarray(img_c), np.ascontiguousarray(seed_c))
        y = torch.from_numpy(np.ascontiguousarray(tgt_c).astype(np.float32))[None]
        return torch.from_numpy(x), y


SEED_MIX = (0.50, 0.25, 0.25)  # extent-robust seeds: production jitter, short, long
SHORT_COV, LONG_COV = (0.40, 0.85), (1.20, 2.00)  # fraction of the spine's arc length a seed covers
CTX_RANGE = (1.5, 3.0)  # training window side / seed long side, log-uniform
INFER_CONTEXT = 2.25  # inference window for extent-robust refiners
MIN_PERTURB_LEN = 40.0  # px of spine; shorter filaments always get the production jitter


def axis_points(f: Filament, step: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Spine resampled every `step` px as (xy (n, 2), arc length s (n,)); polygon major axis if no spine."""
    p = np.asarray(f.spine, float).reshape(-1, 2) if len(f.spine) >= 4 else np.zeros((0, 2))
    if len(p):
        p = p[np.r_[True, np.linalg.norm(np.diff(p, axis=0), axis=1) > 1e-6]]
    if len(p) < 2:
        q = np.asarray(f.polygon, float).reshape(-1, 2)
        c = q.mean(0)
        u = np.linalg.eigh(np.cov((q - c).T))[1][:, -1]
        t = (q - c) @ u
        p = np.stack([c + t.min() * u, c + t.max() * u])
    sn = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    s = np.arange(0.0, sn[-1] + 1e-9, step) if sn[-1] > 0 else np.zeros(1)
    return np.c_[np.interp(s, sn, p[:, 0]), np.interp(s, sn, p[:, 1])], s


def interval_box(f: Filament, lo: float, hi: float, h: int, w: int) -> list[float]:
    """xywh box around the part of the filament whose arc length lies in [lo, hi].

    [lo, hi] may exceed [0, L]: the spine is then extended straight along its end tangents.
    """
    from scipy.spatial import cKDTree

    xy, s = axis_points(f)
    L = float(s[-1])
    half = float(np.clip(f.area / max(L, 1.0), 2.0, 64.0)) / 2
    q = np.asarray(f.polygon, float).reshape(-1, 2)
    edges = np.diff(np.vstack([q, q[:1]]), axis=0)
    n = np.maximum(1, np.ceil(np.linalg.norm(edges, axis=1)).astype(int))
    boundary = np.vstack([q[i] + edges[i] * np.arange(k)[:, None] / k for i, k in enumerate(n)])
    sb = s[cKDTree(xy).query(boundary)[1]]
    pts = [boundary[(sb >= lo) & (sb <= hi)], xy[(s >= lo) & (s <= hi)]]
    k = min(len(xy) - 1, 8)
    if lo < 0:
        d = xy[0] - xy[k]
        pts.append(xy[0] + d / max(np.linalg.norm(d), 1e-9) * np.arange(0.0, -lo + 1)[:, None])
    if hi > L:
        d = xy[-1] - xy[-1 - k]
        pts.append(xy[-1] + d / max(np.linalg.norm(d), 1e-9) * np.arange(0.0, hi - L + 1)[:, None])
    axis_pts = np.vstack(pts[1:]) if len(pts) > 1 else np.zeros((0, 2))
    allp = np.vstack([pts[0], axis_pts - half, axis_pts + half])
    x0, y0 = np.clip(allp.min(0), 0, [w - 2, h - 2])
    x1, y1 = np.clip(allp.max(0), [x0 + 2, y0 + 2], [w, h])
    return [float(x0), float(y0), float(x1 - x0), float(y1 - y0)]


def mild_jitter(bbox: list[float], h: int, w: int, rng: random.Random, frac: float = 0.05) -> list[float]:
    """+-5 % shift and scale of a short/long seed box."""
    x, y, bw, bh = bbox
    cx = x + bw / 2 + rng.uniform(-frac, frac) * bw
    cy = y + bh / 2 + rng.uniform(-frac, frac) * bh
    bw, bh = max(8.0, bw * rng.uniform(1 - frac, 1 + frac)), max(8.0, bh * rng.uniform(1 - frac, 1 + frac))
    x, y = float(np.clip(cx - bw / 2, 0, w - 2)), float(np.clip(cy - bh / 2, 0, h - 2))
    return [x, y, float(min(bw, w - x)), float(min(bh, h - y))]


def sample_seed_window(f: Filament, h: int, w: int, rng: random.Random) -> dict:
    """Seed box (xywh), pad and window for one extent-robust sample.

    50 % production jitter; 25 % short: a box around a stretch covering 0.4-0.85
    of the spine (half centred, half anchored at a random end); 25 % long: the
    spine stretched to 1.2-2.0x along its end tangents. The window is centred
    on the seed, side = c x seed long side with c log-uniform in [1.5, 3].
    """
    L = float(axis_points(f)[1][-1])
    u = rng.random()
    kind = "in" if L < MIN_PERTURB_LEN or u < SEED_MIX[0] else ("short" if u < SEED_MIX[0] + SEED_MIX[1] else "long")
    if kind == "in":
        bbox, pad = jitter_bbox(f.bbox, h, w, rng)
    else:
        if kind == "short":
            length = rng.uniform(*SHORT_COV) * L
            lo = (L - length) / 2 if rng.random() < 0.5 else rng.choice([0.0, L - length])
            hi = lo + length
        else:
            extra = (rng.uniform(*LONG_COV) - 1.0) * L
            a = rng.random()
            lo, hi = -a * extra, L + (1 - a) * extra
        bbox = mild_jitter(interval_box(f, lo, hi, h, w), h, w, rng)
        pad = rng.uniform(PAD_RATIO * 0.5, PAD_RATIO * 1.5)
    ctx = math.exp(rng.uniform(math.log(CTX_RANGE[0]), math.log(CTX_RANGE[1])))
    return {"kind": kind, "bbox": bbox, "pad": pad, "ctx": ctx, "win": square_bounds(bbox, h, w, context=ctx)}


class ExtentRobustDataset(RefinerDataset):
    """RefinerDataset with extent-perturbed seeds, seed-centred random-context windows and soft
    whole-filament targets (INTER_AREA-resized). augment=False is deterministic per index (same
    seed mix, no flips), so the validation crop Dice tracks extent robustness."""

    def __init__(self, records: list[Record], images_dir: Path, crop: int = REFINE_CROP, augment: bool = True):
        super().__init__(records, images_dir, crop=crop, augment=augment)
        self._draw = 0

    def sample(self, idx: int, rng: random.Random) -> tuple[torch.Tensor, torch.Tensor, dict]:
        rec, f = self.items[idx]
        img = load_image_gray(self.images_dir, rec.file_name)
        h, w = img.shape
        meta = sample_seed_window(f, h, w, rng)
        x0, y0, x1, y1 = meta["win"]
        img_c = cv2.resize(img[y0:y1, x0:x1], (self.crop, self.crop), interpolation=cv2.INTER_LINEAR)
        tgt = rasterize_in_window(f.polygon, x0, y0, x1, y1).astype(np.float32)
        tgt_c = cv2.resize(tgt, (self.crop, self.crop), interpolation=cv2.INTER_AREA)
        seed_c = seed_crop(meta["bbox"], meta["pad"], meta["win"], self.crop)
        if self.augment and rng.random() < 0.5:
            img_c, seed_c, tgt_c = np.fliplr(img_c), np.fliplr(seed_c), np.fliplr(tgt_c)
        if self.augment and rng.random() < 0.5:
            img_c, seed_c, tgt_c = np.flipud(img_c), np.flipud(seed_c), np.flipud(tgt_c)
        x = refiner_input(np.ascontiguousarray(img_c), np.ascontiguousarray(seed_c))
        y = torch.from_numpy(np.ascontiguousarray(tgt_c).astype(np.float32))[None]
        return torch.from_numpy(x), y, meta

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.augment:  # persistent workers never see `epoch` updates: count draws per process instead
            self._draw += 1
            rng = random.Random(f"{idx}/{self.epoch}/{self._draw}")
        else:
            rng = random.Random(idx * 1_000_003 + 7)
        x, y, _ = self.sample(idx, rng)
        return x, y


def build_refiner(pretrained: bool = True) -> torch.nn.Module:
    """U-Net with a ResNet-34 encoder, 3 input channels, 1 logit channel.

    pretrained=True starts the encoder from ImageNet (training); inference
    loads a full state dict, so it skips the ImageNet download.
    """
    import segmentation_models_pytorch as smp

    return smp.Unet(encoder_name="resnet34", encoder_weights="imagenet" if pretrained else None,
                    in_channels=3, classes=1)


def refine_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """0.45 BCE + 0.55 soft Dice, Dice averaged over samples with a non-empty target.

    No positive weighting: in a box-centred crop the filament is a large share
    of the pixels. Dice counters the thin-structure bias of plain BCE (losing a
    1 px rim costs little BCE but a lot of IoU on a 4 px ribbon).
    """
    bce = F.binary_cross_entropy_with_logits(logits, target)
    prob = torch.sigmoid(logits)
    inter = (prob * target).sum(dim=(2, 3))
    pos = (target.sum(dim=(2, 3)) > 0).float()
    n_pos = pos.sum().clamp(min=1.0)
    den = prob.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice_i = 1 - (2 * inter + 1) / (den + 1)
    dice = (dice_i * pos).sum() / n_pos
    return 0.45 * bce + 0.55 * dice


@torch.no_grad()
def crop_dice(model: torch.nn.Module, ds: RefinerDataset, device: torch.device, limit: int = 512) -> float:
    """Mean Dice at threshold 0.5 over ~limit evenly spaced validation crops.

    Only a checkpoint-selection signal: compare models with fold PQ, not this.
    """
    model.eval()
    n = len(ds.items)
    dices = []
    for i in range(0, n, max(1, n // limit)):
        x, y = ds[i]
        with torch.autocast("cuda", enabled=device.type == "cuda"):
            logit = model(x[None].to(device))
        pred = (torch.sigmoid(logit)[0, 0].float() > 0.5).cpu().numpy()
        tgt = y[0].numpy() > 0.5
        if not tgt.any():
            continue
        inter = (pred & tgt).sum()
        den = pred.sum() + tgt.sum()
        dices.append(2 * inter / max(den, 1))
    model.train()
    return float(np.mean(dices)) if dices else 0.0


def train_refiner(fold: int, out: Path, epochs: int = 12, batch: int = 8, extent_robust: bool = False) -> float:
    """Train on the other four folds, keep the epoch with the best fold crop Dice.

    AdamW (lr 3e-4, wd 1e-4), cosine schedule, mixed precision on CUDA.
    Writes the best state dict to `out` and returns its crop Dice.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    images_dir = data_root() / TRAIN_IMAGES
    records = load_records()
    val_stems = fold_stems(fold)
    ds_cls = ExtentRobustDataset if extent_robust else RefinerDataset
    train_ds = ds_cls([r for r in records if r.stem not in val_stems], images_dir, augment=True)
    val_ds = ds_cls([r for r in records if r.stem in val_stems], images_dir, augment=False)
    dl = DataLoader(
        train_ds,
        batch_size=batch,
        shuffle=True,
        num_workers=2,
        pin_memory=device.type == "cuda",
        drop_last=True,
        persistent_workers=True,
    )
    print(f"fold {fold}: {len(train_ds)} train crops / {len(val_ds)} val crops")

    model = build_refiner(pretrained=True).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    best = -1.0
    for epoch in range(epochs):
        train_ds.epoch = epoch
        losses = []
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            with torch.autocast("cuda", enabled=device.type == "cuda"):
                loss = refine_loss(model(x), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            losses.append(loss.item())
        sched.step()
        dice = crop_dice(model, val_ds, device)
        print(f"epoch {epoch}: loss {np.mean(losses):.4f} val crop-dice {dice:.4f}", flush=True)
        if dice > best:
            best = dice
            torch.save(model.state_dict(), out)
    print(f"best val crop-dice {best:.4f}; checkpoint at {out}")
    return best


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the box-seed refiner on all folds except --fold.",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--fold", type=int, required=True, help="held-out fold (training uses the other four)")
    ap.add_argument("--out", type=Path, required=True, help="best checkpoint path, e.g. weights/refiner_f0.pt")
    ap.add_argument("--epochs", type=int, default=12, help="training epochs (cosine schedule)")
    ap.add_argument("--batch", type=int, default=8, help="batch size")
    ap.add_argument("--extent-robust", action="store_true", help="short/long spine seeds, random-context windows")
    args = ap.parse_args()
    train_refiner(args.fold, args.out, args.epochs, args.batch, args.extent_robust)


if __name__ == "__main__":
    main()
