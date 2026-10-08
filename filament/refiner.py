"""Stage 2: the box-seeded crop refiner (U-Net, ResNet-34 encoder).

A detector box says *where* a filament is; the refiner says *which pixels*.
For every box, a square window (1.5x the box, at least 96 px) is cut from the
full-resolution image and resized to 512 x 512 (median window ~140 px, so a
4 px ribbon is ~15 px wide at the network input). The input has three channels:

  [raw image, CLAHE(image), seed]

CLAHE lifts faint filaments out of limb darkening. The seed is a filled box
padded by 40 % per side: the same representation at training time (a
jittered ground-truth box) and at test time (a detector box). The seed is
never the ground-truth mask, which would leak the answer.

Training uses one sample per (annotator record, filament), so annotator
disagreement acts as label noise the model averages over.

    python -m filament.refiner --fold 0 --out weights/refiner_f0.pt
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
    the jittered one. Randomness is a pure function of (index, epoch), so every
    epoch draws fresh jitter and flips yet runs are repeatable sample by sample.
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


def train_refiner(fold: int, out: Path, epochs: int = 12, batch: int = 8) -> float:
    """Train on the other four folds, keep the epoch with the best fold crop Dice.

    AdamW (lr 3e-4, wd 1e-4), cosine schedule, mixed precision on CUDA.
    Writes the best state dict to `out` and returns its crop Dice.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    images_dir = data_root() / TRAIN_IMAGES
    records = load_records()
    val_stems = fold_stems(fold)
    train_ds = RefinerDataset([r for r in records if r.stem not in val_stems], images_dir, augment=True)
    val_ds = RefinerDataset([r for r in records if r.stem in val_stems], images_dir, augment=False)
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
    args = ap.parse_args()
    train_refiner(args.fold, args.out, args.epochs, args.batch)


if __name__ == "__main__":
    main()
