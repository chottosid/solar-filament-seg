"""Compare this package with the original research code on real checkpoints.

Needs the research repository (../SOLAR_FILAMENT or $FS26_SOURCE); skipped
otherwise. About 10 minutes on CPU.
"""

import functools
import os
import sys
import types
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

SOURCE = Path(os.environ.get("FS26_SOURCE", REPO.parent / "SOLAR_FILAMENT"))
DET = SOURCE / "kernel_out/f0_det/runs/detect/runs/det/weights/best.pt"
REF = SOURCE / "kernel_out/f0_refiner/refiner_best.pt"
IMAGES = SOURCE / "MAGFiLO_1.0_Kaggle_2026/train/train_images"
STEMS = ("20151228024254Uh", "20160401170314Mh")  # fold-0 images with overlapping detections
SIZES = (1280, 1280, 1280, 1280, 1280, 1536, 1280)
MODES = ("nearest", "bilinear")


def _missing() -> str | None:
    coco = IMAGES.parent / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
    for path in (SOURCE / "src/anchor_ensemble.py", DET, REF, coco, *(IMAGES / f"{s}.jpeg" for s in STEMS)):
        if not path.exists():
            return f"research repo or checkpoint not found: {path}"
    try:
        import segmentation_models_pytorch  # noqa: F401
        import ultralytics  # noqa: F401
    except ImportError as e:
        return str(e)
    return None


def _skip_if_missing() -> None:
    reason = _missing()
    if reason:
        import pytest

        pytest.skip(reason)


@functools.cache
def _research() -> types.SimpleNamespace:
    """The research repository's modules (imported from its src/, without writing bytecode there)."""
    import importlib

    sys.dont_write_bytecode = True  # leave the research tree untouched
    sys.path.insert(0, str(SOURCE / "src"))
    names = ("anchor_ensemble", "ensemble_pool", "predict_refine", "refiner_data", "refiner_extent", "train_refiner",
             "yolo_data", "data")
    return types.SimpleNamespace(**{name: importlib.import_module(name) for name in names})


@functools.cache
def _models():
    """(old anchor module, old predict module, old refiner, new refiner, device), loaded once."""
    import torch

    from filament import inference

    old = _research()
    device = torch.device("cpu")
    old_ref = old.predict_refine.load_refiner(REF, device)  # downloads ImageNet weights, then overwrites them
    new_ref = inference.load_refiner(REF, device)
    return old.anchor_ensemble, old.predict_refine, old_ref, new_ref, device


def test_training_inputs_match():
    """Refiner samples (two epochs of jitter/flips), the loss and the YOLO labels are unchanged."""
    _skip_if_missing()
    import tempfile

    import torch

    from filament import detector, refiner
    from filament.data import fold_stems, load_records

    old = _research()
    coco = IMAGES.parent / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
    old_ais, new_recs, val = old.data.load_annotations(coco), load_records(coco), fold_stems(0)

    for augment, in_val in ((True, False), (False, True)):  # augmented train crops, plain val crops
        old_ds = old.refiner_data.RefinerDataset([a for a in old_ais if (a.stem in val) == in_val], IMAGES,
                                                 augment=augment)
        new_ds = refiner.RefinerDataset([r for r in new_recs if (r.stem in val) == in_val], IMAGES, augment=augment)
        assert len(old_ds) == len(new_ds) > 1000
        for epoch in (0, 3):
            old_ds.epoch = new_ds.epoch = epoch
            for idx in (0, 1, 777, len(new_ds) - 1):
                (x_old, y_old), (x_new, y_new) = old_ds[idx], new_ds[idx]
                assert torch.equal(x_old, x_new) and torch.equal(y_old, y_new), (augment, epoch, idx)

    for augment in (True, False):  # extent-robust samples: same draws (per-call counter) -> same tensors
        old_ds = old.refiner_extent.ExtentRobustDataset([a for a in old_ais if a.stem not in val], IMAGES, augment=augment)
        new_ds = refiner.ExtentRobustDataset([r for r in new_recs if r.stem not in val], IMAGES, augment=augment)
        for idx in (0, 1, 777, 777, len(new_ds) - 1):
            (x_old, y_old), (x_new, y_new) = old_ds[idx], new_ds[idx]
            assert torch.equal(x_old, x_new) and torch.equal(y_old, y_new), ("extent", augment, idx)

    gen = torch.Generator().manual_seed(0)
    logits = torch.randn(4, 1, 64, 64, generator=gen) * 3
    target = (torch.rand(4, 1, 64, 64, generator=gen) > 0.7).float()
    target[1] = 0  # an empty target exercises the positive-only Dice average
    assert torch.equal(old.train_refiner.refine_loss(logits, target), refiner.refine_loss(logits, target))

    with tempfile.TemporaryDirectory() as tmp:
        for per_record in (False, True):
            out_old, out_new = Path(tmp) / f"old_{per_record}", Path(tmp) / f"new_{per_record}"
            counts = old.yolo_data.export_yolo(old_ais, val, IMAGES, out_old, per_record=per_record)
            assert counts == detector.export_yolo(new_recs, val, IMAGES, out_new, per_record)
            for split in ("train", "val"):
                labels = [{p.name: p.read_text() for p in (out / "labels" / split).iterdir()}
                          for out in (out_old, out_new)]
                assert labels[0] == labels[1] and labels[0], (per_record, split)


def test_single_model_primitives_match():
    _skip_if_missing()
    import cv2
    import torch
    from ultralytics import YOLO

    from filament import inference
    from filament.data import load_image_gray

    old_anchor, old_pr, old_ref, new_ref, device = _models()
    old_sd, new_sd = old_ref.state_dict(), new_ref.state_dict()
    assert list(old_sd) == list(new_sd) and all(torch.equal(old_sd[k], new_sd[k]) for k in old_sd)

    img = load_image_gray(IMAGES, STEMS[0] + ".jpeg")
    bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    det_args = types.SimpleNamespace(conf=0.20, iou=0.50, imgsz=1280, max_det=300)
    old_boxes = old_pr.detect_boxes(YOLO(str(DET)), bgr, det_args)
    new_boxes = inference.detect_boxes(YOLO(str(DET)), bgr, 1280)
    assert len(old_boxes) == len(new_boxes) > 0
    for (c_old, b_old), (c_new, b_new) in zip(old_boxes, new_boxes):
        assert c_old == c_new and b_old.dtype == b_new.dtype and np.array_equal(b_old, b_new)

    box = old_boxes[0][1]
    p_old, w_old = old_pr.refine_prob(old_ref, img, box, 512, 8, device)
    p_new, w_new = inference.refine_prob(new_ref, img, box, device, 512, 8)
    assert w_old == w_new and p_old.dtype == p_new.dtype and np.array_equal(p_old, p_new)
    assert np.array_equal(old_pr.paste_full(p_old, w_old, *img.shape, 0.5),
                          inference.paste_full(p_new, w_new, img.shape, 0.5, "bilinear"))
    assert np.array_equal(old_anchor.paste_mask(p_old, w_old, img.shape, 0.5, "nearest"),
                          inference.paste_full(p_new, w_new, img.shape, 0.5, "nearest"))


def test_ensemble_rows_match():
    _skip_if_missing()
    from ultralytics import YOLO

    from filament import ensemble
    from filament.data import load_image_gray

    old_anchor, _, old_ref, new_ref, device = _models()
    old_args = types.SimpleNamespace(conf=0.20, iou=0.50, imgsz=1280, max_det=300, vote_iou=0.30, vote_conf=0.40,
                                     min_votes=5, crop=512, jitter=8, thr=0.50, min_area=200,
                                     post_paint_min_area=200, min_contrast=0.05)
    old_dets = [YOLO(str(DET)) for _ in SIZES]
    new_dets = [YOLO(str(DET)) for _ in SIZES]
    cfg = ensemble.EnsembleConfig()  # the CLI defaults must be the production configuration
    assert cfg.sizes == SIZES
    for stem in STEMS:
        img = load_image_gray(IMAGES, stem + ".jpeg")
        old = old_anchor.image_rows(old_dets, [old_ref] * 5, list(SIZES), stem, img, device, old_args, list(MODES))
        new = ensemble.image_rows(new_dets, [new_ref] * 5, stem, img, device, cfg, MODES)
        for mode in MODES:
            assert old[mode] == new[mode], (stem, mode)
            print(f"{stem} {mode}: {len(new[mode])} identical instances", flush=True)




def test_pmean_fusion_matches_research():
    """The final probability-mean fusion equals the research pool dumper's (fast: fake soft refiner)."""
    if not (SOURCE / "src/ensemble_pool.py").exists() or not (IMAGES / f"{STEMS[0]}.jpeg").exists():
        import pytest

        pytest.skip("research repo (src/ensemble_pool.py) or images not found")
    import importlib

    import torch

    from filament import ensemble
    from filament.data import load_image_gray

    _research()  # puts the research src/ on sys.path
    pool = importlib.import_module("ensemble_pool")

    class SoftRefiner(torch.nn.Module):
        def forward(self, x):  # dark pixels inside the seed: graded probabilities, so averaging matters
            dark = x[:, 0:1].mean(dim=(2, 3), keepdim=True) - x[:, 0:1]
            return torch.where(x[:, 2:3] > 0, 4.0 * (dark - 0.3), torch.full_like(dark, -10.0))

    img = load_image_gray(IMAGES, STEMS[0] + ".jpeg")
    box = np.array([900.0, 700.0, 1100.0, 780.0])
    refs = [SoftRefiner() for _ in range(5)]
    cfg = ensemble.EnsembleConfig()
    args = types.SimpleNamespace(crop=cfg.crop, jitter=cfg.jitter, thr=cfg.thr)
    old, _ = pool.seed_variants(refs, img, box, args, torch.device("cpu"))
    new = ensemble.seed_masks(refs, img, box, torch.device("cpu"), cfg, ("pmean", "nearest", "bilinear"))
    for mode in ("pmean", "nearest", "bilinear"):
        assert np.array_equal(old[mode], new[mode]), mode
    assert old["pmean"].any() and not np.array_equal(new["pmean"], new["nearest"])


if __name__ == "__main__":
    if _missing():
        print("SKIP:", _missing())
    else:
        for name, fn in sorted(globals().items()):
            if name.startswith("test_"):
                fn()
                print("ok", name)


def test_candidate_pool_matches_research():
    """filament.ensemble.image_pool == the research pool dumper, column for column (fake models, one real image)."""
    _skip_if_missing()
    import torch

    from filament.data import load_image_gray, load_records
    from filament.ensemble import EnsembleConfig, image_pool

    class Boxes:
        def __init__(self, items):
            self.xyxy = torch.tensor([b for _, b in items], dtype=torch.float32).reshape(-1, 4)
            self.conf = torch.tensor([c for c, _ in items], dtype=torch.float32)

        def __len__(self):
            return len(self.conf)

    class Det:  # fixed boxes, also on flipped inputs (the flip votes then land elsewhere)
        def __init__(self, items):
            self.items = items

        def predict(self, **kw):
            return [types.SimpleNamespace(boxes=Boxes([(c, b) for c, b in self.items if c >= kw["conf"]]))]

    class Refiner(torch.nn.Module):  # soft, image-dependent logits inside the seed
        def forward(self, x):
            dark = x[:, 0:1].mean(dim=(2, 3), keepdim=True) - x[:, 0:1]
            return torch.where(x[:, 2:3] > 0, 20.0 * (dark - 0.3), torch.full_like(dark, -10.0))

    old = _research()
    stem = STEMS[0]
    rec = next(r for r in load_records(IMAGES.parent / "MAGFiLO_1.0_Annotations_kaggle2026_train.json")
               if r.stem == stem and len(r.filaments) >= 3)
    gt = [[x, y, x + w, y + h] for x, y, w, h in sorted((f.bbox for f in rec.filaments), key=lambda b: -b[2] * b[3])[:3]]
    dets = [Det([(0.9 - 0.02 * v, gt[0])] + ([(0.45, gt[1])] if v < 5 else []) + ([(0.3, gt[2])] if v < 2 else []))
            for v in range(7)]
    refiners, device, img = [Refiner() for _ in range(5)], torch.device("cpu"), load_image_gray(IMAGES, stem + ".jpeg")
    for context in (1.5, 2.25):
        args = types.SimpleNamespace(conf=0.20, iou=0.50, vote_iou=0.30, max_det=300, thr=0.50, crop=512, jitter=8,
                                     context=context, extra=())
        expected = old.ensemble_pool.image_pool(dets, refiners, list(SIZES), stem, img, device, args)
        got = image_pool(dets, refiners, stem, img, device, EnsembleConfig(context=context))
        assert len(got) == len(expected) == 3, (len(got), len(expected))
        for g, e in zip(got, expected):
            assert g == {k: e[k] for k in g}, context

