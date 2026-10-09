# Solar filament segmentation (Kaggle Solar Filament Segmentation Challenge 2026)

Instance segmentation of solar filaments in GONG H-alpha full-disk images.
A two-stage pipeline: YOLO11 detectors propose boxes, a box-seeded U-Net
draws each filament's pixels, and a 7-detector / 5-refiner vote turns them
into non-overlapping instances. **Public leaderboard: 0.40 PQ.**

## The problem

Each test image is a 2048 x 2048 H-alpha frame of the full solar disk.
Filaments are dark, elongated absorption features in the chromosphere: thin
ribbons only 3-12 px wide (median area about 1,200 px), often faint against
limb darkening. Training data is MAGFiLO 1.0 (Kaggle 2026 release): 707
images, 1,154 annotator records, 8,199 filament polygons; 296 images were
labelled independently by two or three annotators. The test set has 180
images. A submission lists one COCO RLE mask per predicted filament.

## The metric and its quirks

The organisers' Panoptic Quality (ported from their self-evaluation notebook
to `filament/metric.py`, with a fast RLE backend that reproduces it exactly)
has properties that shape every design decision:

| Quirk | Consequence |
|---|---|
| **Pooled**: TP IoUs, FP and FN are summed over the whole test set, then PQ = sum(TP IoU) / (TP + FP/2 + FN/2) | Images with many filaments dominate; no per-image averaging. |
| **Scored against each annotator record**: an image with three records scores the same predictions three times | A spurious instance costs one FP *per record*. A filament only one of three annotators drew earns one TP but two FPs if predicted, and one FN if not: a candidate pays only when its annotator-averaged matched IoU exceeds PQ/2 (about 0.2). |
| **TP = IoU > 0.5, all pairs**: no one-to-one matching; FP/FN come from the rows and columns of the hit matrix | Mask quality on 3-12 px ribbons decides TPs: a 1 px boundary shift can move IoU across 0.5. |
| **Non-overlap requirement**: overlapping instances make the host reject the file | Masks are painted greedily into a single label map, and every CSV is validated (`filament/submission.py`). |

## Pipeline

```
 2048 x 2048 H-alpha frame
     |
     |--> YOLO11s @1280, folds 0-4 ---------+
     |--> YOLO11s @1536 (fold-0 split) -----+--> boxes with conf >= 0.20 (7 voters)
     |--> YOLO11m @1280 (fold-0 split) -----+
                                            |
          union-find: link boxes of different voters with box IoU > 0.30
          keep a cluster if >= 5 voters have their own conf >= 0.40
                                            |  seed = the cluster's highest-confidence box
                                            v
     5 fold refiners (U-Net, ResNet-34) x 3 views (identity, vertical flip, horizontal flip)
       input 512 x 512 crop of [image, CLAHE(image), filled padded seed box]
       seed-jitter TTA: 5 seeds shifted by +-8 px, probabilities averaged, threshold 0.5
       the 15 probability maps (5 refiners x 3 views) are averaged; pixel kept if mean > 0.5
                                            |
          drop masks < 200 px -> paint greedily by seed confidence, no overlap
          -> drop painted fragments < 200 px -> drop masks with ring contrast < 0.05
                                            v
                         submission.csv (filament_id, COCO RLE)
```

Why two stages: a box is an easy target for a filament whose exact outline
annotators disagree on, and refining a box crop at roughly 3-5x magnification
(median window ~140 px, resized to 512) keeps 3-12 px ribbons intact. The
reasoning behind each setting is in the module docstrings.

## Results

| Model | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 | All 707 images |
|---|---|---|---|---|---|---|
| Single model per fold (80 % data), out-of-fold PQ | 0.4151 | 0.4152 | 0.4058 | 0.4024 | 0.3924 | 0.406 (mean) |
| 7 / 5 ensemble, binary-majority masks, nested CV | 0.4361 | 0.4409 | 0.4331 | 0.4496 | 0.4305 | 0.4379 |
| **7 / 5 ensemble, probability-mean masks (default), nested CV** | **0.4401** | **0.4448** | **0.4361** | **0.4541** | **0.4356** | **0.4421** |

Public leaderboard (about 50 % of the test set): **0.40** for both ensemble variants;
single models 0.36-0.37. Private leaderboard: *to be added when the competition closes (15 Nov 2026).*

* **Nested cross-validation.** A 5-fold ensemble cannot be scored on its own folds
  (four of its five members saw every training image). For each outer fold we
  retrained the complete ensemble (7 detectors, 5 refiners) on day-grouped inner
  folds of the other 80 % of images (64 % of the data per model, 60 models) and
  ran it on the held-out 20 %. Probability-mean fusion beats the binary majority
  by +0.0041 PQ (95 % CI +0.0033 to +0.0049, paired observing-day bootstrap) in
  every fold, with the same instances kept.
* Matched filaments agree with each annotator better than annotators agree with
  each other: mean IoU 0.690 vs 0.634, Dice 0.814 vs 0.773. 1.5 % of filaments are
  split across two or more of our instances (human vs human: 2.4 %); 3.2 % of our
  instances cover two or more filaments of one record (human: 3.1 %).
* Folds group images by observing day (`data/cv_folds.json`), so near-identical
  frames of one day never straddle train and validation.

## Reproduction

Everything runs from this repository plus the competition data. Training needs
nothing else (ultralytics and segmentation-models-pytorch fetch the COCO /
ImageNet starting weights themselves); inference needs only the 12 trained
weight files (see Model weights) or the output of the training step.

### 1. Environment

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
```

See the comments in `requirements.txt` for Kaggle's Python 3.13 images.

### 2. Data

Put the competition data at `data/MAGFiLO_1.0_Kaggle_2026/` (containing
`train/` and `test/`) or point `MAGFILO_DIR` at it:

```bash
export MAGFILO_DIR=/kaggle/input/competitions/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026
python -m filament.data --check     # the committed fold split regenerates exactly
```

### 3. Training (12 models)

```bash
for f in 0 1 2 3 4; do
  python -m filament.detector --fold $f --out weights/det_f$f.pt              # YOLO11s @1280
done
python -m filament.detector --fold 0 --imgsz 1536 --out weights/det_s1536_f0.pt
python -m filament.detector --fold 0 --model yolo11m.pt --batch 6 --out weights/det_m1280_f0.pt
for f in 0 1 2 3 4; do
  python -m filament.refiner --fold $f --out weights/refiner_f$f.pt
done
```

Detectors: 40 epochs, batch 8 (YOLO11m: 6), seed 42, patience 10,
ultralytics defaults otherwise, labels = union of all annotator records'
boxes. Refiners: 12 epochs, batch 8, AdamW 3e-4, cosine schedule, 0.45 BCE +
0.55 soft Dice, one sample per (annotator record, filament).

Runtime on a Kaggle P100: about 35 min per YOLO11s @1280 detector and 45 min
per refiner (measured); the @1536 and YOLO11m variants take longer (roughly
1-1.5 h each, estimated). Roughly 9-10 P100 GPU hours in total.

### 4. Inference

```bash
python -m filament.ensemble --weights weights --out submission.csv
python -m filament.submission submission.csv --test
```

About 25 minutes for the 180 test images on a Kaggle T4 (measured, including
model loading). CPU works but takes hours. All settings are CLI flags whose
defaults are the submitted configuration (`python -m filament.ensemble -h`);
`--paste bilinear` switches to the alternative paste-back.

### 5. Leak-free fold check

```bash
python -m filament.ensemble --weights weights --fold 0 --out oof_f0.csv   # fold 0's own models only
python -m filament.metric --pred oof_f0.csv --fold 0
```

### 6. Tests

```bash
python -m pytest -q            # or: python tests/test_metric.py, etc.
```

`tests/test_equivalence.py` replays this package against the original research
code and asserts identical training inputs (refiner samples, loss, YOLO labels)
and identical inference outputs on real checkpoints; it is skipped when that
code is not present.

### Notebook

`notebooks/pipeline.ipynb` walks through the whole pipeline: data and
annotator overlays, the fold split, training (guarded by `RUN_TRAINING`),
weights, test inference, validation, a fold-0 PQ evaluation and overlays of
predictions.

## Model weights

`WEIGHTS_URL` in `filament/ensemble.py` (placeholder:
`https://example.com/REPLACE-ME/solar-filament-seg-weights.zip`) points to an
archive of the 12 files below; unpack it into `weights/`.

| File | Model | Trained on | Inference size |
|---|---|---|---|
| `det_f0.pt` ... `det_f4.pt` | YOLO11s detector | all folds except k | 1280 |
| `det_s1536_f0.pt` | YOLO11s detector | all folds except 0 | 1536 |
| `det_m1280_f0.pt` | YOLO11m detector | all folds except 0 | 1280 |
| `refiner_f0.pt` ... `refiner_f4.pt` | U-Net / ResNet-34 refiner (state dict) | all folds except k | 512 crop |

## Exactness notes

* The code was ported from the research repository with every retained
  computation unchanged; `tests/test_equivalence.py` checks refiner training
  samples, the loss, YOLO label files, detector boxes, probability maps,
  paste-back and the full ensemble rows (both paste modes) against it.
* The submitted CSV applied the post-paint 200 px rule as a filter on the
  written file, which kept the original instance numbering (6 ids missing).
  The CLI applies the same rule inside painting and numbers instances
  contiguously; the masks are identical, and the metric ignores the number.
* GPU inference runs the refiners in fp16 autocast, so a different GPU or
  library version can flip rare boundary pixels. Training is not bit-exact on
  GPU (and the refiner trainer is unseeded, as in the original runs); the
  released weights reproduce the submission.

## Honesty statement

* Training used only the competition's training images and annotations, plus
  COCO-pretrained YOLO11 and ImageNet-pretrained ResNet-34 starting weights.
* No external annotations, no test-set labels, no overlap with the public
  MAGFiLO test data, no pseudo-labels.
* Inference uses only the test H-alpha images: no neighbouring frames,
  magnetograms or metadata.
* Model and post-processing settings were chosen by cross-validation on
  day-grouped folds, ensemble-level settings by nested cross-validation (see
  Results). The vote operating point (5 of 7 voters at conf 0.40) is the one the
  public leaderboard preferred over both looser and stricter rules.

## Repository layout

```
filament/
  data.py        COCO records, images, day-grouped folds
  metric.py      the organisers' PQ (dense reference + fast RLE backend), fold evaluation CLI
  detector.py    YOLO export (union / per-record labels) and detector training CLI
  refiner.py     box-seed crop dataset, U-Net, loss and refiner training CLI
  inference.py   detector boxes, refiner probability maps, paste-back
  ensemble.py    voting, majority masks, painting, gates and the prediction CLI
  submission.py  RLE helpers and the submission validator CLI
tests/           assert-based tests (pytest or plain python)
notebooks/       pipeline.ipynb
data/            cv_folds.json (+ the competition data, not committed)
```

## License

MIT, see `LICENSE`.
