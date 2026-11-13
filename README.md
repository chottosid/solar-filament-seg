# Solar filament segmentation (Kaggle Solar Filament Segmentation Challenge 2026)

Instance segmentation of solar filaments in GONG H-alpha full-disk images.
Seven YOLO11 detectors vote on candidate filaments; box-seeded U-Nets draw
each candidate's pixels; a gradient-boosting model decides which candidates
to keep at the vote rule's instance count; extent-robust U-Nets draw the
final masks, which are painted without overlap. **Public leaderboard: 0.41 PQ**
(`results/submission_x48.csv`, Kaggle submission 58306582).

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
     |--> YOLO11s @1536 (fold-0 split) -----+--> boxes with conf >= 0.20 (7 voters; the same
     |--> YOLO11m @1280 (fold-0 split) -----+    voters also run on two flipped copies)
                                            |
          union-find: link boxes of different voters with box IoU > 0.30
          every cluster with >= 2 voters is a candidate; seed = its highest-confidence box
                                            |
          +---------------------------------+-------------------------------+
          v                                                                 v
   5 fold refiners, window 1.5 x seed                 5 extent-robust fold refiners, window 2.25 x seed
   x 3 views (identity, v-flip, h-flip):              x 3 views: 15 probability maps averaged,
   15 probability maps averaged, kept > 0.5           kept > 0.5 -> the submitted masks
          |                                                                 |
          v                                                                 |
   count-matched swap: gradient boosting predicts each candidate's          |
   annotator-averaged matched IoU u from 88 features (votes, mask           |
   shape, pixels, context); keep the top N by u, N = the number of          |
   instances the 5-of-7 vote rule (5 voters at conf >= 0.40) keeps          |
          +----------------------------------+------------------------------+
                                             v
          paint the kept candidates' extent-robust masks in u order without overlap;
          drop masks < 200 px before painting, fragments < 200 px after it, and
          masks whose ring contrast is < 0.05
                                             v
                            submission.csv (filament_id, COCO RLE)
```

Why two stages: a box is an easy target for a filament whose exact outline
annotators disagree on, and refining a box crop at roughly 3-5x magnification
(median window ~140 px, resized to 512) keeps 3-12 px ribbons intact.

Why a selection model at a fixed count: the metric charges a spurious
instance once per annotator record, and every variant that predicted more
instances (looser votes, a probability threshold on u) gained in
cross-validation but lost on the leaderboard. The swap keeps the vote's
count and changes only *which* candidates are kept (`filament/selection.py`).

Why extent-robust refiners: the production refiner copies its seed box (mask
length follows box length with slope 0.8-0.9), so a detector box that covers
half a filament yields half a mask. The extent-robust refiner is trained on
deliberately short and long seeds inside wider, seed-centred windows with the
whole filament as target (`--extent-robust` in `filament/refiner.py`); it
both extends short masks and trims long ones. The reasoning behind every
setting is in the module docstrings.

## Results

| Model | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 | All 707 images |
|---|---|---|---|---|---|---|
| Single model per fold (80 % data), out-of-fold PQ | 0.4151 | 0.4152 | 0.4058 | 0.4024 | 0.3924 | 0.406 (mean) |
| 7 / 5 vote rule, binary-majority masks, nested CV | 0.4361 | 0.4409 | 0.4331 | 0.4496 | 0.4305 | 0.4379 |
| 7 / 5 vote rule, probability-mean masks, nested CV | 0.4401 | 0.4448 | 0.4361 | 0.4541 | 0.4356 | 0.4421 |
| + count-matched swap (same instance count), nested CV | 0.4539 | 0.4540 | 0.4443 | 0.4560 | 0.4364 | 0.4488 |
| **+ extent-robust masks (final pipeline), nested CV** | **0.4541** | **0.4539** | **0.4490** | **0.4588** | **0.4393** | **0.4509** |

Public leaderboard (about 50 % of the test set), final pipelines by refiner
training length: 12 epochs 0.40, 24 epochs 0.41 (submission 57106989), **48 epochs
0.41 (submission 58306582, the submitted file)**; every other ensemble row 0.40 or
lower; single models 0.36-0.37. Two-decimal scores on about 90 images cannot resolve
differences below 0.01, which is why every choice above was made by nested CV.
Private leaderboard: *to be added when the competition closes (15 Nov 2026).*

* **Nested cross-validation.** A 5-fold ensemble cannot be scored on its own folds
  (four of its five members saw every training image). For each outer fold we
  retrained the complete ensemble (7 detectors, 5 refiners) on day-grouped inner
  folds of the other 80 % of images (64 % of the data per model, 60 models) and
  ran it on the held-out 20 %. Probability-mean fusion beats the binary majority
  by +0.0041 PQ (95 % CI +0.0033 to +0.0049, paired observing-day bootstrap) in
  every fold, with the same instances kept. The count-matched swap adds +0.0067
  (+0.0029 to +0.0104), and painting its choice with extent-robust masks
  (25 more nested refiners) brings the total to +0.0088 (+0.0048 to +0.0127)
  over the vote rule, positive in every fold. The nested pipeline rows above used
  12-epoch refiners. The submitted refiners train longer, and that step is
  measured one refiner at a time on identical nested seed boxes (the
  single-refiner gate of `analysis`, not a rerun of the whole nested pipeline):
  24 instead of 12 epochs gained +0.0020 PQ (+0.0008 to +0.0032), positive in all
  five folds, and 48 instead of 24 epochs gained another +0.0020 (+0.0007 to
  +0.0031; per fold +0.0021, +0.0028, +0.0014, +0.0019, +0.0016). Earlier in the
  project the single-refiner gate overstated the ensemble effect of the extent-robust
  recipe (+0.0047 for one refiner against +0.0026 in the nested ensemble), so the
  gain of the five-refiner average is probably smaller than +0.0020; on the public
  board it is invisible at two decimals.
* **Things we tried after the pipeline was fixed, and dropped.** A ResNeXt-50
  encoder for the refiner gained +0.0009 PQ (95 % interval -0.0002 to +0.0021)
  over the 24-epoch ResNet-34, positive in four of five folds and zero in the
  fifth, below our adoption rule. Averaging the 24-epoch and 48-epoch refiners
  (ten maps instead of five) scored 0.0008 below the 48-epoch refiners alone on
  the three folds we ran. A different family (Mask2Former proposals refined by
  SAM 2) scored 0.38 on the public board, and four label-free consensus
  submissions that combined it with this pipeline scored 0.39-0.40, below the
  0.41 of the pipeline alone, although a small reused validation set had predicted
  gains. None of these is in the released code.
* Matched filaments agree with each annotator better than annotators agree with
  each other: mean IoU 0.693 vs 0.634, Dice 0.816 vs 0.773. 1.2 % of filaments are
  split across two or more of our instances (human vs human: 2.4 %); 2.6 % of our
  instances cover two or more filaments of one record (human: 3.1 %). Median
  ribbon width (2 area / perimeter) 9.8 px vs 8.3 px for annotators.
* Folds group images by observing day (`data/cv_folds.json`), so near-identical
  frames of one day never straddle train and validation.

## Reproduction

Everything runs from this repository plus the competition data. Training needs
nothing else (ultralytics and segmentation-models-pytorch fetch the COCO /
ImageNet starting weights themselves); inference needs only the 17 trained
weight files (see Model weights) or the output of the training step, plus the
selection model's training table, which is committed (`data/uhat_train.csv.gz`).

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

### 3. Training (17 models)

```bash
for f in 0 1 2 3 4; do
  python -m filament.detector --fold $f --out weights/det_f$f.pt              # YOLO11s @1280
done
python -m filament.detector --fold 0 --imgsz 1536 --out weights/det_s1536_f0.pt
python -m filament.detector --fold 0 --model yolo11m.pt --batch 6 --out weights/det_m1280_f0.pt
for f in 0 1 2 3 4; do
  python -m filament.refiner --fold $f --out weights/refiner_f$f.pt
  python -m filament.refiner --fold $f --extent-robust --epochs 48 --out weights/refiner_extent_f$f.pt
done
```

Detectors: 40 epochs, batch 8 (YOLO11m: 6), seed 42, patience 10,
ultralytics defaults otherwise, labels = union of all annotator records'
boxes. Refiners: 12 epochs, batch 8, AdamW 3e-4, cosine schedule, 0.45 BCE +
0.55 soft Dice, one sample per (annotator record, filament).

Extent-robust refiners: the same trainer with `--extent-robust` (seeds redrawn
every epoch: 50 % jittered true boxes, 25 % short seeds covering 0.4-0.85 of
the spine, 25 % long seeds 1.2-2.0x along the spine's end tangents; a window
centred on the seed with side c x its long side, c ~ logU[1.5, 3]; target =
the whole filament) and 48 epochs: with fresh seeds every epoch, longer
training keeps helping (+0.0020 PQ for 12 -> 24 and another +0.0020 for 24 -> 48
epochs, single refiner, all five folds each time).

Runtime on a Kaggle P100: about 35 min per YOLO11s @1280 detector and 45 min
per refiner (measured; about 90 min for a 24-epoch extent-robust refiner, and
between 1.5 and 3 h for the 48-epoch ones on Kaggle's GPUs); the @1536 and YOLO11m
variants take longer (roughly 1-1.5 h each, estimated). Roughly 25 P100 GPU hours
in total, of which the five 48-epoch refiners are about 15.

### 4. Inference

```bash
# candidate pool: every cluster with >= 2 voters, its votes and its masks (production refiners)
python -m filament.ensemble --weights weights --pool test_pool.csv.gz
# the same clusters, masks redrawn by the 48-epoch extent-robust refiners
python -m filament.repaint --weights weights --pool test_pool.csv.gz --out test_pool_extent.csv.gz
# selection on the first pool, masks from the second (CPU, a few minutes)
python -m filament.selection --pool test_pool.csv.gz --masks-pool test_pool_extent.csv.gz --out submission.csv
python -m filament.submission submission.csv --test
```

The first pool took 54-55 minutes for the 180 test images on a Kaggle T4
(measured end to end, including package setup; 21 detector passes per image
and 75 refiner passes per candidate: 5 refiners x 3 views x 5 shifted seeds);
the repaint repeats only the refiner passes (under half an hour on an RTX 5080).
CPU works but takes many hours.

The detectors run once on purpose. Their outputs move slightly with the library
version and GPU (in one comparison, two of 1,985 clusters changed their order),
and the selection step needs both pools to hold exactly the same clusters.
Repainting copies boxes, votes and cluster ids from the first pool and
recomputes only the refiner columns, so the two cannot disagree; `filament.selection`
still refuses a mismatched pair. All settings are CLI flags whose defaults are the
submitted configuration (`python -m filament.ensemble -h`). The vote rule
alone (the earlier LB-0.40 submission) is
`python -m filament.ensemble --weights weights --out submission.csv`.

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
code and asserts identical training inputs (refiner samples, loss, YOLO labels,
extent-robust samples) and identical inference outputs on real checkpoints,
including the candidate pool; `tests/test_selection.py` rebuilds the submitted
swap files (production, 12-, 24- and 48-epoch extent-robust masks) row for row
from the research test pools. Both are skipped when that code is not present.
The remaining tests run anywhere on CPU in about a minute and use synthetic
boxes, masks and models; `tests/test_submission.py` also checks that
`results/submission_x48.csv` is intact (SHA-256 below) and valid.

### Notebook

`notebooks/pipeline.ipynb` walks through the whole pipeline: data and
annotator overlays, the fold split, training (guarded by `RUN_TRAINING`),
weights, test inference, validation, a fold-0 PQ evaluation and overlays of
predictions.

## Model weights

The 17 files below are published as one archive, a GitHub release asset of this
repository (`WEIGHTS_URL` in `filament/weights.py`; the link becomes live when the
v1.0 release is published). Every CLI and the notebook download it
automatically when a file is missing, and `python -m filament.weights --dest weights`
does so on its own. Every file is checked against `data/weights.sha256`; to check
them by hand, `cd weights && sha256sum -c ../data/weights.sha256`.

| File | Model | Trained on | Inference size |
|---|---|---|---|
| `det_f0.pt` ... `det_f4.pt` | YOLO11s detector | all folds except k | 1280 |
| `det_s1536_f0.pt` | YOLO11s detector | all folds except 0 | 1536 |
| `det_m1280_f0.pt` | YOLO11m detector | all folds except 0 | 1280 |
| `refiner_f0.pt` ... `refiner_f4.pt` | U-Net / ResNet-34 refiner (state dict) | all folds except k | 512 crop, window 1.5 x seed |
| `refiner_extent_f0.pt` ... `refiner_extent_f4.pt` | the same U-Net, `--extent-robust`, 48 epochs (the submitted masks) | all folds except k | 512 crop, window 2.25 x seed |

The selection model is refitted at run time (a few seconds) from
`data/uhat_train.csv.gz`: one row per candidate cluster of the nested
cross-validation pools (7,302 clusters of the 707 training images, each
scored by an ensemble that never saw its image), 88 features plus the target
u and the image's record count.

## Exactness notes

* The code was ported from the research repository with every retained
  computation unchanged; `tests/test_equivalence.py` checks refiner training
  samples (both recipes), the loss, YOLO label files, detector boxes,
  probability maps, paste-back, the full ensemble rows (both paste modes) and
  the candidate pool against it; `tests/test_selection.py` checks the 88
  features and reproduces the submitted selection files exactly.
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
  day-grouped folds, ensemble-level settings (fusion, selection) by nested
  cross-validation (see Results). The instance count (that of the 5-of-7 vote
  at conf 0.40) is the one the public leaderboard preferred over both looser
  and stricter rules.
* The selection model's training table is built from training images only.

## Repository layout

```
filament/
  data.py        COCO records, images, day-grouped folds
  metric.py      the organisers' PQ (dense reference + fast RLE backend), fold evaluation CLI
  detector.py    YOLO export (union / per-record labels) and detector training CLI
  refiner.py     box-seed crop dataset, U-Net, loss and refiner training CLI
  inference.py   detector boxes, refiner probability maps, paste-back
  ensemble.py    voting, majority masks, painting, gates, candidate pools and the prediction CLI
  repaint.py     the same pool with another refiner set's masks (clusters unchanged)
  weights.py     download and SHA-256 verification of the 17 weight files
  selection.py   count-matched swap: 88 candidate features, u-hat model, selection CLI
  submission.py  RLE helpers and the submission validator CLI
tests/           assert-based tests (pytest or plain python)
notebooks/       pipeline.ipynb
data/            cv_folds.json, uhat_train.csv.gz, weights.sha256 (+ the competition data, not committed)
results/         submission_x48.csv: the submitted file (SHA-256 f3ae2ebd...9bea54)
```

## License

MIT, see `LICENSE`.
