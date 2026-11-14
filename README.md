# Solar filament segmentation

Our solution to the [Solar Filament Segmentation Challenge 2026](https://www.kaggle.com/competitions/filament-segmentation-2026)
(team Anonymous Authors). Public leaderboard: 0.41 PQ.

## Method

A two-stage instance segmentation pipeline:

1. **Detection.** Seven YOLO11 detectors (five YOLO11s at 1280 px, one per fold,
   plus a YOLO11s at 1536 px and a YOLO11m at 1280 px) propose boxes. Boxes from
   different detectors are grouped, and every group with at least two detectors
   becomes a candidate.
2. **Segmentation.** For each candidate, five U-Nets (ResNet-34 encoder) segment
   the filament in a crop around its box. The final masks come from a second set
   of U-Nets trained on boxes that are too short or too long, so they do not just
   copy the box's extent.
3. **Selection.** A gradient-boosting model predicts how well each candidate
   matches the annotations and keeps the best ones, as many as a 5-of-7 detector
   vote would keep. Masks are painted without overlap and small or low-contrast
   masks are removed.

Details and results are in the report.

## Results

Nested cross-validation on the 707 training images (the whole ensemble is
retrained inside each outer fold):

| Pipeline | PQ |
|---|---|
| 7-detector vote, majority of masks | 0.4379 |
| 7-detector vote, mean probability | 0.4421 |
| + candidate selection | 0.4488 |
| + extent-robust masks (final) | 0.4509 |

The submitted file is `results/submission_x48.csv` (Kaggle submission 58306582).

## Setup

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
export MAGFILO_DIR=/path/to/MAGFiLO_1.0_Kaggle_2026     # competition data (train/ and test/)
python -m filament.weights --dest weights               # download the 17 trained models (1.1 GB)
```

On Kaggle's Python 3.13 images, keep the preinstalled torch and torchvision and
install the other packages from `requirements.txt`.

The weights are attached to the [v1.0 release](https://github.com/chottosid/solar-filament-seg/releases/tag/v1.0)
and checked against `data/weights.sha256`. The scripts and the notebook download
them automatically if they are missing.

## Inference

```bash
python -m filament.ensemble --weights weights --pool test_pool.csv.gz
python -m filament.repaint --weights weights --pool test_pool.csv.gz --out test_pool_extent.csv.gz
python -m filament.selection --pool test_pool.csv.gz --masks-pool test_pool_extent.csv.gz --out submission.csv
python -m filament.submission submission.csv --test
```

This takes just under two hours for the 180 test images on a Kaggle T4. A fresh
run reproduced 1048 of the 1049 submitted masks at IoU > 0.95; small differences
come from GPU and library versions.

## Training

```bash
for f in 0 1 2 3 4; do
  python -m filament.detector --fold $f --out weights/det_f$f.pt
  python -m filament.refiner --fold $f --out weights/refiner_f$f.pt
  python -m filament.refiner --fold $f --extent-robust --epochs 48 --out weights/refiner_extent_f$f.pt
done
python -m filament.detector --fold 0 --imgsz 1536 --out weights/det_s1536_f0.pt
python -m filament.detector --fold 0 --model yolo11m.pt --batch 6 --out weights/det_m1280_f0.pt
```

About 25 GPU hours on a Kaggle P100. The selection model is fitted at run time
from `data/uhat_train.csv.gz`.

## Notebook

`notebooks/pipeline.ipynb` runs the whole pipeline: data, folds, training
(off by default), weights, test inference, validation and a fold-0 evaluation.
`notebooks/pipeline_executed.ipynb` is the same notebook after a full run on a
Kaggle T4.

## Tests

```bash
python -m pytest -q
```

`tests/test_equivalence.py` and `tests/test_selection.py` compare against our
research code and are skipped without it.

## Data

Only the competition's training images and annotations were used, with
COCO-pretrained YOLO11 and ImageNet-pretrained ResNet-34 weights. Folds are
grouped by observing day (`data/cv_folds.json`).

## Layout

```
filament/    package (data, metric, detector, refiner, inference, ensemble, repaint, selection, submission, weights)
notebooks/   pipeline notebook
tests/       tests
data/        folds, selection training table, weight checksums
results/     submitted file
```

## License

MIT
