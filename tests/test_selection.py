"""Count-matched swap selection vs the submitted files (needs the research repository's test pools).

The research test pools (production detectors; production or extent-robust refiners) plus
data/uhat_train.csv.gz must reproduce the two submitted CSVs row for row:
swap selection with production masks (LB 0.40) and with extent-robust masks (LB 0.40).
Skipped without the research repository (default ../SOLAR_FILAMENT or $FS26_SOURCE). ~5 min on CPU.
"""

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SOURCE = Path(os.environ.get("FS26_SOURCE", REPO.parent / "SOLAR_FILAMENT"))
POOL = SOURCE / "kernel_out/test_pool/pool_test.csv.gz"
POOL_EXTENT = SOURCE / "kernel_out/test_pool_extent/pool_test_extent.csv.gz"
SUBMITTED = SOURCE / "out/analysis/selection/submission_swap_deploy.csv"
SUBMITTED_EXTENT = SOURCE / "out/submission_swap_extent.csv"
FEATS = SOURCE / "out/analysis/selection/feats_test.csv.gz"


def test_swap_reproduces_submissions():
    missing = [p for p in (POOL, POOL_EXTENT, SUBMITTED, SUBMITTED_EXTENT, FEATS) if not p.exists()]
    if missing:
        import pytest

        pytest.skip(f"research artefacts not found: {missing[0]}")
    os.environ.setdefault("MAGFILO_DIR", str(SOURCE / "MAGFiLO_1.0_Kaggle_2026"))
    from filament.data import TEST_IMAGES, data_root
    from filament.selection import count_matched, design, fit_uhat

    image_dir = data_root() / TEST_IMAGES
    pool, extent = pd.read_csv(POOL), pd.read_csv(POOL_EXTENT)
    X = design(pool, image_dir)
    ref = pd.read_csv(FEATS, index_col=0)  # research features (written/read through CSV: ~1 ulp)
    assert np.allclose(X[ref.columns].to_numpy(float), ref.to_numpy(float), rtol=1e-9, atol=1e-12, equal_nan=True)
    model, cols = fit_uhat()
    score = model.predict(X[cols])
    for masks, submitted in ((None, SUBMITTED), (extent, SUBMITTED_EXTENT)):
        rows = count_matched(pool, score, image_dir, masks)
        assert rows == list(map(tuple, pd.read_csv(submitted).to_numpy())), submitted.name


if __name__ == "__main__":
    test_swap_reproduces_submissions()
    print("ok")
