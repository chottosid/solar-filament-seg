"""Rebuild the submitted CSVs from the research test pools.

Needs the research repository (../SOLAR_FILAMENT or $FS26_SOURCE); skipped
otherwise. About 40 minutes on CPU.
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
POOL_EXTENT24 = SOURCE / "kernel_out/test_pool_extent24/pool_test_extent24.csv.gz"
POOL_X48 = SOURCE / "runs/rtx_queue_20261113/repaint_x48.csv.gz"
SUBMITTED = SOURCE / "out/analysis/selection/submission_swap_deploy.csv"
SUBMITTED_EXTENT = SOURCE / "out/submission_swap_extent.csv"
SUBMITTED_EXTENT24 = SOURCE / "out/submission_swap_extent24.csv"
SUBMITTED_X48 = REPO / "results/submission_x48.csv"
FEATS = SOURCE / "out/analysis/selection/feats_test.csv.gz"


def test_swap_reproduces_submissions():
    missing = [p for p in (POOL, POOL_EXTENT, POOL_EXTENT24, POOL_X48, SUBMITTED, SUBMITTED_EXTENT, SUBMITTED_EXTENT24, FEATS)
               if not p.exists()]
    if missing:
        import pytest

        pytest.skip(f"research artefacts not found: {missing[0]}")
    os.environ.setdefault("MAGFILO_DIR", str(SOURCE / "MAGFiLO_1.0_Kaggle_2026"))
    from filament.data import TEST_IMAGES, data_root
    from filament.selection import count_matched, design, fit_uhat

    image_dir = data_root() / TEST_IMAGES
    pool, extent, extent24, x48 = (pd.read_csv(p) for p in (POOL, POOL_EXTENT, POOL_EXTENT24, POOL_X48))
    X = design(pool, image_dir)
    ref = pd.read_csv(FEATS, index_col=0)  # research features (written/read through CSV: ~1 ulp)
    assert np.allclose(X[ref.columns].to_numpy(float), ref.to_numpy(float), rtol=1e-9, atol=1e-12, equal_nan=True)
    model, cols = fit_uhat()
    score = model.predict(X[cols])
    for masks, submitted in ((None, SUBMITTED), (extent, SUBMITTED_EXTENT), (extent24, SUBMITTED_EXTENT24), (x48, SUBMITTED_X48)):
        rows = count_matched(pool, score, image_dir, masks)
        assert rows == list(map(tuple, pd.read_csv(submitted).to_numpy())), submitted.name


if __name__ == "__main__":
    test_swap_reproduces_submissions()
    print("ok")
