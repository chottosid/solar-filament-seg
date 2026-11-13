"""Repainting keeps clusters, votes and seed boxes and recomputes only the refiner columns."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from filament.ensemble import EnsembleConfig, image_pool  # noqa: E402
from filament.repaint import repaint_pool  # noqa: E402
from filament.submission import rles_to_layers  # noqa: E402
from tests.test_ensemble import _box, _Detector, _SeedRefiner  # noqa: E402


class _HalfRefiner(torch.nn.Module):
    """Paints only the left half of the seed box, so repainted masks visibly differ from _SeedRefiner's."""

    def forward(self, x):
        seed = (x[:, 2:3] > 0).float()
        half = torch.zeros_like(seed)
        half[..., : seed.shape[-1] // 2] = 1
        return (seed * half) * 20 - 10


def test_repaint_changes_only_refiner_columns():
    img = np.full((2048, 2048), 200, np.uint8)
    a = _box(300, 300, 340, 500)
    img[300:500, 300:340] = 60
    dets = [_Detector([(0.9, a)]) for _ in range(7)]
    cpu, cfg = torch.device("cpu"), EnsembleConfig(crop=64)
    pool = pd.DataFrame(image_pool(dets, [_SeedRefiner()] * 5, "img", img, cpu, cfg))
    assert len(pool) == 1
    new = repaint_pool(pool, [_HalfRefiner()] * 5, lambda stem: img, cpu, cfg)
    same = [c for c in pool.columns if not (c.startswith(("area_", "contrast_", "rle_")) or c == "refiner_agree")]
    assert pool[same].equals(new[same])  # votes, seed box, cluster id untouched
    assert new.area_pmean[0] < pool.area_pmean[0]  # masks were redrawn
    full, half = (rles_to_layers([p.rle_pmean[0]])[0] for p in (pool, new))
    half, full = half.astype(bool), full.astype(bool)
    assert not (half & ~full).any() and half.sum() == new.area_pmean[0]  # a subset of the old mask


if __name__ == "__main__":
    test_repaint_changes_only_refiner_columns()
    print("ok")
