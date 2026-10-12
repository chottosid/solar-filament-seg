"""Extent-robust seeds: spine-interval boxes and the seed mix (synthetic straight filament)."""

import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from filament.data import Filament  # noqa: E402
from filament.refiner import SEED_MIX, interval_box, sample_seed_window  # noqa: E402


def test_interval_boxes_and_seed_mix():
    # horizontal ribbon x 100..300, y 500..510, spine along its centre (L = 200, width 10)
    f = Filament([100, 500, 300, 500, 300, 510, 100, 510], [100, 500, 200, 10], 2000.0, 1, [100, 505, 200, 505, 300, 505])
    full = interval_box(f, 0, 200, 2048, 2048)
    assert abs(full[0] - 95) <= 1 and abs(full[2] - 210) <= 2, full
    mid = interval_box(f, 50, 150, 2048, 2048)  # the middle half of the spine
    assert abs(mid[0] - 145) <= 1 and abs(mid[2] - 110) <= 2, mid
    long = interval_box(f, -50, 250, 2048, 2048)  # 1.5x, extended straight past both ends
    assert abs(long[0] - 45) <= 1 and abs(long[2] - 310) <= 2, long
    nospine = Filament(f.polygon, f.bbox, f.area, 1, [])
    assert abs(interval_box(nospine, 0, 200, 2048, 2048)[2] - full[2]) <= 2  # polygon major-axis fallback
    rng, kinds = random.Random(0), []
    for _ in range(400):
        m = sample_seed_window(f, 2048, 2048, rng)
        x0, y0, x1, y1 = m["win"]
        assert x1 - x0 == y1 - y0 >= 96 and x1 - x0 >= math.ceil(1.5 * max(m["bbox"][2], m["bbox"][3])) - 1
        kinds.append(m["kind"])
    frac = [kinds.count(k) / len(kinds) for k in ("in", "short", "long")]
    assert all(abs(a - b) < 0.08 for a, b in zip(frac, SEED_MIX)), frac


if __name__ == "__main__":
    test_interval_boxes_and_seed_mix()
    print("ok")
