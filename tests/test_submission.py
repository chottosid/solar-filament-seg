"""Submission validator on synthetic CSVs: accepts a clean file, rejects what the host rejects."""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from filament.submission import layers_to_rles, validate, write_submission  # noqa: E402


def _rle(y0, y1, x0, x1):
    m = np.zeros((2048, 2048), np.uint8)
    m[y0:y1, x0:x1] = 1
    return layers_to_rles([m])[0]


def _expect_error(rows, needle, **kwargs):
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "sub.csv"
        if isinstance(rows, pd.DataFrame):
            rows.to_csv(path, index=False)
        else:
            write_submission(rows, path)
        try:
            validate(path, **kwargs)
        except ValueError as e:
            assert needle in str(e), e
        else:
            raise AssertionError(f"accepted a submission with: {needle}")


def test_validator():
    good = [("imgA_1", _rle(0, 10, 0, 10)), ("imgA_2", _rle(10, 20, 0, 10)), ("imgB_1", _rle(0, 10, 0, 10))]
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "sub.csv"
        write_submission(good, path)
        assert validate(path).startswith("OK: 3 rows / 2 images")
        test_dir = Path(tmp) / "test_images"
        test_dir.mkdir()
        for stem in ("imgA", "imgB", "imgC"):
            (test_dir / f"{stem}.jpeg").touch()
        assert "1 test images without predictions" in validate(path, test_dir)

    _expect_error(good + [("imgA_3", _rle(5, 15, 5, 15))], "overlapping pixels")
    _expect_error(good + [("imgA_1", _rle(100, 110, 0, 10))], "duplicate filament_id")
    _expect_error(good + [("imgC_1", _rle(0, 0, 0, 0))], "empty instance mask")
    _expect_error(pd.DataFrame(good, columns=["id", "rle"]), "columns")
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "imgA.jpeg").touch()
        _expect_error(good, "stems not in test set", test_dir=Path(tmp))


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
