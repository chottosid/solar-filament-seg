"""PQ checks: the organisers' quirks on tiny synthetic masks, RLE vs dense backends."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from pycocotools import mask as mask_util  # noqa: E402

from filament.metric import _as_float_layers, pq_from_rles, pq_one_image, pq_totals  # noqa: E402
from filament.submission import layers_to_rles, rles_to_layers  # noqa: E402


def test_split_square_self_check():
    """One exact hit, one miss each way: PQ = 1 / (1 + 0.5 + 0.5)."""
    gt = np.zeros((2, 64, 64), dtype=np.uint8)
    gt[0, :16, :16] = 1
    gt[1, 32:48, 32:48] = 1
    pred = np.zeros((2, 64, 64), dtype=np.uint8)
    pred[0, :16, :16] = 1
    pred[1, 48:64, 48:64] = 1
    res = pq_totals(*pq_one_image(_as_float_layers(gt), _as_float_layers(pred)))
    assert abs(res["pq"] - 1.0 / (1 + 0.5 + 0.5)) < 1e-9, res
    # RLE round trip through the submission format (counts carry no size)
    rles = layers_to_rles(gt)
    assert (rles_to_layers(rles, 64, 64).astype(np.uint8) == gt).all()
    # str-counts decode path, as in the organisers' scorer
    decoded = mask_util.decode([{"size": [64, 64], "counts": r.encode("utf-8")} for r in rles]).transpose(2, 0, 1)
    assert (decoded == gt).all()


def _rle(*boxes):
    m = np.zeros((2048, 2048), np.uint8)
    for x0, y0, x1, y1 in boxes:
        m[y0:y1, x0:x1] = 1
    return layers_to_rles([m])[0]


def test_per_record_pooled_accounting():
    """Two records of one image score the same predictions twice; unlabelled images are pure FP;
    one prediction can be a TP for two overlapping GT filaments (no one-to-one matching)."""
    a, b, c = (100, 100, 140, 300), (500, 500, 540, 700), (900, 900, 940, 1100)
    gt = [
        ("010401-imgA_1", _rle(a)),
        ("010401-imgA_2", _rle(b)),
        ("010402-imgA_1", _rle(a)),  # second annotator saw only filament a
        ("010401-imgB_1", _rle(c)),  # imgB gets no prediction -> FN
        ("010401-imgD_1", _rle(a)),  # two near-identical GT ...
        ("010401-imgD_2", _rle((100, 100, 140, 290))),  # ... both hit by one prediction
    ]
    pred = [
        ("imgA_1", _rle(a)),
        ("imgA_2", _rle(c)),  # spurious: FP once per record of imgA
        ("imgC_1", _rle(a)),  # image without any record: FP
        ("imgD_1", _rle(a)),
    ]
    res = pq_from_rles(gt, pred)
    assert (res["tp"], res["fp"], res["fn"]) == (4, 3, 2), res
    assert abs(res["pq"] - (3 + 190 / 200) / (4 + 1.5 + 1.0)) < 1e-6, res
    assert res == pq_from_rles(gt, pred, backend="dense")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
