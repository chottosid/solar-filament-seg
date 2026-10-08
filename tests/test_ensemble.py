"""Voting, painting and the full image_rows path on synthetic boxes, masks and models."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from filament.ensemble import EnsembleConfig, cluster_boxes, image_rows, resolve_overlaps, ring_contrast  # noqa: E402
from filament.submission import rles_to_layers  # noqa: E402


def _box(x0, y0, x1, y1):
    return np.array([x0, y0, x1, y1], np.float32)


def test_cluster_boxes_links_voters_transitively():
    long_box = _box(100, 100, 400, 140)
    per_voter = [
        [(0.9, _box(100, 100, 250, 140)), (0.6, _box(250, 100, 400, 140))],  # voter 0 split the filament in two
        [(0.8, long_box)],  # voter 1 links both halves
        [(0.7, _box(105, 100, 395, 142))],
        [(0.95, _box(1000, 1000, 1040, 1100))],  # lone box
        [],
    ]
    clusters = cluster_boxes(per_voter, vote_iou=0.30)
    assert sorted(len(c.members) for c in clusters) == [1, 3]
    big = max(clusters, key=lambda c: len(c.members))
    by_voter = {v: conf for v, conf, _ in big.members}
    assert by_voter == {0: 0.9, 1: 0.8, 2: 0.7}  # one box per voter: voter 0 keeps its best half
    # boxes of the same voter are never linked to each other directly
    assert len(cluster_boxes([[(0.9, long_box), (0.8, long_box)]], 0.30)) == 2


def test_resolve_overlaps_paints_greedily_and_drops_fragments():
    def mask(x0, x1):
        m = np.zeros((64, 64), np.uint8)
        m[10:30, x0:x1] = 1
        return m

    scored = [
        (0.5, mask(0, 20)),  # 400 px, keeps everything (highest score)
        (0.4, mask(15, 40)),  # 500 px, loses 100 px to the first mask
        (0.3, mask(0, 18)),  # fully covered -> vanishes
        (0.2, mask(39, 45)),  # only a 100 px fragment survives -> dropped, but keeps its pixels
        (0.1, mask(40, 60)),  # gets only pixels nobody claimed (45..60), even after the fragment is dropped
    ]
    out = resolve_overlaps(scored, post_min_area=0)
    assert [int(m.sum()) for m in out] == [400, 400, 100, 300]
    out = resolve_overlaps(scored, post_min_area=150)
    assert [int(m.sum()) for m in out] == [400, 400, 300]
    stack = np.stack(resolve_overlaps(scored))
    assert stack.sum(0).max() == 1  # never two instances on one pixel


def test_ring_contrast_prefers_dark_features():
    img = np.full((200, 200), 200, np.uint8)
    img[90:110, 50:150] = 60
    m = np.zeros((200, 200), np.uint8)
    m[90:110, 50:150] = 1
    assert ring_contrast(m, img) > 0.5
    assert ring_contrast(m, 255 - img) < 0  # a bright feature is not a filament
    assert ring_contrast(np.zeros_like(m), img) == 0.0


class _Boxes:
    def __init__(self, items):
        self.xyxy = torch.tensor(np.array([b for _, b in items], np.float32).reshape(-1, 4))
        self.conf = torch.tensor([c for c, _ in items], dtype=torch.float32)

    def __len__(self):
        return len(self.conf)


class _Result:
    def __init__(self, items):
        self.boxes = _Boxes(items)


class _Detector:
    """Stands in for an ultralytics YOLO model: returns fixed boxes."""

    def __init__(self, items):
        self.items = items

    def predict(self, **kwargs):
        return [_Result(self.items)]


class _SeedRefiner(torch.nn.Module):
    """Predicts exactly the seed channel: the mask is the padded seed box."""

    def forward(self, x):
        return (x[:, 2:3] > 0).float() * 20 - 10


def test_image_rows_votes_and_paints():
    img = np.full((2048, 2048), 200, np.uint8)
    a, b, lone = _box(300, 300, 340, 500), _box(1000, 1000, 1040, 1200), _box(1500, 400, 1540, 600)
    for x0, y0, x1, y1 in (a, b, lone):
        img[int(y0):int(y1), int(x0):int(x1)] = 60  # dark filaments
    dets = [_Detector([(0.9, a), (0.9, b)]) for _ in range(4)]  # b: 4 confident voters ...
    dets += [_Detector([(0.9, a), (0.3, b)]) for _ in range(2)]  # ... + 2 below the vote conf
    dets += [_Detector([(0.9, a), (0.9, lone)])]  # lone: a single voter
    refiners = [_SeedRefiner()] * 5
    cfg = EnsembleConfig(crop=64)
    rows = image_rows(dets, refiners, "img", img, torch.device("cpu"), cfg, ("nearest", "bilinear"))
    for mode, mode_rows in rows.items():
        assert [fid for fid, _ in mode_rows] == ["img_1"], (mode, mode_rows)  # only a has >= 5 votes
        ys, xs = np.nonzero(rles_to_layers([mode_rows[0][1]])[0])
        assert xs.min() < 300 and xs.max() >= 340 and ys.min() < 300 and ys.max() >= 500, mode  # covers a


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
