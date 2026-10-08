"""YOLO export on a tiny synthetic COCO file: union-of-records vs one-sample-per-record labels."""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from filament.data import load_records  # noqa: E402
from filament.detector import export_yolo  # noqa: E402

COCO = {
    "images": [
        {"id": "010401-imgA", "file_name": "imgA.jpeg"},
        {"id": "010402-imgA", "file_name": "imgA.jpeg"},  # second annotator record of imgA
        {"id": "010401-imgB", "file_name": "imgB.jpeg"},
        {"id": "010403-imgC", "file_name": "imgC.jpeg"},  # validation image
    ],
    "annotations": [
        {"image_id": image_id, "segmentation": [[0, 0, 1, 0, 1, 1]], "bbox": bbox, "area": 1, "category_id": 1}
        for image_id, bbox in [
            ("010401-imgA", [0, 0, 2048, 2048]),
            ("010401-imgA", [100, 200, 40, 60]),
            ("010402-imgA", [104, 198, 36, 66]),
            ("010401-imgB", [10, 10, 20, 20]),
            ("010403-imgC", [50, 50, 30, 30]),
        ]
    ],
}


def _labels(out: Path, split: str) -> dict[str, int]:
    return {p.stem: len(p.read_text().splitlines()) for p in sorted((out / "labels" / split).glob("*.txt"))}


def test_union_vs_per_record_export():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "coco.json").write_text(json.dumps(COCO))
        images = tmp / "images"
        images.mkdir()
        for stem in ("imgA", "imgB", "imgC"):
            (images / f"{stem}.jpeg").touch()
        records = load_records(tmp / "coco.json")
        assert [len(r.filaments) for r in records] == [2, 1, 1, 1]

        assert export_yolo(records, {"imgC"}, images, tmp / "union") == (2, 1)
        assert _labels(tmp / "union", "train") == {"imgA": 3, "imgB": 1}  # union of both imgA records
        assert _labels(tmp / "union", "val") == {"imgC": 1}
        first = (tmp / "union" / "labels" / "train" / "imgA.txt").read_text().splitlines()[0]
        assert first == "0 0.500000 0.500000 1.000000 1.000000"  # normalised xywh centre format
        assert (tmp / "union" / "images" / "train" / "imgA.jpeg").exists()

        assert export_yolo(records, {"imgC"}, images, tmp / "per_record", per_record=True) == (3, 1)
        assert _labels(tmp / "per_record", "train") == {"imgA__010401": 2, "imgA__010402": 1, "imgB__010401": 1}
        assert _labels(tmp / "per_record", "val") == {"imgC": 1}  # validation stays union
        linked = tmp / "per_record" / "images" / "train" / "imgA__010402.jpeg"
        assert linked.resolve() == (images / "imgA.jpeg").resolve()  # record samples share the image


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
