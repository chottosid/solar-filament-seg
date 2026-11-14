"""YOLO11 filament detector (one class). Labels are the union of all annotators' boxes.

    python -m filament.detector --fold 0 --out weights/det_f0.pt
    python -m filament.detector --fold 0 --imgsz 1536 --out weights/det_s1536_f0.pt
    python -m filament.detector --fold 0 --model yolo11m.pt --batch 6 --out weights/det_m1280_f0.pt
"""

from __future__ import annotations

import argparse
import shutil
from collections import defaultdict
from pathlib import Path

from .data import IMG_SIZE, TRAIN_IMAGES, Record, data_root, fold_stems, load_records


def export_yolo(records: list[Record], val_stems: set[str], images_dir: Path, out_dir: Path,
                per_record: bool = False) -> tuple[int, int]:
    """Write a YOLO dataset (images/ symlinks, labels/, data.yaml); return (n_train, n_val) samples.

    Default: one sample per physical image with the union of all records'
    boxes (how the submitted detectors were trained). per_record=True instead
    gives TRAIN one sample per record (same pixels, that annotator's boxes
    only), so confidence estimates "a random annotator draws it" rather than
    "any annotator does". Validation always uses the union.
    """
    boxes_by_file: dict[str, list[list[float]]] = defaultdict(list)
    record_boxes: dict[str, list[list[float]]] = defaultdict(list)  # "<stem>__<record>.jpeg" -> boxes
    for rec in records:
        for f in rec.filaments:
            boxes_by_file[rec.file_name].append(f.bbox)
            batch_id = rec.record_id.split("-", 1)[0]
            record_boxes[f"{rec.stem}__{batch_id}{Path(rec.file_name).suffix}"].append(f.bbox)

    n = [0, 0]
    for split, stems in (("train", None), ("val", val_stems)):
        img_dir = out_dir / "images" / split
        lbl_dir = out_dir / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        samples = record_boxes if (per_record and split == "train") else boxes_by_file
        for file_name, boxes in sorted(samples.items()):
            stem = file_name.rsplit(".", 1)[0].split("__", 1)[0]
            if (stems is None) == (stem in val_stems):  # train: not in val; val: in val
                continue
            src = images_dir / f"{stem}{Path(file_name).suffix}"
            dst = img_dir / file_name
            if not dst.exists():
                try:
                    dst.symlink_to(src)
                except OSError:  # filesystems without symlinks
                    shutil.copy2(src, dst)
            lines = [f"0 {(x + bw / 2) / IMG_SIZE:.6f} {(y + bh / 2) / IMG_SIZE:.6f} {bw / IMG_SIZE:.6f} "
                     f"{bh / IMG_SIZE:.6f}" for x, y, bw, bh in boxes]
            (lbl_dir / f"{Path(file_name).stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
            n[0 if split == "train" else 1] += 1
    (out_dir / "data.yaml").write_text(
        f"path: {out_dir}\ntrain: images/train\nval: images/val\nnames:\n  0: filament\n"
    )
    return n[0], n[1]


def train_detector(fold: int, out: Path, model: str = "yolo11s.pt", imgsz: int = 1280, batch: int = 8,
                   epochs: int = 40, work: Path = Path("runs"), per_record: bool = False) -> Path:
    """Export fold `fold` and train one YOLO model on the other folds; copy best.pt to `out`.

    imgsz 1280 (not the 640 default): boxes of 5-15 px wide filaments must
    survive the letterbox downscale of the 2048 px frame. Everything else is
    ultralytics' default augmentation; seed 42, early stop after 10 flat epochs.
    """
    import torch
    from ultralytics import YOLO

    work = Path(work).resolve() / Path(out).stem  # e.g. runs/det_f0/{yolo_data,det}
    n_tr, n_va = export_yolo(load_records(), fold_stems(fold), data_root() / TRAIN_IMAGES, work / "yolo_data",
                             per_record)
    print(f"yolo fold {fold}: train {n_tr} / val {n_va} samples -> {work / 'yolo_data'}")
    if not (n_tr and n_va):
        raise ValueError("empty train or val split")
    YOLO(model).train(data=str(work / "yolo_data" / "data.yaml"), epochs=epochs, imgsz=imgsz, batch=batch,
                      workers=2, seed=42, device=0 if torch.cuda.is_available() else "cpu", patience=10,
                      project=str(work), name="det", exist_ok=True)
    best = work / "det" / "weights" / "best.pt"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, out)
    print(f"saved {out} from {best}")
    return Path(out)


def main() -> None:
    ap = argparse.ArgumentParser(description="Train one YOLO filament detector on all folds except --fold.",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--fold", type=int, required=True, help="held-out fold (training uses the other four)")
    ap.add_argument("--out", type=Path, required=True, help="where to copy best.pt, e.g. weights/det_f0.pt")
    ap.add_argument("--model", default="yolo11s.pt", help="COCO-pretrained start (auto-downloaded by ultralytics)")
    ap.add_argument("--imgsz", type=int, default=1280, help="training image size")
    ap.add_argument("--batch", type=int, default=8, help="batch size (6 for yolo11m on a 16 GB GPU)")
    ap.add_argument("--epochs", type=int, default=40, help="maximum epochs (early stop: patience 10)")
    ap.add_argument("--work", type=Path, default=Path("runs"), help="scratch dir for the YOLO dataset and logs")
    ap.add_argument("--per-record", action="store_true",
                    help="one train sample per annotator record (not used by the submitted models)")
    args = ap.parse_args()
    train_detector(args.fold, args.out, args.model, args.imgsz, args.batch, args.epochs, args.work, args.per_record)


if __name__ == "__main__":
    main()
