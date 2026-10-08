"""Solar filament instance segmentation on GONG H-alpha full-disk images.

Two-stage pipeline: YOLO11 boxes -> box-seeded U-Net masks, combined by a
7-detector / 5-refiner vote and painted into non-overlapping instances.

Modules (import order = data flow):
    data        COCO annotations, annotator records, images, day-grouped folds
    metric      the organisers' Panoptic Quality, replicated exactly
    detector    YOLO dataset export and detector training
    refiner     box-seed crop dataset, U-Net and refiner training
    inference   single-model primitives (boxes, probability maps, paste-back)
    ensemble    voting, mask majority, painting and the submission CLI
    submission  RLE helpers and the submission validator
"""
