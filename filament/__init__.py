"""Solar filament instance segmentation on GONG H-alpha full-disk images.

Modules:
    data        annotations, images, cross-validation folds
    metric      the competition's PQ metric
    detector    YOLO11 detector training
    refiner     U-Net refiner training
    inference   detector boxes and refiner probability maps
    ensemble    voting, candidate pools, painting, prediction CLI
    repaint     redraw a pool's masks with another refiner set
    selection   candidate ranking and selection
    submission  RLE helpers and submission validator
    weights     download and checksum the trained weights
"""
