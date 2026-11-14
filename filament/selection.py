"""Rank the candidates and keep as many as the vote rule would.

A gradient-boosting model predicts each candidate's matched IoU (averaged over
annotator records) from 88 features: detector votes, mask shape and contrast,
pixel statistics and overlaps with other candidates. It is fitted on
data/uhat_train.csv.gz (nested cross-validation pools of the training images).
The top N candidates are kept, N being the vote rule's instance count.

    python -m filament.selection --pool test_pool.csv.gz --masks-pool test_pool_extent.csv.gz --out submission.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from pycocotools import mask as mask_util

from .data import IMG_SIZE, ROOT, TEST_IMAGES, data_root, load_image_gray
from .ensemble import EnsembleConfig, resolve_overlaps, ring_contrast
from .submission import validate, write_submission

N_VOTERS = 7
DISK_CX, DISK_CY, DISK_R = 1023.1, 1024.1, 899.7  # fixed GONG disk geometry (+-1 px over the dataset)
VOTE_LEVELS = (0.2, 0.3, 0.4, 0.5, 0.6)
HGB = dict(max_depth=4, max_iter=200, learning_rate=0.06, min_samples_leaf=40, l2_regularization=1.0, random_state=0)
TRAIN_TABLE = ROOT / "data" / "uhat_train.csv.gz"
PAD = 48  # crop margin around a mask for its pixel features
_yy, _xx = np.mgrid[:IMG_SIZE, :IMG_SIZE]
RR = (np.hypot(_xx - DISK_CX, _yy - DISK_CY) / DISK_R).astype(np.float32)  # radius in disk radii
del _yy, _xx


def _rle(counts: str) -> dict:
    return {"size": [IMG_SIZE, IMG_SIZE], "counts": counts.encode("ascii")}


def support(pool: pd.DataFrame, vote_conf: float) -> np.ndarray:
    """Number of voters with their own confidence >= vote_conf, per cluster."""
    return (pool[[f"conf{v}" for v in range(N_VOTERS)]].to_numpy() >= vote_conf).sum(1)


def vote_features(pool: pd.DataFrame) -> pd.DataFrame:
    """Detector votes, mask area / contrast / agreement, seed-box shape and disk position (23 columns)."""
    conf = pool[[f"conf{v}" for v in range(N_VOTERS)]].to_numpy()
    present = conf > 0
    f = pd.DataFrame({f"conf{v}": conf[:, v] for v in range(N_VOTERS)})
    for t in VOTE_LEVELS:
        f[f"votes{t}"] = (conf >= t).sum(1)
    f["conf_mean"] = np.where(present.any(1), (conf * present).sum(1) / np.maximum(present.sum(1), 1), 0)
    f["conf_max"], f["seed_conf"] = conf.max(1), pool["seed_conf"].to_numpy()
    f["refiner_agree"] = pool["refiner_agree"].to_numpy()
    area = pool["area_pmean"].to_numpy().astype(float)
    f["log_area"] = np.log1p(area)
    f["pmean_ratio"] = pool["area_pmean"].to_numpy() / np.maximum(area, 1)
    f["contrast"] = pool["contrast_pmean"].to_numpy()
    w, h = (pool.x2 - pool.x1).to_numpy(), (pool.y2 - pool.y1).to_numpy()
    f["log_w"], f["log_h"], f["aspect"] = np.log1p(w), np.log1p(h), np.log(np.maximum(w, 1) / np.maximum(h, 1))
    f["r_norm"] = np.hypot((pool.x1 + pool.x2) / 2 - DISK_CX, (pool.y1 + pool.y2) / 2 - DISK_CY).to_numpy() / DISK_R
    fc = pool[[c for c in pool.columns if c.startswith("fconf")]].to_numpy()  # votes on the flipped images
    f["f_mean"], f["f_votes4"], f["f_votes3"] = fc.mean(1), (fc >= 0.4).sum(1), (fc >= 0.3).sum(1)
    f["f_max"], f["all_votes4"] = fc.max(1), (fc >= 0.4).sum(1) + f["votes0.4"]
    return f


def _ridge(g: np.ndarray, s: float, bg: float) -> np.ndarray:
    """sigma^2 * largest Hessian eigenvalue / bg: positive on dark ribbons of width ~2-3 sigma."""
    b = cv2.GaussianBlur(g, (0, 0), s)
    gy, gx = np.gradient(b)
    ixx, ixy = np.gradient(gx, axis=1), np.gradient(gx, axis=0)
    iyy = np.gradient(gy, axis=0)
    return s * s * (0.5 * (ixx + iyy) + np.sqrt(0.25 * (ixx - iyy) ** 2 + ixy ** 2)) / bg


def _crop_features(g: np.ndarray, m: np.ndarray, rr: np.ndarray) -> dict:
    """Pixel and shape features of one mask in its padded crop (g float32 image, m uint8 mask)."""
    mb = m.astype(bool)
    a = int(mb.sum())
    din = cv2.distanceTransform(m, cv2.DIST_L2, 3)
    dout = cv2.distanceTransform((1 - m).astype(np.uint8), cv2.DIST_L2, 3)
    bgsel = (dout >= 10) & (dout <= 30)
    if bgsel.sum() < 20:
        bgsel = ~mb
    bg = max(float(np.median(g[bgsel])), 1.0)
    inside = g[mb]

    def prof(sel):
        return (bg - float(g[sel].mean())) / bg if sel.any() else 0.0

    f = {"bg_rel": bg, "bg_std": float(g[bgsel].std()) / bg,
         "p_in": prof(mb), "p_core": prof(din >= 2) if (din >= 2).any() else prof(mb),
         "p_d1": prof((dout > 0) & (dout <= 1.5)), "p_d3": prof((dout > 1.5) & (dout <= 3.5)),
         "p_d6": prof((dout > 3.5) & (dout <= 6.5)), "p_d9": prof((dout > 6.5) & (dout <= 9.5)),
         "q05": (bg - float(np.percentile(inside, 5))) / bg, "q50": (bg - float(np.median(inside))) / bg,
         "q95": (bg - float(np.percentile(inside, 95))) / bg,
         "dark_frac": float((inside < 0.95 * bg).mean()), "in_std": float(inside.std()) / bg}
    f["edge_step"] = f["p_in"] - f["p_d1"]
    f["leak"] = f["p_d3"] - f["p_d9"]
    b1 = cv2.GaussianBlur(g, (0, 0), 1.0)
    gm = np.hypot(cv2.Sobel(b1, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(b1, cv2.CV_32F, 0, 1, ksize=3)) / 8
    edge = mb & (din <= 1.0)
    f["grad_edge"] = float(gm[edge].mean()) / bg if edge.any() else 0.0
    f["grad_ratio"] = float(gm[edge].mean() / (gm[bgsel].mean() + 1e-3)) if edge.any() else 0.0
    for s in (1.5, 3.0):
        r = _ridge(g, s, bg)
        tag = f"r{int(s * 10)}"
        f[f"{tag}_in"] = float(r[mb].mean())
        f[f"{tag}_in90"] = float(np.percentile(r[mb], 90))
        f[f"{tag}_bg"] = float(r[bgsel].mean())
        strong = r > (0.015 if s < 2 else 0.025)
        near, wide = (strong & (dout <= 2)).sum(), (strong & (dout <= 20)).sum()
        f[f"{tag}_cover"] = float(near / max(wide, 1))
        f[f"{tag}_strong_in"] = float(strong[mb].mean())
    n_cc, _, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    f["n_cc"] = n_cc - 1
    f["cc_big"] = float(st[1:, cv2.CC_STAT_AREA].max() / a) if n_cc > 1 else 1.0
    f["dt_mean"], f["dt_max"] = float(din[mb].mean()), float(din.max())
    f["len_proxy"] = a / max(f["dt_mean"], 0.5)
    ys, xs = np.nonzero(mb)
    ev = np.linalg.eigvalsh(np.cov(np.vstack([xs, ys]).astype(float)) + 1e-6 * np.eye(2))
    f["elong"] = float(np.sqrt(ev[1] / max(ev[0], 1e-6)))
    f["fill"] = a / float((np.ptp(xs) + 1) * (np.ptp(ys) + 1))
    f["limb_frac"] = float((rr[mb] > 1.0).mean())
    f["r_cent"] = float(rr[mb].mean())
    return f


def image_features(img: np.ndarray, sub: pd.DataFrame) -> list[dict]:
    """Image, context and pixel features for one image's clusters (rows of `sub`, keyed by its index)."""
    img = img.astype(np.float32)
    dv = img[RR < 0.95]
    gmed = float(np.median(dv))
    glob = {"g_med": gmed, "g_std": float(dv.std()) / gmed, "g_p01": float(np.percentile(dv, 1)) / gmed,
            "g_dark": float((dv < 0.8 * gmed).mean())}
    rles = [_rle(s) for s in sub.rle_pmean]
    n = len(rles)
    area = mask_util.area(rles).astype(float)
    C = np.asarray(mask_util.iou(rles, rles, [1] * n)).reshape(n, n)  # C[i, j] = |i & j| / |i|
    np.fill_diagonal(C, 0)
    inter = C * area[:, None]
    iou = inter / np.maximum(area[:, None] + area[None, :] - inter, 1)
    sc, sup5, sup4 = sub.seed_conf.to_numpy(), support(sub, 0.40), support(sub, 0.30)
    cx, cy = ((sub.x1 + sub.x2) / 2).to_numpy(), ((sub.y1 + sub.y2) / 2).to_numpy()
    dist = np.hypot(cx[:, None] - cx[None, :], cy[:, None] - cy[None, :])
    anchor = sup5 >= 5
    out = []
    for i, row in enumerate(sub.itertuples()):
        f = {"idx": row.Index, **glob}
        higher = sc > sc[i]
        f.update({"cov_max": C[i].max(initial=0), "cont_max": C[:, i].max(initial=0),
                  "iou_max": iou[i].max(initial=0), "n_ovl": int((C[i] > 0.05).sum()),
                  "cov_hi_anchor": min(1.0, (C[i] * (higher & anchor)).sum()),
                  "cov_hi_any": min(1.0, (C[i] * higher).sum()),
                  "cont_lo_anchor": (C[:, i] * (~higher & anchor)).max(initial=0),
                  "n_img": n, "n_img_anchor": int(anchor.sum()), "n_img_sup4": int((sup4 >= 4).sum()),
                  "img_conf_mean": float(sc.mean()), "conf_rank": float((sc > sc[i]).sum()) / n,
                  "conf_gap_top": float(sc.max() - sc[i]),
                  "dens150": int((dist[i] < 150).sum()) - 1, "dens300": int((dist[i] < 300).sum()) - 1})
        j = int(iou[i].argmax()) if n > 1 else i
        f["partner_sup"] = int(sup5[j]) if iou[i, j] > 0 else -1
        f["partner_dconf"] = float(sc[i] - sc[j]) if iou[i, j] > 0 else 0.0
        p = _rle(row.rle_pmean)
        for v in ("nearest", "bilinear"):
            f[f"iou_{v}"] = float(mask_util.iou([p], [_rle(getattr(row, f"rle_{v}"))], [0])[0][0])
        if area[i] > 0:
            x, y, w, h = mask_util.toBbox(p)
            x0, y0 = int(max(0, x - PAD)), int(max(0, y - PAD))
            x1, y1 = int(min(IMG_SIZE, x + w + PAD)), int(min(IMG_SIZE, y + h + PAD))
            m = np.ascontiguousarray(mask_util.decode(p)[y0:y1, x0:x1])
            f.update(_crop_features(img[y0:y1, x0:x1], m, RR[y0:y1, x0:x1]))
            f["bg_rel"] /= gmed
            bx = (row.x1, row.y1, row.x2, row.y2)
            ix = max(0, min(x + w, bx[2]) - max(x, bx[0])) * max(0, min(y + h, bx[3]) - max(y, bx[1]))
            f["mbox_iou"] = ix / max(w * h + (bx[2] - bx[0]) * (bx[3] - bx[1]) - ix, 1)
        out.append(f)
    return out


def design(pool: pd.DataFrame, image_dir: Path) -> pd.DataFrame:
    """The 88-column feature matrix of a pool (one row per cluster, pool order)."""
    rows = []
    for stem, sub in pool.groupby("stem", sort=True):
        rows += image_features(load_image_gray(image_dir, stem + ".jpeg"), sub)
    pix = pd.DataFrame(rows).set_index("idx").reindex(pool.index)
    return pd.concat([vote_features(pool).set_index(pool.index), pix], axis=1)


def fit_uhat(table: Path = TRAIN_TABLE):
    """u-hat regressor fitted on the nested-CV training table (columns: features..., u, nrec)."""
    from sklearn.ensemble import HistGradientBoostingRegressor

    t = pd.read_csv(table, float_precision="round_trip")  # exact floats (the default parser is ~1 ulp off)
    X = t.drop(columns=["u", "nrec"])
    return HistGradientBoostingRegressor(**HGB).fit(X, t.u, sample_weight=t.nrec), list(X.columns)


def paint_rows(pool: pd.DataFrame, keep: np.ndarray, order: np.ndarray, image_dir: Path,
               cfg: EnsembleConfig = EnsembleConfig()) -> list[tuple[str, str]]:
    """Submission rows for the kept clusters, painted per image in descending `order` with the vote rule's gates."""
    from .submission import layers_to_rles

    rows = []
    for stem, sub in pool[keep].groupby("stem", sort=True):
        scored = [(order[i], mask_util.decode(_rle(r))) for i, r, a in zip(sub.index, sub.rle_pmean, sub.area_pmean)
                  if a >= cfg.min_area]
        painted = resolve_overlaps(scored, cfg.post_min_area) if scored else []
        img = load_image_gray(image_dir, stem + ".jpeg") if painted else None
        layers = [m for m in painted if ring_contrast(m, img) >= cfg.min_contrast]
        rows += [(f"{stem}_{k}", rle) for k, rle in enumerate(layers_to_rles(layers), 1)]
    return rows


def count_matched(pool: pd.DataFrame, score: np.ndarray, image_dir: Path,
                  masks: pd.DataFrame | None = None) -> list[tuple[str, str]]:
    """Top-N clusters by score, painted in score order; N gives the 5-of-7 rule's instance count.

    `masks` (same clusters, other refiners) supplies the painted masks; selection and the
    target count always use `pool`.
    """
    order = np.asarray(score, float)
    target = len(paint_rows(pool, support(pool, 0.40) >= 5, pool.seed_conf.to_numpy(), image_dir))
    elig = pool.area_pmean.to_numpy() >= EnsembleConfig().min_area
    ranked = np.flatnonzero(elig)[np.lexsort((-pool.seed_conf.to_numpy()[elig], -order[elig]))]
    n, seen = int(((support(pool, 0.40) >= 5) & elig).sum()), {}
    for _ in range(40):  # painting drops fragments, so search N for an exact row count
        keep = np.zeros(len(pool), bool)
        keep[ranked[:n]] = True
        got = len(paint_rows(pool, keep, order, image_dir))
        seen[n] = (got, keep)
        if got == target or n + (1 if got < target else -1) in seen:
            break
        n += target - got if abs(target - got) > 1 else (1 if got < target else -1)
    got, keep = seen[min(seen, key=lambda q: (seen[q][0] > target, abs(seen[q][0] - target)))]
    return paint_rows(pool if masks is None else masks, keep, order, image_dir)


def main() -> None:
    ap = argparse.ArgumentParser(description="Count-matched swap selection on a candidate pool (CPU).",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--pool", type=Path, required=True, help="candidate pool from python -m filament.ensemble --pool")
    ap.add_argument("--masks-pool", type=Path, default=None,
                    help="same clusters with another refiner set: paint these masks (selection uses --pool)")
    ap.add_argument("--train", type=Path, default=TRAIN_TABLE, help="nested-CV training table")
    ap.add_argument("--out", type=Path, default=Path("submission.csv"))
    args = ap.parse_args()
    image_dir = data_root() / TEST_IMAGES
    pool = pd.read_csv(args.pool)
    masks = None
    if args.masks_pool:
        masks = pd.read_csv(args.masks_pool)
        key = ["stem", "cluster_id", "x1", "y1", "x2", "y2"]
        if len(masks) != len(pool) or not (masks[key].to_numpy() == pool[key].to_numpy()).all():
            ap.error("--masks-pool clusters do not match --pool (run both with the same detectors)")
    model, cols = fit_uhat(args.train)
    score = model.predict(design(pool, image_dir)[cols])
    rows = count_matched(pool, score, image_dir, masks)
    write_submission(rows, args.out)
    print(f"wrote {args.out}: {len(rows)} rows")
    print(validate(args.out, image_dir))


if __name__ == "__main__":
    main()
