"""search_mask_pattern.py

Mask-driven pattern search (scale-invariant) + visualization.

This script is the "first approach": use the pixels you painted in mask mode
as a template and search every rendered page image for that same pattern.

Key properties
--------------
* Components vary in size: we search across a scale range.
* We do NOT care about absolute size; we care that the *pattern* matches.
* The quality score is "masked pixel overlap":

     overlap = (# template pixels that are black in the candidate patch)
                  -----------------------------------------------
                             (# template pixels)

So overlap>=0.90 means "90% of your painted-black template pixels are present".

Speed strategy
--------------
Searching every location at every scale at full resolution is slow.
We use a coarse-to-fine pipeline:

1) Build a binary template from your overlay:
      template = painted_pixels AND black_pixels_in_source
2) For each page:
    a) downscale the page (default 0.25x)
    b) run matchTemplate on the downscaled image for a set of scales
    c) extract local maxima above a coarse threshold (candidate locations)
    d) map candidates back to full-res and verify with the overlap metric
    e) apply non-maximum suppression so you get clean boxes
3) Save annotated images with bboxes so you can eyeball results.
"""

import argparse
import json
import os
import re
from dataclasses import dataclass
from typing import Iterable

import cv2
import numpy as np


@dataclass(frozen=True)
class Match:
    page: int
    overlap: float
    scale: float
    x: int
    y: int
    w: int
    h: int


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _natural_page_sort_key(path: str):
    m = re.search(r"page_(\d+)\.png$", os.path.basename(path))
    return int(m.group(1)) if m else 10**9


def _iter_page_images(pages_dir: str) -> Iterable[tuple[int, str]]:
    files = [
        os.path.join(pages_dir, f)
        for f in os.listdir(pages_dir)
        if f.lower().endswith(".png") and f.startswith("page_")
    ]
    files.sort(key=_natural_page_sort_key)
    for path in files:
        m = re.search(r"page_(\d+)\.png$", os.path.basename(path))
        if not m:
            continue
        yield int(m.group(1)), path


def _load_json(path: str):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _resolve_masks_dir(data_dir: str, ann_dir: str, template_page: int) -> str:
    """Pick a mask folder, preferring the annotations-level path."""
    primary = os.path.join(ann_dir, "masks")
    legacy = os.path.join(data_dir, "masks")
    template_overlay = f"page_{template_page}_overlay.png"
    if os.path.exists(os.path.join(primary, template_overlay)):
        return primary
    if os.path.exists(os.path.join(legacy, template_overlay)):
        return legacy
    return primary


def _black_map(gray: np.ndarray) -> np.ndarray:
    # Binary "black pixel map" (255 = black-ish).
    # This aligns with how you paint masks (blue only when touching black pixels).
    return (gray < 80).astype(np.uint8) * 255


def _local_maxima(res: np.ndarray, threshold: float, max_peaks: int, min_dist: int) -> list[tuple[int, int, float]]:
    """Find local maxima in a matchTemplate response map.

    Returns a list of (x, y, score) in the response grid.

    Implementation:
    - Threshold first to keep things fast.
    - Use dilation to identify pixels that are equal to the neighborhood max.
    """
    if res.size == 0:
        return []
    mask = res >= threshold
    if not np.any(mask):
        return []

    k = max(1, int(min_dist))
    kernel = np.ones((k * 2 + 1, k * 2 + 1), dtype=np.uint8)
    dil = cv2.dilate(res, kernel)
    is_peak = (res == dil) & mask
    ys, xs = np.where(is_peak)
    peaks = [(int(x), int(y), float(res[y, x])) for y, x in zip(ys, xs)]
    peaks.sort(key=lambda p: p[2], reverse=True)
    return peaks[:max_peaks]


def _nms(boxes: list[tuple[int, int, int, int, float]], iou_thresh: float) -> list[tuple[int, int, int, int, float]]:
    """Very small NMS to de-duplicate overlapping detections.

    boxes: (x, y, w, h, score)
    """
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda b: b[4], reverse=True)
    kept: list[tuple[int, int, int, int, float]] = []

    def iou(a, b) -> float:
        ax1, ay1, aw, ah, _ = a
        bx1, by1, bw, bh, _ = b
        ax2, ay2 = ax1 + aw, ay1 + ah
        bx2, by2 = bx1 + bw, by1 + bh
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0:
            return 0.0
        ua = aw * ah + bw * bh - inter
        return float(inter) / float(max(ua, 1))

    for b in boxes:
        if all(iou(b, k) < iou_thresh for k in kept):
            kept.append(b)
    return kept


def _extract_template_pattern(
    data_dir: str,
    template_page: int,
    template_pair_index: int,
) -> tuple[np.ndarray, int, int]:
    """Return (templ_masked_black, templ_w, templ_h) at full resolution."""
    metadata = _load_json(os.path.join(data_dir, "metadata.json"))
    scale = float(metadata.get("scale", 4.1666666667))

    ann_path = os.path.join(os.path.dirname(__file__), "annotations", f"page_{template_page}.json")
    ann = _load_json(ann_path)
    pairs = ann.get("pairs", [])
    if template_pair_index >= len(pairs):
        raise SystemExit(f"Template pair index {template_pair_index} out of range for page {template_page}")

    bbox_pdf = pairs[template_pair_index]["component"]["bbox"]
    x0 = int(round(bbox_pdf[0] * scale))
    y0 = int(round(bbox_pdf[1] * scale))
    x1 = int(round(bbox_pdf[2] * scale))
    y1 = int(round(bbox_pdf[3] * scale))

    pages_dir = os.path.join(data_dir, "pages")
    ann_dir = os.path.join(os.path.dirname(__file__), "annotations")
    masks_dir = _resolve_masks_dir(data_dir, ann_dir, template_page)

    page_img_path = os.path.join(pages_dir, f"page_{template_page}.png")
    overlay_path = os.path.join(masks_dir, f"page_{template_page}_overlay.png")

    page = cv2.imread(page_img_path, cv2.IMREAD_COLOR)
    overlay = cv2.imread(overlay_path, cv2.IMREAD_UNCHANGED)
    if page is None:
        raise SystemExit(f"Could not read {page_img_path}")
    if overlay is None:
        raise SystemExit(f"Could not read {overlay_path}")
    if overlay.shape[0] != page.shape[0] or overlay.shape[1] != page.shape[1]:
        raise SystemExit("Overlay and page image dimensions do not match")

    # Clamp bbox
    h, w = page.shape[:2]
    x0 = max(0, min(w - 1, x0))
    x1 = max(0, min(w, x1))
    y0 = max(0, min(h - 1, y0))
    y1 = max(0, min(h, y1))
    if x1 <= x0 or y1 <= y0:
        raise SystemExit("Invalid template bbox")

    page_crop = page[y0:y1, x0:x1]
    overlay_crop = overlay[y0:y1, x0:x1]

    # Mask = where user painted (alpha > 0).
    if overlay_crop.shape[2] >= 4:
        alpha = overlay_crop[:, :, 3]
    else:
        # Fallback if somehow saved without alpha.
        alpha = cv2.cvtColor(overlay_crop, cv2.COLOR_BGR2GRAY)
    painted = (alpha > 0).astype(np.uint8) * 255

    gray = cv2.cvtColor(page_crop, cv2.COLOR_BGR2GRAY)
    black = _black_map(gray)

    # Template is exactly: (pixels you painted) AND (pixels that are black in the source page)
    templ = cv2.bitwise_and(black, black, mask=painted)

    nonzero = int(np.count_nonzero(templ))
    if nonzero < 25:
        raise SystemExit(
            f"Template has too few masked-black pixels ({nonzero}). Mask a bit more of the symbol."  # noqa: E501
        )

    th, tw = templ.shape[:2]
    return templ, tw, th


def _overlap_ratio(patch_black: np.ndarray, templ_black: np.ndarray) -> float:
    t = templ_black > 0
    if not np.any(t):
        return 0.0
    p = patch_black > 0
    overlap = np.logical_and(p, t).sum()
    total = t.sum()
    return float(overlap) / float(total)


def _best_match_for_page(
    page_black_full: np.ndarray,
    page_black_small: np.ndarray,
    templ_black_full: np.ndarray,
    scales: list[float],
    downscale: float,
    min_overlap: float,
    jitter: int,
) -> Match | None:
    # Candidate scan (small image) across all scales; keep the best few.
    best_coarse: tuple[float, float, tuple[int, int]] | None = None  # (score, scale, loc_small)

    for scale in scales:
        sw = max(5, int(round(templ_black_full.shape[1] * scale * downscale)))
        sh = max(5, int(round(templ_black_full.shape[0] * scale * downscale)))
        if sw >= page_black_small.shape[1] or sh >= page_black_small.shape[0]:
            continue
        templ_small = cv2.resize(templ_black_full, (sw, sh), interpolation=cv2.INTER_NEAREST)
        res = cv2.matchTemplate(page_black_small, templ_small, cv2.TM_CCORR_NORMED)
        _minv, maxv, _minloc, maxloc = cv2.minMaxLoc(res)
        if best_coarse is None or maxv > best_coarse[0]:
            best_coarse = (float(maxv), float(scale), (int(maxloc[0]), int(maxloc[1])))

    if best_coarse is None:
        return None

    # Refinement: test a small set of scales around best coarse scale.
    _, best_scale, best_loc_small = best_coarse
    refine_min = max(0.30, best_scale - 0.15)
    refine_max = best_scale + 0.15
    refine_scales = [round(s, 4) for s in np.arange(refine_min, refine_max + 1e-9, 0.02).tolist()]

    best: Match | None = None
    for scale in refine_scales:
        sw = max(5, int(round(templ_black_full.shape[1] * scale * downscale)))
        sh = max(5, int(round(templ_black_full.shape[0] * scale * downscale)))
        if sw >= page_black_small.shape[1] or sh >= page_black_small.shape[0]:
            continue
        templ_small = cv2.resize(templ_black_full, (sw, sh), interpolation=cv2.INTER_NEAREST)
        res = cv2.matchTemplate(page_black_small, templ_small, cv2.TM_CCORR_NORMED)
        _minv, _maxv, _minloc, maxloc = cv2.minMaxLoc(res)

        # Convert to full coords
        x0_full = int(round(maxloc[0] / downscale))
        y0_full = int(round(maxloc[1] / downscale))

        fw = int(round(templ_black_full.shape[1] * scale))
        fh = int(round(templ_black_full.shape[0] * scale))
        if fw < 5 or fh < 5:
            continue
        if fw >= page_black_full.shape[1] or fh >= page_black_full.shape[0]:
            continue

        templ_full = cv2.resize(templ_black_full, (fw, fh), interpolation=cv2.INTER_NEAREST)

        best_local_overlap = -1.0
        best_local_xy = (x0_full, y0_full)
        for dx in range(-jitter, jitter + 1):
            for dy in range(-jitter, jitter + 1):
                x = x0_full + dx
                y = y0_full + dy
                if x < 0 or y < 0 or x + fw > page_black_full.shape[1] or y + fh > page_black_full.shape[0]:
                    continue
                patch = page_black_full[y : y + fh, x : x + fw]
                ov = _overlap_ratio(patch, templ_full)
                if ov > best_local_overlap:
                    best_local_overlap = ov
                    best_local_xy = (x, y)

        if best_local_overlap >= min_overlap:
            x, y = best_local_xy
            m = Match(
                page=-1,
                overlap=float(best_local_overlap),
                scale=float(scale),
                x=int(x),
                y=int(y),
                w=int(fw),
                h=int(fh),
            )
            if best is None or m.overlap > best.overlap:
                best = m

    return best


def _find_matches_for_page(
    page_black_full: np.ndarray,
    page_black_small: np.ndarray,
    templ_black_full: np.ndarray,
    coarse_scales: list[float],
    downscale: float,
    min_overlap: float,
    jitter: int,
    coarse_score_thresh: float,
    max_candidates_per_scale: int,
    max_matches_per_page: int,
    nms_iou: float,
) -> list[Match]:
    """Return a list of verified matches for a page.

    We use downscaled matchTemplate to propose candidates, then verify with the
    overlap ratio at full resolution.
    """
    candidates: list[tuple[float, float, int, int, int, int]] = []
    # (coarse_score, scale, x_full, y_full, w_full, h_full)

    for scale in coarse_scales:
        sw = max(5, int(round(templ_black_full.shape[1] * scale * downscale)))
        sh = max(5, int(round(templ_black_full.shape[0] * scale * downscale)))
        if sw >= page_black_small.shape[1] or sh >= page_black_small.shape[0]:
            continue

        templ_small = cv2.resize(templ_black_full, (sw, sh), interpolation=cv2.INTER_NEAREST)
        res = cv2.matchTemplate(page_black_small, templ_small, cv2.TM_CCORR_NORMED)

        peaks = _local_maxima(res, coarse_score_thresh, max_candidates_per_scale, min_dist=6)
        if not peaks:
            continue

        fw = int(round(templ_black_full.shape[1] * scale))
        fh = int(round(templ_black_full.shape[0] * scale))
        if fw < 5 or fh < 5:
            continue
        if fw >= page_black_full.shape[1] or fh >= page_black_full.shape[0]:
            continue

        for (x_s, y_s, score) in peaks:
            x_full = int(round(x_s / downscale))
            y_full = int(round(y_s / downscale))
            candidates.append((float(score), float(scale), x_full, y_full, fw, fh))

    # Highest coarse-score candidates first.
    candidates.sort(key=lambda c: c[0], reverse=True)

    verified: list[Match] = []
    boxes_for_nms: list[tuple[int, int, int, int, float]] = []

    for coarse_score, scale, x0_full, y0_full, fw, fh in candidates:
        if len(verified) >= max_matches_per_page:
            break

        templ_full = cv2.resize(templ_black_full, (fw, fh), interpolation=cv2.INTER_NEAREST)

        best_local_overlap = -1.0
        best_local_xy = None
        for dx in range(-jitter, jitter + 1):
            for dy in range(-jitter, jitter + 1):
                x = x0_full + dx
                y = y0_full + dy
                if x < 0 or y < 0 or x + fw > page_black_full.shape[1] or y + fh > page_black_full.shape[0]:
                    continue
                patch = page_black_full[y : y + fh, x : x + fw]
                ov = _overlap_ratio(patch, templ_full)
                if ov > best_local_overlap:
                    best_local_overlap = ov
                    best_local_xy = (x, y)

        if best_local_xy is None:
            continue
        if best_local_overlap < min_overlap:
            continue

        x, y = best_local_xy
        boxes_for_nms.append((x, y, fw, fh, float(best_local_overlap)))

    # NMS in full-res coordinates.
    kept = _nms(boxes_for_nms, iou_thresh=nms_iou)
    for x, y, w, h, score in kept[:max_matches_per_page]:
        verified.append(Match(page=-1, overlap=score, scale=1.0, x=x, y=y, w=w, h=h))

    return verified


def main():
    ap = argparse.ArgumentParser(description="Multi-scale mask pattern search across pages")
    ap.add_argument("--data-dir", default=os.path.join(os.path.dirname(__file__), "data"))
    ap.add_argument("--template-page", type=int, default=7)
    ap.add_argument("--template-pair-index", type=int, default=0)
    ap.add_argument("--min-overlap", type=float, default=0.90)
    ap.add_argument("--downscale", type=float, default=0.25)
    ap.add_argument("--scale-min", type=float, default=0.60)
    ap.add_argument("--scale-max", type=float, default=1.60)
    ap.add_argument("--scale-step", type=float, default=0.10)
    ap.add_argument("--jitter", type=int, default=4)
    ap.add_argument("--max-pages", type=int, default=0, help="0=all")
    ap.add_argument("--out-dir", default=None, help="Directory for annotated PNGs and JSON")
    ap.add_argument("--label", default="", help="Label shown on bbox overlays")
    ap.add_argument("--max-matches-per-page", type=int, default=6)
    ap.add_argument("--max-candidates-per-scale", type=int, default=10)
    ap.add_argument("--coarse-score-thresh", type=float, default=0.55)
    ap.add_argument("--nms-iou", type=float, default=0.25)
    ap.add_argument("--draw", action="store_true", help="Save annotated images with bboxes")
    args = ap.parse_args()

    data_dir = args.data_dir
    pages_dir = os.path.join(data_dir, "pages")

    templ_black, tw, th = _extract_template_pattern(
        data_dir=data_dir,
        template_page=args.template_page,
        template_pair_index=args.template_pair_index,
    )

    scales = []
    s = args.scale_min
    while s <= args.scale_max + 1e-9:
        scales.append(round(s, 4))
        s += args.scale_step

    if args.out_dir is None:
        args.out_dir = os.path.join(data_dir, "masks", f"search_vis_page{args.template_page}_pair{args.template_pair_index}")
    _ensure_dir(args.out_dir)

    if not args.label:
        args.label = f"page{args.template_page}_pair{args.template_pair_index}"

    print(f"Template: page {args.template_page}, pair {args.template_pair_index}  size={tw}x{th}  coarse_scales={len(scales)}")
    print(f"Search: min_overlap={args.min_overlap:.2f} downscale={args.downscale} jitter={args.jitter}px")
    print(f"Output: {args.out_dir}")

    all_matches: list[dict] = []

    for n, (page_num, path) in enumerate(_iter_page_images(pages_dir), start=1):
        if args.max_pages and n > args.max_pages:
            break

        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        black_full = _black_map(gray)

        if args.downscale != 1.0:
            # IMPORTANT: downscale the *binary* black map, not grayscale.
            # Downscaling grayscale with INTER_AREA can blur thin lines into gray,
            # and then thresholding loses structure.
            black_small = cv2.resize(
                black_full,
                (0, 0),
                fx=args.downscale,
                fy=args.downscale,
                interpolation=cv2.INTER_NEAREST,
            )
        else:
            black_small = black_full

        page_matches = _find_matches_for_page(
            page_black_full=black_full,
            page_black_small=black_small,
            templ_black_full=templ_black,
            coarse_scales=scales,
            downscale=args.downscale,
            min_overlap=args.min_overlap,
            jitter=args.jitter,
            coarse_score_thresh=args.coarse_score_thresh,
            max_candidates_per_scale=args.max_candidates_per_scale,
            max_matches_per_page=args.max_matches_per_page,
            nms_iou=args.nms_iou,
        )

        if page_matches:
            for m in page_matches:
                m = Match(page=page_num, overlap=m.overlap, scale=m.scale, x=m.x, y=m.y, w=m.w, h=m.h)
                all_matches.append(m.__dict__)
                print(f"page_{page_num}: overlap={m.overlap:.3f} at ({m.x},{m.y}) size=({m.w}x{m.h})")

            # Draw bboxes on the original page image for human review.
            if args.draw:
                vis = img.copy()
                for m in page_matches:
                    cv2.rectangle(vis, (m.x, m.y), (m.x + m.w, m.y + m.h), (0, 255, 0), 2)
                    txt = f"{args.label} {m.overlap:.2f}"
                    cv2.putText(vis, txt, (m.x, max(0, m.y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                out_img = os.path.join(args.out_dir, f"page_{page_num}_detections.png")
                cv2.imwrite(out_img, vis)

        if n % 10 == 0:
            print(f"... scanned {n} pages")

    out_path = os.path.join(data_dir, "masks", "search_results_mcb3phase.json")
    out_path = os.path.join(args.out_dir, "search_results.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(all_matches, fh, indent=2)

    print(f"\nMatches >= {args.min_overlap:.2f}: {len(all_matches)}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
