"""fingerprint_distance_search.py

Distance-matrix fingerprint search (scale / rotation / mirror invariant).

What you proposed
-----------------
Pick N identifying pixels from a masked component (the "fingerprint points").
Compute the distances between those points.

Because pairwise distances are invariant to:
  - translation (moving the symbol)
  - rotation (turning the symbol)
  - mirroring (flipping the symbol)

and can be made invariant to scale by normalizing, this becomes a searchable
pattern for "any size" symbols.

This script implements that idea with a few practical adjustments:

1) Balanced point sampling
   -----------------------
We select N points spread across the mask using stratified sampling.
Default is 500 points for denser geometric description.
   That makes the signature stable and prevents points clustering in one corner.

2) Scale normalization
   -------------------
   We normalize all distances by the median of non-zero distances in the sample.
   This makes the signature scale-invariant.

3) Matching
   --------
   For a candidate region, we compute the same normalized distance signature.
   A candidate passes when a large fraction (default 90%) of points in each
   matched segment also pass the same relative-distance tolerance.

4) "Vector data" source
   ---------------------
   You asked to search "page 7 vector data". We do that by rasterizing the
   vector drawings stored in tools/annotator/data/vectors.db into a binary
   image per page, then cropping candidate bboxes from that vector raster.

   This keeps the detection pipeline grounded in vector geometry rather than
   the rendered PNG pixels (though both are aligned at 300 DPI).

Workflow
--------
A) Build template signature from page 7, pair 10 (ELB 3 Phase) mask overlay.
B) Search page 7 by scanning candidate component bboxes (from annotations).
C) If detected, search pages 7..40 the same way.

Outputs
-------
- Prints best matches.
- Writes JSON results.
- Optionally writes PNG images with bounding boxes drawn.

Run
---
  .venv\Scripts\python.exe tools\annotator\fingerprint_distance_search.py \
    --template-page 7 --template-pair 10 \
    --pages 7 40 --min-match 0.90 --draw
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sqlite3
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np


# -----------------------------------------------------------------------------
# Constants / coordinate systems
# -----------------------------------------------------------------------------
# Annotator uses 300 DPI page images. PDF units are points (1/72 inch).
VIEW_DPI = 300
PDF_TO_PX = VIEW_DPI / 72.0


# -----------------------------------------------------------------------------
# Data models
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class Detection:
    page: int
    pair: int
    label: str
    score: float
    bbox_pdf: Tuple[float, float, float, float]
    # Optional visualization image path (relative to tools/annotator/data).
    fingerprint_vis: Optional[str] = None


# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------

def _load_json(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _resolve_masks_dir(template_page: int, ann_dir: str, data_dir: str) -> str:
    """Pick a mask folder, preferring the annotations-level path."""
    primary = os.path.join(ann_dir, "masks")
    legacy = os.path.join(data_dir, "masks")
    template_overlay = f"page_{template_page}_overlay.png"
    if os.path.exists(os.path.join(primary, template_overlay)):
        return primary
    if os.path.exists(os.path.join(legacy, template_overlay)):
        return legacy
    return primary


def _pdf_bbox_to_px(bbox_pdf: List[float]) -> Tuple[int, int, int, int]:
    """Convert PDF bbox in points to pixel bbox in the 300DPI image space."""
    x0, y0, x1, y1 = map(float, bbox_pdf)
    return (
        int(round(x0 * PDF_TO_PX)),
        int(round(y0 * PDF_TO_PX)),
        int(round(x1 * PDF_TO_PX)),
        int(round(y1 * PDF_TO_PX)),
    )


def _clamp_bbox(b: Tuple[int, int, int, int], w: int, h: int) -> Tuple[int, int, int, int]:
    x0, y0, x1, y1 = b
    x0 = max(0, min(w - 1, x0))
    y0 = max(0, min(h - 1, y0))
    x1 = max(0, min(w, x1))
    y1 = max(0, min(h, y1))
    if x1 <= x0:
        x1 = min(w, x0 + 1)
    if y1 <= y0:
        y1 = min(h, y0 + 1)
    return x0, y0, x1, y1


def _relative_match_fraction(a: np.ndarray, b: np.ndarray, rel_tol: float) -> float:
    """Return fraction of elements within relative tolerance.

    We compare sorted normalized distance vectors element-wise.

    rel_err = |a-b| / max(a, 1e-9)

    This is robust against absolute scale changes.
    """
    if a.size == 0 or b.size == 0:
        return 0.0
    n = min(a.size, b.size)
    a = a[:n]
    b = b[:n]
    denom = np.maximum(a, 1e-9)
    rel_err = np.abs(a - b) / denom
    return float(np.mean(rel_err <= rel_tol))


# -----------------------------------------------------------------------------
# 1) Balanced point sampling from a binary mask
# -----------------------------------------------------------------------------

def _balanced_points_from_mask(mask: np.ndarray, n_points: int, grid: int = 7, seed: int = 0) -> np.ndarray:
    """Pick N points spread across the mask.

    mask: uint8, >0 means "foreground".

    Strategy: stratified sampling over a grid.
    - Split the mask bbox into grid x grid cells.
    - In each cell, if there are foreground pixels, pick one (pseudo-random).
    - Continue until N points are collected.

    Why this works:
    - Points are spread across the whole shape (balanced)
    - Stable across runs (seeded)

    Returns: array of shape (k,2) in (x,y) pixel coords.
    """
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return np.zeros((0, 2), dtype=np.int32)

    rng = np.random.default_rng(seed)

    # Heuristic: choose a grid size based on requested points.
    # For N=500 this yields a ~23x23 grid so points naturally cover whole-symbol area.
    if grid <= 0:
        grid = int(math.ceil(math.sqrt(max(1, n_points))))
    else:
        grid = max(grid, int(math.ceil(math.sqrt(max(1, n_points)))))

    h, w = mask.shape[:2]
    cell_w = max(1, w // grid)
    cell_h = max(1, h // grid)

    points: List[Tuple[int, int]] = []

    # Try to fill one per cell first.
    for gy in range(grid):
        for gx in range(grid):
            x0 = gx * cell_w
            y0 = gy * cell_h
            x1 = w if gx == grid - 1 else (gx + 1) * cell_w
            y1 = h if gy == grid - 1 else (gy + 1) * cell_h

            cell = mask[y0:y1, x0:x1]
            cys, cxs = np.where(cell > 0)
            if len(cxs) == 0:
                continue
            idx = int(rng.integers(0, len(cxs)))
            points.append((int(x0 + cxs[idx]), int(y0 + cys[idx])))

    # If not enough points, fill the rest from all foreground pixels.
    if len(points) < n_points:
        all_idx = rng.permutation(len(xs))
        for i in all_idx:
            if len(points) >= n_points:
                break
            points.append((int(xs[i]), int(ys[i])))

    # If too many points (grid filled a lot), subsample deterministically.
    if len(points) > n_points:
        idx = rng.permutation(len(points))[:n_points]
        points = [points[i] for i in idx]

    return np.array(points, dtype=np.int32)


def _segment_point_groups_from_mask(
    mask: np.ndarray,
    n_points: int,
    seed: int = 0,
) -> list[np.ndarray]:
    """Split mask into connected components and sample points per component.

    The 500-point fingerprint is distributed across components in proportion
    to component area so each structural segment contributes relative shape.
    """
    bin_mask = (mask > 0).astype(np.uint8)
    if bin_mask.size == 0:
        return []
    if not np.any(bin_mask):
        return []

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bin_mask, connectivity=8)
    if num_labels <= 1:
        return []

    comps: list[tuple[int, int]] = []
    for sid in range(1, num_labels):
        area = int(stats[sid, cv2.CC_STAT_AREA])
        if area > 0:
            comps.append((sid, area))

    if not comps:
        return []

    # Keep only the largest n components if we have too many tiny components.
    if len(comps) > n_points:
        comps = sorted(comps, key=lambda x: x[1], reverse=True)[:n_points]

    total_area = float(sum(area for _, area in comps))
    if total_area <= 0:
        return []

    comps = sorted(comps, key=lambda x: x[1], reverse=True)
    area_fracs = []
    n_comp = len(comps)
    base: list[int] = [0] * n_comp

    # Initial proportional allocation.
    for i, (_, area) in enumerate(comps):
        raw = (area / total_area) * float(max(1, n_points))
        base[i] = int(math.floor(raw))
        area_fracs.append((raw - base[i], i))

    remaining = n_points - sum(base)
    area_fracs.sort(key=lambda kv: kv[0], reverse=True)
    for i in range(max(0, remaining)):
        base[area_fracs[i % n_comp][1]] += 1

    # Ensure no zero-alloc component receives no sample points if n_points allows.
    if n_points >= n_comp:
        for i in range(n_comp):
            if base[i] == 0:
                base[i] = 1

    # Re-normalize if we accidentally allocated too many due to "ensure" rule.
    # Keep it deterministic and keep total exactly near n_points by trimming largest
    # buckets first.
    if sum(base) > n_points:
        surplus = sum(base) - n_points
        for i in sorted(range(n_comp), key=lambda i: base[i], reverse=True):
            take = min(surplus, base[i] - 1)
            if take <= 0:
                continue
            base[i] -= take
            surplus -= take
            if surplus <= 0:
                break

    point_groups: list[np.ndarray] = []
    for (sid, _), k in zip(comps, base):
        if k <= 0:
            continue
        comp_mask = np.where(labels == sid, 255, 0).astype(np.uint8)
        # Make sure each component contributes points from its own connected geometry.
        # Use per-component seed offset so component ordering is stable.
        pts = _balanced_points_from_mask(
            comp_mask,
            n_points=k,
            grid=0,
            seed=seed + sid * 97,
        )
        if pts.size > 0:
            point_groups.append(pts)

    return point_groups


def _segment_signatures_from_mask(mask: np.ndarray, n_points: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return per-segment sampled point clouds and their signatures."""
    groups = _segment_point_groups_from_mask(mask, n_points, seed=seed)
    out: list[tuple[np.ndarray, np.ndarray]] = []
    for pts in groups:
        if len(pts) < 3:
            continue
        sig = _distance_signature(pts)
        out.append((pts, sig))
    return out


def _segment_match_fraction(
    template_segments: list[tuple[np.ndarray, np.ndarray]],
    candidate_segments: list[tuple[np.ndarray, np.ndarray]],
    rel_tol: float,
    min_match: float,
) -> float:
    """Compare template and candidate signatures segment-by-segment.

    For each template segment, the best matching candidate segment is chosen.
    A point contributes positively only when that segment pair clears min_match.
    """
    if not template_segments:
        return 0.0

    # Largest segments first for stable greedy matching.
    tlist = sorted(
        template_segments,
        key=lambda item: len(item[0]),
        reverse=True,
    )
    clist = sorted(
        candidate_segments,
        key=lambda item: len(item[0]),
        reverse=True,
    )

    if not clist:
        return 0.0

    used = set()
    total_points = max(1, sum(len(pts) for pts, _ in tlist))
    passed_points = 0

    for ti, (t_pts, t_sig) in enumerate(tlist):
        if len(t_pts) < 3:
            continue

        best_score = -1.0
        best_ci = None
        for ci, (c_pts, c_sig) in enumerate(clist):
            if ci in used:
                continue
            if len(c_pts) < 3 and len(t_pts) < 3:
                score = 1.0
            else:
                score = _relative_match_fraction(t_sig, c_sig, rel_tol)
            if score > best_score:
                best_score = score
                best_ci = ci

        if best_ci is not None and best_score >= min_match:
            passed_points += len(t_pts)

        if best_ci is not None:
            used.add(best_ci)

    return float(passed_points) / float(total_points)


# -----------------------------------------------------------------------------
# 2) Fingerprint construction: normalized pairwise distances
# -----------------------------------------------------------------------------

def _distance_signature(points_xy: np.ndarray) -> np.ndarray:
    """Compute a scale-normalized, rotation/mirror invariant signature.

    We compute all pairwise Euclidean distances, normalize by median distance,
    and sort.

    - translation invariance: distances ignore translation
    - rotation/mirror invariance: distances ignore orientation
    - scale invariance: normalization

    Output: 1D float array of length n*(n-1)/2
    """
    n = points_xy.shape[0]
    if n < 3:
        return np.array([], dtype=np.float32)

    pts = points_xy.astype(np.float32)

    # Pairwise distances: vectorized via broadcasting.
    dx = pts[:, None, 0] - pts[None, :, 0]
    dy = pts[:, None, 1] - pts[None, :, 1]
    dist = np.sqrt(dx * dx + dy * dy)

    # Upper triangle (exclude diagonal)
    iu = np.triu_indices(n, k=1)
    v = dist[iu]

    # Normalize by median non-zero distance.
    v_nonzero = v[v > 1e-6]
    if v_nonzero.size == 0:
        return np.array([], dtype=np.float32)
    med = float(np.median(v_nonzero))
    v = v / max(med, 1e-6)

    # Sort so point ordering doesn't matter.
    v.sort()
    return v.astype(np.float32)


# -----------------------------------------------------------------------------
# 3) Vector rasterization from vectors.db
# -----------------------------------------------------------------------------

def _stroke_px(stroke_width_pdf: Optional[float]) -> int:
    """Convert PDF stroke width to pixel thickness.

    NOTE: We clamp thickness so we don't over-thicken thin strokes.
    This is used only to create a binary raster for point sampling.
    """
    if stroke_width_pdf is None:
        return 1
    try:
        px = int(round(float(stroke_width_pdf) * PDF_TO_PX))
    except (TypeError, ValueError):
        return 1
    return max(1, min(6, px))


def _bezier_points(p1, p2, p3, p4, steps: int) -> np.ndarray:
    """Sample a cubic Bézier into a polyline (approx)."""
    t = np.linspace(0.0, 1.0, steps + 1, dtype=np.float32)
    mt = 1.0 - t
    x = (
        mt * mt * mt * p1[0]
        + 3 * mt * mt * t * p2[0]
        + 3 * mt * t * t * p3[0]
        + t * t * t * p4[0]
    )
    y = (
        mt * mt * mt * p1[1]
        + 3 * mt * mt * t * p2[1]
        + 3 * mt * t * t * p3[1]
        + t * t * t * p4[1]
    )
    return np.stack([x, y], axis=1)


def _render_page_vectors_to_binary(con: sqlite3.Connection, page_num: int, out_size: Tuple[int, int]) -> np.ndarray:
    """Render all vector drawings on a page into a binary image.

    This converts vector data to a pixel grid so we can sample "identifying pixels"
    in candidate regions.

    out_size: (width_px, height_px)

    Returns: uint8 image, 255 = stroke pixel.
    """
    w_px, h_px = out_size
    canvas = np.zeros((h_px, w_px), dtype=np.uint8)

    cur = con.cursor()
    page_id = cur.execute("SELECT page_id FROM pages WHERE page_num=?", (page_num,)).fetchone()[0]

    # Pull drawings + items in one stream so this stays fast.
    # We rely on drawing_id ordering to group items.
    rows = cur.execute(
        """
        SELECT d.drawing_id, d.stroke_width,
               i.kind, i.x1,i.y1,i.x2,i.y2,i.x3,i.y3,i.x4,i.y4
        FROM drawings d
        JOIN drawing_items i ON i.drawing_id = d.drawing_id
        WHERE d.page_id=?
        ORDER BY d.drawing_id
        """,
        (page_id,),
    )

    def to_px(x_pdf: float, y_pdf: float) -> Tuple[int, int]:
        return (int(round(x_pdf * PDF_TO_PX)), int(round(y_pdf * PDF_TO_PX)))

    current_id = None
    current_thickness = 1

    for drawing_id, stroke_width, kind, x1, y1, x2, y2, x3, y3, x4, y4 in rows:
        if current_id != drawing_id:
            current_id = drawing_id
            current_thickness = _stroke_px(stroke_width)

        if kind == "l" and x1 is not None and x2 is not None:
            p1 = to_px(float(x1), float(y1))
            p2 = to_px(float(x2), float(y2))
            cv2.line(canvas, p1, p2, 255, thickness=current_thickness, lineType=cv2.LINE_8)

        elif kind == "re" and x1 is not None and x2 is not None:
            p1 = to_px(float(x1), float(y1))
            p2 = to_px(float(x2), float(y2))
            cv2.rectangle(canvas, p1, p2, 255, thickness=current_thickness)

        elif kind == "c" and x1 is not None and x4 is not None:
            p1 = (float(x1), float(y1))
            p2 = (float(x2), float(y2))
            p3 = (float(x3), float(y3))
            p4 = (float(x4), float(y4))

            approx_len = abs(p1[0] - p4[0]) + abs(p1[1] - p4[1])
            steps = int(max(12, min(80, approx_len * 6)))
            pts_pdf = _bezier_points(p1, p2, p3, p4, steps)
            pts_px = np.array([to_px(x, y) for x, y in pts_pdf], dtype=np.int32)
            cv2.polylines(canvas, [pts_px], False, 255, thickness=current_thickness, lineType=cv2.LINE_8)

    return canvas


# -----------------------------------------------------------------------------
# 4) Template creation from mask overlay
# -----------------------------------------------------------------------------

def _template_signature_from_mask(
    overlay_rgba: np.ndarray,
    comp_bbox_px: Tuple[int, int, int, int],
    n_points: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract N points from mask overlay (alpha>0) and build signature.

    Returns: (points_xy, signature)
    """
    x0, y0, x1, y1 = comp_bbox_px
    crop = overlay_rgba[y0:y1, x0:x1]
    alpha = crop[:, :, 3]

    mask = (alpha > 0).astype(np.uint8) * 255
    pts = _balanced_points_from_mask(mask, n_points=n_points, grid=7, seed=seed)
    sig = _distance_signature(pts)
    return pts, sig


# -----------------------------------------------------------------------------
# 5) Candidate evaluation
# -----------------------------------------------------------------------------

def _candidate_signature_from_vector_raster(
    vector_bin: np.ndarray,
    bbox_px: Tuple[int, int, int, int],
    n_points: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample N points from a candidate bbox (from vector raster) and signature.

    Returns both:
      - pts: (N,2) pixel coords (bbox-local)
      - sig: normalized pairwise distance signature
    """
    x0, y0, x1, y1 = bbox_px
    patch = vector_bin[y0:y1, x0:x1]
    pts = _balanced_points_from_mask(patch, n_points=n_points, grid=7, seed=seed)
    sig = _distance_signature(pts)
    return pts, sig


def _draw_fingerprint_starburst(
    base_bgr: np.ndarray,
    points_xy: np.ndarray,
    color: Tuple[int, int, int] = (255, 0, 255),
) -> np.ndarray:
    """Draw a 1px-wide starburst from the identifier pixel to all anchors.

    User request:
      - choose one identifier pixel
      - draw a 1-pixel-wide line from it to each of the other anchor pixels

    We pick identifier = points_xy[0] (stable under our sampling seed).
    """
    if base_bgr is None:
        return base_bgr
    if points_xy is None or points_xy.shape[0] < 2:
        return base_bgr

    out = base_bgr.copy()
    x0, y0 = int(points_xy[0, 0]), int(points_xy[0, 1])
    for i in range(1, points_xy.shape[0]):
        x1, y1 = int(points_xy[i, 0]), int(points_xy[i, 1])
        cv2.line(out, (x0, y0), (x1, y1), color, thickness=1, lineType=cv2.LINE_8)

    # Mark identifier pixel (single-pixel dot)
    if 0 <= y0 < out.shape[0] and 0 <= x0 < out.shape[1]:
        out[y0, x0] = color
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--template-page", type=int, default=7)
    ap.add_argument("--template-pair", type=int, default=10)
    ap.add_argument("--pages", nargs=2, type=int, default=[7, 40], help="start end")
    ap.add_argument("--n-points", type=int, default=500)
    ap.add_argument("--min-match", type=float, default=0.90)
    ap.add_argument("--rel-tol", type=float, default=0.04, help="relative tolerance per distance")
    ap.add_argument("--seeds", type=int, default=3, help="try multiple seeds and take best")
    ap.add_argument("--draw", action="store_true")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(here, "data")
    pages_dir = os.path.join(data_dir, "pages")
    ann_dir = os.path.join(here, "annotations")
    masks_dir = _resolve_masks_dir(args.template_page, ann_dir, data_dir)
    db_path = os.path.join(data_dir, "vectors.db")

    out_dir = os.path.join(
        masks_dir,
        f"distance_fp_vis_page{args.template_page}_pair{args.template_pair}_n{args.n_points}",
    )
    _ensure_dir(out_dir)

    # --- Load template annotation + overlay ---
    ann_path = os.path.join(ann_dir, f"page_{args.template_page}.json")
    ann = _load_json(ann_path)
    pairs = ann.get("pairs", [])
    if args.template_pair >= len(pairs):
        raise SystemExit("template pair out of range")

    template_label = pairs[args.template_pair]["label"]["text"]
    template_bbox_pdf = pairs[args.template_pair]["component"]["bbox"]

    overlay_path = os.path.join(masks_dir, f"page_{args.template_page}_overlay.png")
    overlay = cv2.imread(overlay_path, cv2.IMREAD_UNCHANGED)
    if overlay is None or overlay.ndim != 3 or overlay.shape[2] < 4:
        raise SystemExit(f"Overlay missing or not RGBA: {overlay_path}")

    # --- Template signature (try multiple seeds; keep first that works) ---
    template_bbox_px = _clamp_bbox(_pdf_bbox_to_px(template_bbox_pdf), overlay.shape[1], overlay.shape[0])

    template_points = None
    template_sig = None
    template_seed = None

    # For templates, we keep the first seed (stable), but also ensure it works.
    for seed in range(args.seeds):
        pts, sig = _template_signature_from_mask(overlay, template_bbox_px, args.n_points, seed=seed)
        if sig.size > 0 and pts.shape[0] >= 10:
            template_points, template_sig = pts, sig
            template_seed = seed
            break

    if template_sig is None or template_sig.size == 0:
        raise SystemExit("Template signature could not be built (mask too sparse?)")

    x0t, y0t, x1t, y1t = template_bbox_px
    template_mask = (overlay[y0t:y1t, x0t:x1t, 3] > 0).astype(np.uint8) * 255
    template_segments = _segment_signatures_from_mask(template_mask, args.n_points, seed=template_seed if template_seed is not None else 0)

    print(f"Template: page {args.template_page} pair {args.template_pair} label={template_label}")
    print(f"Template bbox (PDF): {template_bbox_pdf}")
    print(f"Template points: {template_points.shape[0]}  signature length: {template_sig.size}")

    # Save a template fingerprint visualization crop for reference.
    if args.draw:
        page_img_path = os.path.join(pages_dir, f"page_{args.template_page}.png")
        page_img = cv2.imread(page_img_path, cv2.IMREAD_COLOR)
        if page_img is not None:
            tx0, ty0, tx1, ty1 = template_bbox_px
            crop = page_img[ty0:ty1, tx0:tx1]
            vis = _draw_fingerprint_starburst(crop, template_points, color=(255, 0, 255))
            out_path = os.path.join(out_dir, f"template_fingerprint_starburst_seed{template_seed}.png")
            cv2.imwrite(out_path, vis)

    # --- Open DB and get page render sizes from metadata ---
    meta = _load_json(os.path.join(data_dir, "metadata.json"))
    display_w, display_h = map(int, meta["pages"][0]["display_size"])  # all pages same for this doc

    con = sqlite3.connect(db_path)

    # Global list across all scanned pages (used for final JSON output).
    detections: List[Detection] = []

    start_page, end_page = args.pages

    # --- Page loop ---
    for page_num in range(start_page, end_page + 1):
        # Detections for the *current* page only.
        # This is important for correctness + clarity: pages can have many
        # instances (100+) and we want to draw/report all of them.
        page_detections: List[Detection] = []

        # Render page vector content once.
        vector_bin = _render_page_vectors_to_binary(con, page_num, out_size=(display_w, display_h))

        # Load page annotations as candidate regions.
        page_ann_path = os.path.join(ann_dir, f"page_{page_num}.json")
        if not os.path.exists(page_ann_path):
            continue
        page_ann = _load_json(page_ann_path)
        page_pairs = page_ann.get("pairs", [])

        best_for_page: Optional[Detection] = None

        for pair_idx, p in enumerate(page_pairs):
            if p.get("type") == "continuation":
                continue
            bbox_pdf = p["component"]["bbox"]
            bbox_px = _clamp_bbox(_pdf_bbox_to_px(bbox_pdf), display_w, display_h)

            # Candidate match: try a few seeds and take the best score.
            best_score = 0.0
            best_seed = None
            best_points = None
            for seed in range(args.seeds):
                pts, cand_sig = _candidate_signature_from_vector_raster(vector_bin, bbox_px, args.n_points, seed=seed)
                if cand_sig.size == 0:
                    continue
                x0, y0, x1, y1 = bbox_px
                patch = vector_bin[y0:y1, x0:x1]
                cand_segments = _segment_signatures_from_mask(patch, args.n_points, seed=seed)
                if template_segments:
                    # New matching mode: require minimum segment-wise confidence.
                    score = _segment_match_fraction(
                        template_segments,
                        cand_segments,
                        rel_tol=args.rel_tol,
                        min_match=args.min_match,
                    )
                else:
                    # Backward-compatible path if segmentation fails.
                    score = _relative_match_fraction(template_sig, cand_sig, rel_tol=args.rel_tol)
                if score > best_score:
                    best_score = score
                    best_seed = seed
                    best_points = pts

            if best_score >= args.min_match:
                fp_vis_rel = None
                if args.draw and best_points is not None:
                    # Save a bbox-local visualization crop showing the fingerprint.
                    img_path = os.path.join(pages_dir, f"page_{page_num}.png")
                    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                    if img is not None:
                        x0, y0, x1, y1 = bbox_px
                        crop = img[y0:y1, x0:x1]
                        vis = _draw_fingerprint_starburst(crop, best_points, color=(255, 0, 255))
                        fname = f"fp_page_{page_num}_pair_{pair_idx}_seed{best_seed}_score{best_score:.3f}.png"
                        abs_out = os.path.join(out_dir, fname)
                        cv2.imwrite(abs_out, vis)
                        # Store relative to tools/annotator/data so the app can find it.
                        fp_vis_rel = os.path.relpath(abs_out, data_dir).replace("\\", "/")

                det = Detection(
                    page=page_num,
                    pair=pair_idx,
                    label=p["label"]["text"],
                    score=best_score,
                    bbox_pdf=(float(bbox_pdf[0]), float(bbox_pdf[1]), float(bbox_pdf[2]), float(bbox_pdf[3])),
                    fingerprint_vis=fp_vis_rel,
                )
                # Store both globally and per-page.
                detections.append(det)
                page_detections.append(det)
                if best_for_page is None or det.score > best_for_page.score:
                    best_for_page = det

        # Optional visualization per page
        if args.draw and page_detections:
            img_path = os.path.join(pages_dir, f"page_{page_num}.png")
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if img is not None:
                # Draw ALL detections on this page.
                # NOTE: a page can legitimately have 100+ instances; this will
                # draw 100+ boxes.
                for det in page_detections:
                    bx0, by0, bx1, by1 = _pdf_bbox_to_px(list(det.bbox_pdf))
                    bx0, by0, bx1, by1 = _clamp_bbox((bx0, by0, bx1, by1), img.shape[1], img.shape[0])
                    cv2.rectangle(img, (bx0, by0), (bx1, by1), (0, 255, 0), 2)
                    cv2.putText(
                        img,
                        f"{template_label} {det.score:.2f}",
                        (bx0, max(0, by0 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                    )
                cv2.imwrite(os.path.join(out_dir, f"page_{page_num}_fp_detections.png"), img)

        # Always print a per-page summary if anything was detected.
        if page_detections:
            best = max(page_detections, key=lambda d: d.score)
            print(
                f"page_{page_num}: {len(page_detections)} match(es) >= {args.min_match:.2f} "
                f"(best score={best.score:.3f}, pair={best.pair}, label={best.label})"
            )
        elif page_num == args.template_page:
            # Template page should almost always detect itself.
            print(f"page_{page_num}: no matches >= {args.min_match:.2f}")

        if page_num % 5 == 0:
            print(f"... scanned page {page_num}")

    con.close()

    # Save results
    out_json = os.path.join(out_dir, "fingerprint_distance_results.json")
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump([d.__dict__ for d in detections], fh, indent=2)

    print(f"\nTotal detections >= {args.min_match:.2f}: {len(detections)}")
    print(f"Saved: {out_json}")


if __name__ == "__main__":
    main()
