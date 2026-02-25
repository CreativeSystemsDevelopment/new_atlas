"""extract_mask_fingerprint.py

Extract a *vector fingerprint* for a masked component.

Goal
----
Given:
  - a page number
  - a component pair index on that page (from annotations)
  - a mask overlay PNG (blue pixels with alpha)
  - a SQLite DB of PDF vector drawings (vectors.db)

We want to identify which vector drawings (from `page.get_drawings()`) are most
likely part of the *symbol you masked*.

Why not just "bbox overlap"?
----------------------------
A plain bbox-overlap query is a good *candidate generator*, but it includes
wires, nearby text, and anything else that intersects the component rectangle.

This script uses your intended technique:
  1) Use the mask overlay to find the *painted pixel region*.
  2) Query the R-Tree for vector drawings that overlap that painted region.
  3) For each candidate drawing:
       - rasterize its primitives into an image at the same resolution
       - compute how many painted mask pixels it intersects
  4) Rank drawings by "mask coverage".

Important: "only" mask intersection (without ANY candidate step)
---------------------------------------------------------------
In theory you could test every drawing against every mask pixel.
In practice we have ~67k drawings across the document.

So we still use an overlap query as a *performance optimization*.
To avoid missing strokes, we do NOT use the annotation bbox as the query box;
we use the *painted mask bbox* (optionally with a margin).

This guarantees:
  - If you painted over a stroke, that stroke must overlap the painted bbox,
    and therefore can be found by the candidate query.

Dependencies
------------
- OpenCV (cv2)
- numpy
- sqlite3

Run
---
  .venv\Scripts\python.exe tools\annotator\extract_mask_fingerprint.py --page 7 --pair 10

Outputs
-------
- Prints a ranked list of drawing_ids and coverage
- Writes JSON to the active masks folder (preferred: tools/annotator/annotations/masks).
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np


# The annotator renders pages at 300 DPI.
# PDF coords are points (1/72 inch). So pixels = pdf_points * (300/72).
VIEW_DPI = 300
PDF_TO_PX = VIEW_DPI / 72.0


@dataclass(frozen=True)
class CandidateResult:
    drawing_id: int
    seqno: Optional[int]
    bbox_pdf: Tuple[float, float, float, float]
    stroke_width_pdf: Optional[float]
    mask_pixels: int
    hit_pixels: int
    coverage: float


def _load_json(path: str) -> dict:
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


def _pdf_bbox_to_px(bbox_pdf: List[float]) -> Tuple[int, int, int, int]:
    x0, y0, x1, y1 = bbox_pdf
    return (
        int(round(x0 * PDF_TO_PX)),
        int(round(y0 * PDF_TO_PX)),
        int(round(x1 * PDF_TO_PX)),
        int(round(y1 * PDF_TO_PX)),
    )


def _px_bbox_to_pdf(px_bbox: Tuple[int, int, int, int]) -> Tuple[float, float, float, float]:
    x0, y0, x1, y1 = px_bbox
    return (
        float(x0) / PDF_TO_PX,
        float(y0) / PDF_TO_PX,
        float(x1) / PDF_TO_PX,
        float(y1) / PDF_TO_PX,
    )


def _tight_bbox_from_mask(mask: np.ndarray) -> Tuple[int, int, int, int]:
    """Return tight bbox (x0,y0,x1,y1) in *mask-local* pixel coords."""
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        raise ValueError("No painted pixels found in mask crop")
    x0, x1 = int(xs.min()), int(xs.max() + 1)
    y0, y1 = int(ys.min()), int(ys.max() + 1)
    return x0, y0, x1, y1


def _bezier_points(p1, p2, p3, p4, steps: int) -> List[Tuple[float, float]]:
    """Sample a cubic Bézier curve as a polyline.

    We purposely keep this simple. The goal is not perfect vector rendering;
    it is to compute *intersection* with your mask pixels.
    """
    pts = []
    for i in range(steps + 1):
        t = i / steps
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
        pts.append((x, y))
    return pts


def _stroke_px(stroke_width_pdf: Optional[float]) -> int:
    """Convert PDF stroke width to pixels, with safe clamping.

    If we draw with a stroke that's too fat, we will over-intersect.
    If we draw too thin, we might miss intersections.

    The clamp keeps results stable.
    """
    if stroke_width_pdf is None:
        return 1
    try:
        px = int(round(float(stroke_width_pdf) * PDF_TO_PX))
    except (TypeError, ValueError):
        return 1
    return max(1, min(8, px))


def _rasterize_drawing_items_to_patch(
    items: List[Tuple],
    patch_shape: Tuple[int, int],
    patch_origin_pdf: Tuple[float, float],
    thickness_px: int,
) -> np.ndarray:
    """Rasterize vector primitives into a binary mask patch.

    Parameters
    ----------
    items:
        Drawing primitives from DB (decoded into tuples).
    patch_shape:
        (h, w) of the output patch (pixels).
    patch_origin_pdf:
        (x0_pdf, y0_pdf) of the patch top-left.
    thickness_px:
        Line thickness for rasterization.

    Returns
    -------
    uint8 image: 0 background, 255 for drawn pixels.

    Note
    ----
    This is an *approximation* to support intersection scoring.
    It's not intended as a perfect renderer.
    """
    h, w = patch_shape
    out = np.zeros((h, w), dtype=np.uint8)

    ox_pdf, oy_pdf = patch_origin_pdf

    def to_px(x_pdf: float, y_pdf: float) -> Tuple[int, int]:
        x = int(round((x_pdf - ox_pdf) * PDF_TO_PX))
        y = int(round((y_pdf - oy_pdf) * PDF_TO_PX))
        return x, y

    for it in items:
        if not it:
            continue
        kind = it[0]

        if kind == "l" and len(it) >= 5:
            # Stored as: ('l', x1,y1, x2,y2)
            x1, y1, x2, y2 = map(float, it[1:5])
            p1 = to_px(x1, y1)
            p2 = to_px(x2, y2)
            cv2.line(out, p1, p2, 255, thickness=thickness_px, lineType=cv2.LINE_8)

        elif kind == "re" and len(it) >= 5:
            # Stored as: ('re', x0,y0, x1,y1)
            x0, y0, x1, y1 = map(float, it[1:5])
            p1 = to_px(x0, y0)
            p2 = to_px(x1, y1)
            cv2.rectangle(out, p1, p2, 255, thickness=thickness_px)

        elif kind == "c" and len(it) >= 9:
            # Stored as: ('c', x1,y1, x2,y2, x3,y3, x4,y4)
            p1 = (float(it[1]), float(it[2]))
            p2 = (float(it[3]), float(it[4]))
            p3 = (float(it[5]), float(it[6]))
            p4 = (float(it[7]), float(it[8]))

            # Sampling steps: more steps for longer curves.
            approx_len = abs(p1[0] - p4[0]) + abs(p1[1] - p4[1])
            steps = int(max(10, min(60, approx_len * 6)))
            pts_pdf = _bezier_points(p1, p2, p3, p4, steps)
            pts_px = [to_px(x, y) for (x, y) in pts_pdf]
            cv2.polylines(out, [np.array(pts_px, dtype=np.int32)], False, 255, thickness=thickness_px)

        else:
            # Unknown or unhandled primitive type.
            continue

    return out


def _query_candidates_by_painted_bbox(
    con: sqlite3.Connection,
    page_num: int,
    painted_bbox_pdf: Tuple[float, float, float, float],
) -> List[Tuple[int, Optional[int], float, float, float, float, Optional[float]]]:
    """Return candidate drawings overlapping the painted bbox.

    Returns tuples:
      (drawing_id, seqno, x0, y0, x1, y1, stroke_width)
    """
    cur = con.cursor()
    page_id = cur.execute("SELECT page_id FROM pages WHERE page_num=?", (page_num,)).fetchone()[0]

    qx0, qy0, qx1, qy1 = painted_bbox_pdf

    # Overlap query using the R-Tree.
    # R-Tree columns are minX,maxX,minY,maxY.
    rows = cur.execute(
        """
        SELECT d.drawing_id, d.seqno, d.x0, d.y0, d.x1, d.y1, d.stroke_width
        FROM drawings_rtree r
        JOIN drawings d ON d.drawing_id = r.drawing_id
        WHERE d.page_id=?
          AND r.x0 <= ? AND r.x1 >= ?
          AND r.y0 <= ? AND r.y1 >= ?
        ORDER BY COALESCE(d.seqno, 1e9), d.drawing_id
        """,
        (page_id, qx1, qx0, qy1, qy0),
    ).fetchall()

    return rows


def _get_drawing_items(con: sqlite3.Connection, drawing_id: int) -> List[Tuple]:
    """Load drawing_items from DB and convert to lightweight tuples.

    We store items in normalized columns. Here we reconstruct a compact tuple:
      - 'l': ('l', x1,y1,x2,y2)
      - 'c': ('c', x1,y1,x2,y2,x3,y3,x4,y4)
      - 're': ('re', x0,y0,x1,y1)

    Unknown items are ignored for now.
    """
    cur = con.cursor()
    items = []
    for row in cur.execute(
        """SELECT kind, x1,y1,x2,y2,x3,y3,x4,y4
           FROM drawing_items WHERE drawing_id=?""",
        (drawing_id,),
    ):
        kind = row[0]
        coords = row[1:]
        if kind == "l":
            items.append(("l", coords[0], coords[1], coords[2], coords[3]))
        elif kind == "re":
            items.append(("re", coords[0], coords[1], coords[2], coords[3]))
        elif kind == "c":
            items.append(("c", coords[0], coords[1], coords[2], coords[3], coords[4], coords[5], coords[6], coords[7]))
        else:
            continue
    return items


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract vector fingerprint using mask pixel intersection")
    ap.add_argument("--page", type=int, default=7)
    ap.add_argument("--pair", type=int, default=10, help="pair index on that page")
    ap.add_argument("--mask-alpha-thresh", type=int, default=1)
    ap.add_argument("--dilate", type=int, default=1, help="dilate mask by N pixels for tolerance (0 disables)")
    ap.add_argument("--margin-px", type=int, default=2, help="expand painted bbox by this many pixels")
    ap.add_argument("--top", type=int, default=25)
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(here, "data")
    ann_dir = os.path.join(here, "annotations")
    masks_dir = _resolve_masks_dir(data_dir, ann_dir, args.page)
    pages_dir = os.path.join(data_dir, "pages")
    db_path = os.path.join(data_dir, "vectors.db")

    overlay_path = os.path.join(masks_dir, f"page_{args.page}_overlay.png")
    ann_path = os.path.join(ann_dir, f"page_{args.page}.json")

    if not os.path.exists(overlay_path):
        raise SystemExit(f"Missing overlay: {overlay_path}")
    if not os.path.exists(ann_path):
        raise SystemExit(f"Missing annotations: {ann_path}")
    if not os.path.exists(db_path):
        raise SystemExit(f"Missing vectors DB: {db_path}")

    ann = _load_json(ann_path)
    pairs = ann.get("pairs", [])
    if args.pair >= len(pairs):
        raise SystemExit(f"pair index {args.pair} out of range for page {args.page} ({len(pairs)} pairs)")

    label = pairs[args.pair]["label"]["text"]
    comp_bbox_pdf = pairs[args.pair]["component"]["bbox"]

    # Load overlay and crop to the component bbox.
    overlay = cv2.imread(overlay_path, cv2.IMREAD_UNCHANGED)
    if overlay is None or overlay.ndim != 3 or overlay.shape[2] < 4:
        raise SystemExit("Overlay must be RGBA (with alpha channel)")

    x0, y0, x1, y1 = _pdf_bbox_to_px(comp_bbox_pdf)
    x0 = max(0, min(overlay.shape[1] - 1, x0))
    x1 = max(0, min(overlay.shape[1], x1))
    y0 = max(0, min(overlay.shape[0] - 1, y0))
    y1 = max(0, min(overlay.shape[0], y1))

    crop = overlay[y0:y1, x0:x1]
    alpha = crop[:, :, 3]

    # Binary painted mask in component-local coords.
    mask = (alpha >= args.mask_alpha_thresh).astype(np.uint8) * 255

    if args.dilate > 0:
        # Dilation gives intersection tolerance, so we don't miss strokes due
        # to 1px rasterization differences.
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (args.dilate * 2 + 1, args.dilate * 2 + 1))
        mask = cv2.dilate(mask, k)

    # Tight bbox of painted pixels in component-local coords
    mx0, my0, mx1, my1 = _tight_bbox_from_mask(mask)

    # Expand by margin (in component-local coords)
    mx0 = max(0, mx0 - args.margin_px)
    my0 = max(0, my0 - args.margin_px)
    mx1 = min(mask.shape[1], mx1 + args.margin_px)
    my1 = min(mask.shape[0], my1 + args.margin_px)

    # Convert painted bbox to full-page pixel coords
    painted_px_bbox_full = (x0 + mx0, y0 + my0, x0 + mx1, y0 + my1)
    painted_pdf_bbox = _px_bbox_to_pdf(painted_px_bbox_full)

    # Crop the mask to the painted bbox for scoring.
    mask_patch = mask[my0:my1, mx0:mx1]
    mask_pixels = int(np.count_nonzero(mask_patch))

    print(f"Page {args.page} pair {args.pair}: {label}")
    print(f"Component bbox (PDF): {comp_bbox_pdf}")
    print(f"Painted bbox (PDF): {tuple(round(v, 2) for v in painted_pdf_bbox)}")
    print(f"Mask pixels (painted): {mask_pixels}")

    # Candidate query
    con = sqlite3.connect(db_path)
    candidates = _query_candidates_by_painted_bbox(con, args.page, painted_pdf_bbox)
    print(f"Candidate drawings (R-Tree overlap): {len(candidates)}")

    # Rasterize each candidate and compute intersection
    patch_h, patch_w = mask_patch.shape[:2]
    patch_origin_pdf = (painted_pdf_bbox[0], painted_pdf_bbox[1])

    results: List[CandidateResult] = []

    for drawing_id, seqno, dx0, dy0, dx1, dy1, stroke_w in candidates:
        items = _get_drawing_items(con, int(drawing_id))
        if not items:
            continue

        thickness = _stroke_px(stroke_w)
        drawing_patch = _rasterize_drawing_items_to_patch(
            items=items,
            patch_shape=(patch_h, patch_w),
            patch_origin_pdf=patch_origin_pdf,
            thickness_px=thickness,
        )

        # Compute how much of the mask this drawing explains.
        hit = int(np.count_nonzero((drawing_patch > 0) & (mask_patch > 0)))
        cov = float(hit) / float(mask_pixels) if mask_pixels else 0.0

        if hit == 0:
            continue

        results.append(
            CandidateResult(
                drawing_id=int(drawing_id),
                seqno=int(seqno) if seqno is not None else None,
                bbox_pdf=(float(dx0), float(dy0), float(dx1), float(dy1)),
                stroke_width_pdf=float(stroke_w) if stroke_w is not None else None,
                mask_pixels=mask_pixels,
                hit_pixels=hit,
                coverage=cov,
            )
        )

    con.close()

    results.sort(key=lambda r: (r.coverage, r.hit_pixels), reverse=True)

    print("\nTop drawings by mask coverage:")
    for r in results[: args.top]:
        bx0, by0, bx1, by1 = r.bbox_pdf
        print(
            f"  id={r.drawing_id} seqno={r.seqno} cov={r.coverage:.3f} "
            f"hit={r.hit_pixels}/{r.mask_pixels} stroke_w={r.stroke_width_pdf} "
            f"bbox=({bx0:.1f},{by0:.1f},{bx1:.1f},{by1:.1f})"
        )

    # Combined coverage: OR all selected drawing patches.
    # This tells you whether the mask is explained by a small number of drawings
    # (good) or spread across many (likely wires/noise).
    combined = np.zeros_like(mask_patch)

    # Re-open DB for reconstruction (we closed it above).
    con = sqlite3.connect(db_path)
    for r in results[: args.top]:
        items = _get_drawing_items(con, r.drawing_id)
        thickness = _stroke_px(r.stroke_width_pdf)
        dp = _rasterize_drawing_items_to_patch(items, (patch_h, patch_w), patch_origin_pdf, thickness)
        combined = cv2.bitwise_or(combined, dp)
    con.close()

    combined_hit = int(np.count_nonzero((combined > 0) & (mask_patch > 0)))
    combined_cov = float(combined_hit) / float(mask_pixels) if mask_pixels else 0.0
    print(f"\nCombined coverage of top {min(args.top, len(results))}: {combined_cov:.3f} ({combined_hit}/{mask_pixels})")

    out_path = os.path.join(masks_dir, f"fingerprint_page_{args.page}_pair_{args.pair}.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "page": args.page,
                "pair": args.pair,
                "label": label,
                "component_bbox_pdf": comp_bbox_pdf,
                "painted_bbox_pdf": painted_pdf_bbox,
                "mask_pixels": mask_pixels,
                "top": args.top,
                "combined_top_coverage": combined_cov,
                "results": [r.__dict__ for r in results],
            },
            fh,
            indent=2,
        )

    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
