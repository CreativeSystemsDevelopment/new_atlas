"""build_vector_db.py

Builds a fast, queryable SQLite database of *PDF vector* content.

Why this exists
---------------
The schematic pages are PDFs containing vector strokes (lines/curves/rects).
PyMuPDF exposes these as `page.get_drawings()`.

For fingerprinting a masked component later, we want to answer questions like:

    - "Which vector drawings overlap this component bbox?" (fast spatial query)
    - "What are the individual primitives (line/curve/rect) that make up it?"

Storing everything in SQLite gives us:
    - A single file we can ship/backup (`vectors.db`)
    - Fast bbox queries via an R-Tree index
    - Simple joins from drawings -> items

Design notes
------------
* Do NOT create one table per page; instead store `page_id` on each row.
    This keeps schema stable and queries simple.
* The R-Tree virtual table uses **(minX, maxX, minY, maxY)** ordering.
    This is easy to get wrong and will throw constraint errors.
* We keep a `raw_json` column on some tables to preserve fields we don't
    explicitly model yet (future-proofing while exploring).
"""

import argparse
import json
import os
import sqlite3
import time
from typing import Any, Iterable, List, Optional, Tuple

import fitz  # PyMuPDF


BASE = os.path.dirname(os.path.abspath(__file__))
ANNOTATOR_DIR = os.path.join(BASE, "annotator")
DATA_DIR = os.path.join(ANNOTATOR_DIR, "data")
DEFAULT_DB_PATH = os.path.join(DATA_DIR, "vectors.db")
DEFAULT_META_PATH = os.path.join(DATA_DIR, "metadata.json")
DEFAULT_PDF_PATH = os.path.join(ANNOTATOR_DIR, "01_SCHEMATIC DIAGRAM_151-E8810-202-0.pdf")


SCHEMA_SQL = """
-- SQLite performance tuning.
-- WAL mode makes bulk inserts much faster and more resilient.
-- synchronous=NORMAL is a good balance for local dev (fast, still safe-ish).
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
PRAGMA temp_store=MEMORY;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS pages (
  page_id   INTEGER PRIMARY KEY,
  page_num  INTEGER NOT NULL UNIQUE,
  pdf_width REAL,
  pdf_height REAL
);

CREATE TABLE IF NOT EXISTS drawings (
  drawing_id INTEGER PRIMARY KEY,
  page_id    INTEGER NOT NULL REFERENCES pages(page_id) ON DELETE CASCADE,
  seqno      INTEGER,
  x0         REAL NOT NULL,
  y0         REAL NOT NULL,
  x1         REAL NOT NULL,
  y1         REAL NOT NULL,
  stroke_color TEXT,
  stroke_width REAL,
  fill_color   TEXT,
  raw_json     TEXT
);

-- R-Tree for bbox queries: drawings overlapping a region.
--
-- CRITICAL: SQLite expects the column order:
--   (minX, maxX, minY, maxY)
-- If you swap these (e.g. x0,y0,x1,y1), inserts can fail with:
--   "rtree constraint failed: (x1<=y1)"
CREATE VIRTUAL TABLE IF NOT EXISTS drawings_rtree USING rtree(
    drawing_id,
    x0, x1, y0, y1
);

CREATE TABLE IF NOT EXISTS drawing_items (
  item_id    INTEGER PRIMARY KEY,
  drawing_id INTEGER NOT NULL REFERENCES drawings(drawing_id) ON DELETE CASCADE,
  kind       TEXT NOT NULL,
  x1 REAL, y1 REAL,
  x2 REAL, y2 REAL,
  x3 REAL, y3 REAL,
  x4 REAL, y4 REAL,
  raw_json TEXT
);

CREATE INDEX IF NOT EXISTS idx_drawings_page_seqno ON drawings(page_id, seqno);
CREATE INDEX IF NOT EXISTS idx_items_drawing ON drawing_items(drawing_id);
"""


def _color_to_hex(color: Any) -> Optional[str]:
    """Convert a PyMuPDF color into a CSS-ish hex string.

    PyMuPDF typically returns colors as (r,g,b) floats in [0..1].
    We store these as #rrggbb so they are easy to compare/filter later.
    """
    if color is None:
        return None
    # PyMuPDF colors are typically float tuples like (r,g,b) in 0..1
    if isinstance(color, (list, tuple)) and len(color) >= 3:
        try:
            r = int(max(0, min(1, float(color[0]))) * 255)
            g = int(max(0, min(1, float(color[1]))) * 255)
            b = int(max(0, min(1, float(color[2]))) * 255)
            return f"#{r:02x}{g:02x}{b:02x}"
        except (ValueError, TypeError):
            return None
    if isinstance(color, str):
        return color
    return None


def _safe_json(obj: Any) -> str:
    """Serialize objects to JSON without crashing on PyMuPDF types.

    PyMuPDF returns some objects (Points/Rects) that aren't JSON serializable.
    This helper provides a conservative fallback so we don't lose information.
    """
    def default(o: Any):
        if hasattr(o, "__iter__") and not isinstance(o, (str, bytes, dict, list, tuple)):
            return list(o)
        return str(o)

    return json.dumps(obj, ensure_ascii=False, default=default)


def _iter_drawing_items(items: List[Tuple]) -> Iterable[Tuple[str, Optional[float], Optional[float], Optional[float], Optional[float], Optional[float], Optional[float], Optional[float], Optional[float], Optional[str]]]:
    """Normalize `page.get_drawings()` item tuples to DB rows.

    `items` contains low-level primitives, typically:
      - ('l', p1, p2)                    line
      - ('c', p1, p2, p3, p4)            cubic Bézier curve
      - ('re', rect)                     rectangle

    We store these in a fixed-width schema so SQL queries are simple.
    Unknown/rare variants are stored in `raw_json` instead of being dropped.
    """
    for it in items:
        if not it:
            continue
        kind = it[0]
        if kind == "l" and len(it) == 3:
            p1, p2 = it[1], it[2]
            yield ("l", float(p1.x), float(p1.y), float(p2.x), float(p2.y), None, None, None, None, None)
        elif kind == "c" and len(it) == 5:
            p1, p2, p3, p4 = it[1], it[2], it[3], it[4]
            yield ("c", float(p1.x), float(p1.y), float(p2.x), float(p2.y), float(p3.x), float(p3.y), float(p4.x), float(p4.y), None)
        elif kind == "re" and len(it) == 2:
            r = it[1]
            yield ("re", float(r.x0), float(r.y0), float(r.x1), float(r.y1), None, None, None, None, None)
        else:
            yield (str(kind), None, None, None, None, None, None, None, None, _safe_json(it))


def build_db(pdf_path: str, meta_path: str, db_path: str, force: bool, max_pages: Optional[int]) -> None:
    """Build (or rebuild) the SQLite database.

    Parameters
    ----------
    pdf_path:
        PDF to parse for vector drawings.
    meta_path:
        Optional annotator metadata.json. Used for page sizes (pdf_width/height).
        If missing, we fall back to `page.rect`.
    db_path:
        Output DB file.
    force:
        If true, delete the existing DB for a clean rebuild.
    max_pages:
        Debug helper to only process the first N pages.
    """
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    if force and os.path.exists(db_path):
        os.remove(db_path)

    # Using one transaction (via `with conn:`) keeps inserts fast.
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(SCHEMA_SQL)

    meta = None
    if os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)

    # PyMuPDF open: this is where we read vector drawings.
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    if max_pages is not None:
        total_pages = min(total_pages, max_pages)

    t0 = time.time()
    with conn:
        for page_idx in range(total_pages):
            page_num = page_idx + 1

            if meta and "pages" in meta and page_idx < len(meta["pages"]):
                pmeta = meta["pages"][page_idx]
                pdf_w = float(pmeta.get("pdf_width")) if pmeta.get("pdf_width") is not None else None
                pdf_h = float(pmeta.get("pdf_height")) if pmeta.get("pdf_height") is not None else None
            else:
                rect = doc[page_idx].rect
                pdf_w, pdf_h = float(rect.width), float(rect.height)

            # Insert or update the page row.
            conn.execute(
                "INSERT OR REPLACE INTO pages(page_num, pdf_width, pdf_height) VALUES(?,?,?)",
                (page_num, pdf_w, pdf_h),
            )
            page_id = conn.execute("SELECT page_id FROM pages WHERE page_num=?", (page_num,)).fetchone()[0]

            # If rebuilding a subset of pages, clear existing rows for this page.
            # This keeps the DB consistent even when `--max-pages` is used.
            conn.execute("DELETE FROM drawing_items WHERE drawing_id IN (SELECT drawing_id FROM drawings WHERE page_id=?)", (page_id,))
            conn.execute("DELETE FROM drawings_rtree WHERE drawing_id IN (SELECT drawing_id FROM drawings WHERE page_id=?)", (page_id,))
            conn.execute("DELETE FROM drawings WHERE page_id=?", (page_id,))

            page = doc[page_idx]
            # Main extraction call.
            # Each returned dict has:
            #   - rect (bbox)
            #   - items (primitive tuples)
            #   - seqno, color, width, fill, etc.
            drawings = page.get_drawings()

            drawing_rows = []
            item_rows = []
            rtree_rows = []

            for d in drawings:
                rect = d.get("rect")
                if rect is None:
                    continue
                rx0, ry0, rx1, ry1 = float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1)
                # Defensive: ensure bbox min<=max (R-Tree requires this).
                x0, x1 = (rx0, rx1) if rx0 <= rx1 else (rx1, rx0)
                y0, y1 = (ry0, ry1) if ry0 <= ry1 else (ry1, ry0)

                seqno = d.get("seqno")
                stroke_color = _color_to_hex(d.get("color"))
                stroke_width = d.get("width")
                fill_color = _color_to_hex(d.get("fill"))
                # Store non-item fields as JSON for inspection/debugging.
                raw_json = _safe_json({k: v for k, v in d.items() if k not in {"items", "rect"}})

                drawing_rows.append((page_id, seqno, x0, y0, x1, y1, stroke_color, stroke_width, fill_color, raw_json, d.get("items", [])))

            # Insert drawings and their item primitives.
            # We insert drawings first to get a stable drawing_id.
            for page_id, seqno, x0, y0, x1, y1, stroke_color, stroke_width, fill_color, raw_json, items in drawing_rows:
                cur = conn.execute(
                    """INSERT INTO drawings(page_id, seqno, x0, y0, x1, y1, stroke_color, stroke_width, fill_color, raw_json)
                       VALUES(?,?,?,?,?,?,?,?,?,?)""",
                    (page_id, seqno, x0, y0, x1, y1, stroke_color, stroke_width, fill_color, raw_json),
                )
                drawing_id = cur.lastrowid
                # Mirror bbox into R-Tree for fast overlap queries.
                rtree_rows.append((drawing_id, x0, x1, y0, y1))

                for kind, ix1, iy1, ix2, iy2, ix3, iy3, ix4, iy4, item_raw in _iter_drawing_items(items):
                    item_rows.append((drawing_id, kind, ix1, iy1, ix2, iy2, ix3, iy3, ix4, iy4, item_raw))

            conn.executemany(
                "INSERT INTO drawings_rtree(drawing_id, x0, x1, y0, y1) VALUES(?,?,?,?,?)",
                rtree_rows,
            )
            conn.executemany(
                """INSERT INTO drawing_items(drawing_id, kind, x1, y1, x2, y2, x3, y3, x4, y4, raw_json)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
                item_rows,
            )

            if page_num % 10 == 0:
                elapsed = time.time() - t0
                rate = page_num / max(elapsed, 1e-6)
                print(f"{page_num}/{total_pages} pages… ({rate:.1f} pages/s)")

    doc.close()
    conn.close()
    elapsed = time.time() - t0
    print(f"Built {db_path} in {elapsed:.1f}s")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build SQLite DB of PDF vector drawings (with R-Tree bbox index).")
    ap.add_argument("--pdf", default=DEFAULT_PDF_PATH, help="Path to schematic PDF")
    ap.add_argument("--meta", default=DEFAULT_META_PATH, help="Path to metadata.json (optional)")
    ap.add_argument("--db", default=DEFAULT_DB_PATH, help="Output SQLite DB path")
    ap.add_argument("--force", action="store_true", help="Delete existing DB before building")
    ap.add_argument("--max-pages", type=int, default=None, help="Only process first N pages (debug)")
    args = ap.parse_args()

    if not os.path.exists(args.pdf):
        raise SystemExit(f"PDF not found: {args.pdf}")

    build_db(args.pdf, args.meta, args.db, args.force, args.max_pages)


if __name__ == "__main__":
    main()
