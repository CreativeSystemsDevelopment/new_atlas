"""
Multi-scale pattern search using masked component templates.
Extracts the binary shape from the mask overlay, then searches every page
of the PDF for matching patterns regardless of size (scale-invariant).
"""

import os, sys, json, time
import numpy as np
import cv2
import fitz  # PyMuPDF

# ---------- Config ----------
BASE        = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(BASE, "annotator", "data")
MASK_DIR    = os.path.join(DATA_DIR, "masks")
PAGES_DIR   = os.path.join(DATA_DIR, "pages")
META_PATH   = os.path.join(DATA_DIR, "metadata.json")
PDF_PATH    = os.path.join(BASE, "annotator",
                           "01_SCHEMATIC DIAGRAM_151-E8810-202-0.pdf")
VIEW_DPI    = 300
SCALE       = VIEW_DPI / 72.0
THRESHOLD   = 0.90   # 90% match
# Multi-scale: try these scale factors relative to the template
SCALE_RANGE = np.linspace(0.5, 2.0, 31)  # 0.5x to 2.0x in 31 steps


def load_metadata():
    with open(META_PATH) as f:
        return json.load(f)


def extract_template(mask_json_path, overlay_path, pair_idx):
    """
    Extract binary template from mask overlay for a specific pair.
    Returns a binary image (0/255) of just the masked shape, cropped tight.
    """
    with open(mask_json_path) as f:
        mdata = json.load(f)

    pair_info = mdata["pair_info"][str(pair_idx)]
    bbox = pair_info["component_bbox"]  # PDF coords

    # Convert to pixel coords
    x0 = int(bbox[0] * SCALE)
    y0 = int(bbox[1] * SCALE)
    x1 = int(bbox[2] * SCALE)
    y1 = int(bbox[3] * SCALE)

    # Load overlay as RGBA
    overlay = cv2.imread(overlay_path, cv2.IMREAD_UNCHANGED)
    if overlay is None:
        raise FileNotFoundError(f"Cannot load overlay: {overlay_path}")

    # Crop to bbox
    crop = overlay[y0:y1, x0:x1]

    # Extract alpha channel — blue pixels have alpha > 50
    if crop.shape[2] == 4:
        alpha = crop[:, :, 3]
    else:
        # Fallback: detect blue-ish pixels
        alpha = np.zeros(crop.shape[:2], dtype=np.uint8)

    # Binary: masked pixels = 255, rest = 0
    template = np.where(alpha > 50, 255, 0).astype(np.uint8)

    # Tight-crop to non-zero pixels
    ys, xs = np.where(template > 0)
    if len(xs) == 0:
        raise ValueError("No masked pixels found in template")
    template = template[ys.min():ys.max()+1, xs.min():xs.max()+1]

    return template


def render_page_binary(pdf_doc, page_idx):
    """Render a PDF page at VIEW_DPI and return binary black-pixel image."""
    page = pdf_doc[page_idx]
    mat = fitz.Matrix(SCALE, SCALE)
    pix = page.get_pixmap(matrix=mat)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
        pix.height, pix.width, pix.n)
    if pix.n == 4:
        gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    elif pix.n == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    # Black pixels (< 80) become 255, rest become 0
    _, binary = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)
    return binary


def search_page(binary_page, template, scales, threshold):
    """
    Multi-scale template matching on binary images.
    Returns list of (score, x, y, w, h, scale_factor) for matches >= threshold.
    """
    th, tw = template.shape[:2]
    ph, pw = binary_page.shape[:2]
    matches = []

    for s in scales:
        new_w = int(tw * s)
        new_h = int(th * s)
        if new_w < 5 or new_h < 5:
            continue
        if new_w >= pw or new_h >= ph:
            continue

        resized = cv2.resize(template, (new_w, new_h),
                             interpolation=cv2.INTER_NEAREST)

        result = cv2.matchTemplate(binary_page, resized,
                                   cv2.TM_CCOEFF_NORMED)

        # Find all locations above threshold
        locs = np.where(result >= threshold)
        for (y, x) in zip(*locs):
            score = result[y, x]
            matches.append((float(score), int(x), int(y),
                            new_w, new_h, float(s)))

    # Non-maximum suppression: keep best match per region
    if not matches:
        return []

    matches.sort(key=lambda m: m[0], reverse=True)
    kept = []
    for m in matches:
        sx, sy, sw, sh = m[1], m[2], m[3], m[4]
        cx, cy = sx + sw // 2, sy + sh // 2
        # Check if too close to an already-kept match
        too_close = False
        for k in kept:
            kx, ky = k[1] + k[3] // 2, k[2] + k[4] // 2
            if abs(cx - kx) < sw * 0.5 and abs(cy - ky) < sh * 0.5:
                too_close = True
                break
        if not too_close:
            kept.append(m)

    return kept


def main():
    label = "MCB 3 Phase"
    pair_idx = 0
    page_num = 7

    print(f"=== Mask Pattern Search: {label} ===")
    print(f"Template source: page {page_num}, pair {pair_idx}")
    print(f"Threshold: {THRESHOLD*100:.0f}%")
    print(f"Scale range: {SCALE_RANGE[0]:.2f}x - {SCALE_RANGE[-1]:.2f}x "
          f"({len(SCALE_RANGE)} steps)")
    print()

    # 1. Extract template
    mask_json = os.path.join(MASK_DIR, f"page_{page_num}_masks.json")
    overlay_png = os.path.join(MASK_DIR, f"page_{page_num}_overlay.png")
    template = extract_template(mask_json, overlay_png, pair_idx)
    print(f"Template size: {template.shape[1]}x{template.shape[0]} px, "
          f"{np.count_nonzero(template)} mask pixels")

    # 2. Open PDF
    pdf = fitz.open(PDF_PATH)
    total_pages = len(pdf)
    print(f"PDF: {total_pages} pages")
    print()

    # 3. Search every page
    all_results = {}
    t0 = time.time()
    for pg_idx in range(total_pages):
        pg_num = pg_idx + 1
        binary = render_page_binary(pdf, pg_idx)
        matches = search_page(binary, template, SCALE_RANGE, THRESHOLD)

        if matches:
            all_results[pg_num] = matches
            for m in matches:
                score, x, y, w, h, s = m
                # Convert pixel back to PDF coords
                pdf_x = x / SCALE
                pdf_y = y / SCALE
                print(f"  Page {pg_num:3d}: score={score:.3f}  "
                      f"scale={s:.2f}x  "
                      f"pos=({pdf_x:.1f}, {pdf_y:.1f})  "
                      f"size={w}x{h}px")

        # Progress every 10 pages
        if (pg_idx + 1) % 10 == 0:
            elapsed = time.time() - t0
            rate = (pg_idx + 1) / elapsed
            eta = (total_pages - pg_idx - 1) / rate
            print(f"  ... {pg_idx+1}/{total_pages} pages "
                  f"({elapsed:.1f}s, ~{eta:.0f}s remaining)")

    elapsed = time.time() - t0
    pdf.close()

    # 4. Summary
    total_matches = sum(len(v) for v in all_results.values())
    print()
    print(f"=== Results ===")
    print(f"Searched {total_pages} pages in {elapsed:.1f}s")
    print(f"Found {total_matches} match(es) on {len(all_results)} page(s)")
    if all_results:
        print(f"Pages with matches: "
              + ", ".join(str(p) for p in sorted(all_results.keys())))


if __name__ == "__main__":
    main()
