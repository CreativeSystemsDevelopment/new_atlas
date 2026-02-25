"""
ATLAS Annotator — PDF Preprocessor
Renders schematic PDF pages at two DPI tiers:
  - Master (1200 DPI): archival-quality PNGs for processing
  - Display (300 DPI): lighter PNGs for the annotator UI

Extracts text blocks and vector shape metadata for snap features.

Usage: python preprocess.py <pdf_path> [--master-dpi 1200] [--display-dpi 300] [--output data]
"""
import fitz
import json
import os
import sys
import argparse
import time


def preprocess(pdf_path, output_dir="data", master_dpi=1200, display_dpi=300):
    master_dir = os.path.join(output_dir, "master")
    pages_dir = os.path.join(output_dir, "pages")
    os.makedirs(master_dir, exist_ok=True)
    os.makedirs(pages_dir, exist_ok=True)

    doc = fitz.open(pdf_path)
    display_scale = display_dpi / 72.0
    pages_data = []
    total = len(doc)
    t0 = time.time()

    print(f"  PDF: {os.path.basename(pdf_path)}")
    print(f"  Pages: {total}")
    print(f"  Master DPI: {master_dpi}  |  Display DPI: {display_dpi}")
    print(f"  Output: {os.path.abspath(output_dir)}")
    print()

    for i in range(total):
        page = doc[i]
        pct = ((i + 1) / total) * 100
        print(f"  [{i+1:3d}/{total}] ({pct:5.1f}%) Page {i+1}...", end=" ", flush=True)

        # Render master image (1200 DPI)
        master_name = f"page_{i+1}.png"
        pix_master = page.get_pixmap(dpi=master_dpi)
        pix_master.save(os.path.join(master_dir, master_name))
        master_w, master_h = pix_master.width, pix_master.height

        # Render display image (300 DPI)
        display_name = f"page_{i+1}.png"
        pix_display = page.get_pixmap(dpi=display_dpi)
        pix_display.save(os.path.join(pages_dir, display_name))
        display_w, display_h = pix_display.width, pix_display.height

        # Extract text spans
        text_blocks = []
        raw = page.get_text("dict")
        for block in raw["blocks"]:
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span["text"].strip()
                    if text:
                        text_blocks.append({
                            "text": text,
                            "bbox": [round(c, 2) for c in span["bbox"]],
                        })

        # Extract vector shapes (bounding boxes for snap)
        shapes = []
        for d in page.get_drawings():
            r = d["rect"]
            w, h = r.x1 - r.x0, r.y1 - r.y0
            if w < 0.5 and h < 0.5:
                continue
            shapes.append({
                "bbox": [round(r.x0, 2), round(r.y0, 2), round(r.x1, 2), round(r.y1, 2)],
            })

        pages_data.append({
            "page": i + 1,
            "pdf_width": round(page.rect.width, 2),
            "pdf_height": round(page.rect.height, 2),
            "image": display_name,
            "master_image": master_name,
            "master_size": [master_w, master_h],
            "display_size": [display_w, display_h],
            "text_blocks": text_blocks,
            "shapes": shapes,
        })

        master_mb = os.path.getsize(os.path.join(master_dir, master_name)) / 1e6
        print(f"master {master_w}x{master_h} ({master_mb:.1f}MB)  "
              f"display {display_w}x{display_h}  "
              f"({len(text_blocks)} texts, {len(shapes)} shapes)")

    doc.close()

    meta = {
        "pdf": os.path.basename(pdf_path),
        "master_dpi": master_dpi,
        "display_dpi": display_dpi,
        "scale": display_scale,
        "pages": pages_data,
    }
    meta_path = os.path.join(output_dir, "metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f)

    elapsed = time.time() - t0
    total_master = sum(
        os.path.getsize(os.path.join(master_dir, p["master_image"]))
        for p in pages_data
    ) / 1e6
    print(f"\n  Done in {elapsed:.1f}s")
    print(f"  Master images: {total_master:.0f} MB total ({master_dir})")
    print(f"  Metadata: {meta_path}")
    return meta_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ATLAS PDF Preprocessor")
    parser.add_argument("pdf", help="Path to schematic PDF")
    parser.add_argument("--master-dpi", type=int, default=1200)
    parser.add_argument("--display-dpi", type=int, default=300)
    parser.add_argument("--output", default="data")
    args = parser.parse_args()
    preprocess(args.pdf, args.output, args.master_dpi, args.display_dpi)
