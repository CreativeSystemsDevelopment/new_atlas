"""Explore PDF vector data structure from PyMuPDF."""
import fitz
import json

pdf_path = "tools/annotator/01_SCHEMATIC DIAGRAM_151-E8810-202-0.pdf"
doc = fitz.open(pdf_path)
page = doc[6]  # page 7 (0-indexed)

drawings = page.get_drawings()
print(f"Page 7: {len(drawings)} drawings")
print()

# Look at first 5 drawings in detail
for i, d in enumerate(drawings[:5]):
    print(f"--- Drawing {i} ---")
    for key in d:
        if key == "items":
            print(f"  items ({len(d[key])}):")
            for item in d["items"][:5]:
                print(f"    {item}")
            remaining = len(d["items"]) - 5
            if remaining > 0:
                print(f"    ... and {remaining} more")
        else:
            print(f"  {key}: {d[key]}")
    print()

# Now look at a component area - let's load annotations for page 7
ann_path = "tools/annotator/annotations/page_7.json"
with open(ann_path) as f:
    ann = json.load(f)

print("=== Annotations on page 7 ===")
for p in ann.get("pairs", []):
    if "component" in p:
        bbox = p["component"]["bbox"]
        label = p.get("label", {}).get("text", "?")
        print(f"  {label}: bbox={bbox}")
        
        # Count drawings inside this bbox
        inside = []
        for d in drawings:
            r = d["rect"]
            # Check if drawing overlaps the component bbox
            if (r.x1 >= bbox[0] and r.x0 <= bbox[2] and
                r.y1 >= bbox[1] and r.y0 <= bbox[3]):
                inside.append(d)
        print(f"    -> {len(inside)} drawings overlap this bbox")
        
        # Show first overlapping drawing's items
        if inside:
            d = inside[0]
            print(f"    First drawing items ({len(d['items'])}):")
            for item in d["items"][:8]:
                print(f"      {item}")

print()

# Examine drawing item types
item_types = {}
for d in drawings:
    for item in d["items"]:
        t = item[0]
        item_types[t] = item_types.get(t, 0) + 1

print("=== Item type counts across all drawings ===")
for t, count in sorted(item_types.items()):
    print(f"  {t}: {count}")

doc.close()
