"""Explore the actual line segments inside a component bbox."""
import fitz
import json

pdf_path = "tools/annotator/01_SCHEMATIC DIAGRAM_151-E8810-202-0.pdf"
doc = fitz.open(pdf_path)
page = doc[6]  # page 7

drawings = page.get_drawings()

# ELB 3 Phase component bbox from annotations
bbox = [178.28, 532.94, 218.38, 587.02]

print(f"ELB 3 Phase bbox: {bbox}")
print(f"bbox size: {bbox[2]-bbox[0]:.1f} x {bbox[3]-bbox[1]:.1f} PDF points")
print()

inside = []
for d in drawings:
    r = d["rect"]
    if (r.x1 >= bbox[0] and r.x0 <= bbox[2] and
        r.y1 >= bbox[1] and r.y0 <= bbox[3]):
        inside.append(d)

print(f"{len(inside)} drawings overlap")
for i, d in enumerate(inside):
    r = d["rect"]
    rsize = f"({r.x1-r.x0:.1f}x{r.y1-r.y0:.1f})"
    print(f"\n  Drawing {i} seqno={d['seqno']} rect={r} {rsize} fill={d.get('fill')} color={d.get('color')}")
    for item in d["items"]:
        if item[0] == "l":
            p1, p2 = item[1], item[2]
            print(f"    LINE ({p1.x:.1f},{p1.y:.1f}) -> ({p2.x:.1f},{p2.y:.1f})")
        elif item[0] == "c":
            p1, p2, p3, p4 = item[1], item[2], item[3], item[4]
            print(f"    CURVE ({p1.x:.1f},{p1.y:.1f}) -> ({p4.x:.1f},{p4.y:.1f})")
        elif item[0] == "re":
            r2 = item[1]
            print(f"    RECT ({r2.x0:.1f},{r2.y0:.1f},{r2.x1:.1f},{r2.y1:.1f})")

doc.close()
