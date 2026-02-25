"""
Convert YOLO .txt labels back to annotator JSON format.
Uses the v2_augmented dataset which has per-component-class labels.
"""
import json, os

YOLO_DIR = r'c:\new_atlas\_backup2\atlas_dataset_v2_augmented'
ANN_DIR  = r'c:\new_atlas\tools\annotator\annotations'
META     = json.load(open(r'c:\new_atlas\tools\annotator\data\metadata.json', encoding='utf-8'))

# Class names from data.yaml
CLASSES = ['A', 'AIR', 'AMP', 'BRAKE', 'CNV', 'CON', 'CONTROLSOURCE', 'CP', 'CPA',
'CPB', 'CPC', 'CPD', 'CR', 'CRF', 'CT', 'DBU', 'DOORINTERLOCK', 'DS', 'EARTHLEAKAGEDETECTOR',
'ELB', 'ELR', 'F', 'G', 'INV', 'KS', 'LED', 'LNF', 'M', 'MC', 'MCA', 'MCB', 'MCC',
'MCF', 'MCR', 'MMS', 'MODULE', 'MOTORBOX', 'MS', 'PBES', 'PC', 'PCES', 'PL', 'R',
'REC', 'RTC', 'SR', 'SRU', 'SZZ', 'T', 'TB', 'THR', 'WHM', 'ZCT']

# Index metadata by page number
pages_by_num = {pg["page"]: pg for pg in META["pages"]}

seen = set()
for split in ['train', 'valid']:
    label_dir = os.path.join(YOLO_DIR, split, 'labels')

    if not os.path.exists(label_dir):
        continue

    for fname in sorted(os.listdir(label_dir)):
        if not fname.startswith('page_') or not fname.endswith('.txt'):
            continue

        page_num = int(fname.replace('page_', '').replace('.txt', ''))
        if page_num in seen:
            continue
        seen.add(page_num)

        pg_meta = pages_by_num.get(page_num)
        if not pg_meta:
            print(f'  SKIP {fname}: no metadata')
            continue

        # Use PDF dimensions directly from metadata — YOLO coords are normalized against these
        pdf_w = pg_meta["pdf_width"]
        pdf_h = pg_meta["pdf_height"]
        
        # Read YOLO labels
        lines = open(os.path.join(label_dir, fname)).read().strip().split('\n')
        if not lines or lines == ['']:
            print(f'  SKIP {fname}: empty')
            continue

        # Parse YOLO records: class_id cx cy w h (normalized 0-1)
        records = []
        for line in lines:
            parts = line.strip().split()
            cls_id = int(parts[0])
            cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            
            # Convert normalized -> PDF points
            x_center_pdf = cx * pdf_w
            y_center_pdf = cy * pdf_h
            w_pdf = bw * pdf_w
            h_pdf = bh * pdf_h
            
            x0 = x_center_pdf - w_pdf / 2
            y0 = y_center_pdf - h_pdf / 2
            x1 = x_center_pdf + w_pdf / 2
            y1 = y_center_pdf + h_pdf / 2
            
            records.append({
                'cls_id': cls_id,
                'cls_name': CLASSES[cls_id] if cls_id < len(CLASSES) else f'UNK{cls_id}',
                'bbox': [round(x0, 2), round(y0, 2), round(x1, 2), round(y1, 2)]
            })
        
        # Group into pairs: each record becomes a component with its class as the label
        # The annotator format: {component: {bbox}, label: {text, bbox}}
        # Since we only have component bboxes, we'll create a label with text = class name
        # and place the label bbox slightly above the component
        pairs = []
        for r in records:
            bx0, by0, bx1, by1 = r['bbox']
            # Create label bbox: small box above the component
            lbl_h = 4.0  # ~4 PDF points tall
            lbl_w = len(r['cls_name']) * 4.0  # rough width
            lbl_cx = (bx0 + bx1) / 2
            lbl_y1 = by0 - 1  # just above component
            lbl_y0 = lbl_y1 - lbl_h
            lbl_x0 = lbl_cx - lbl_w / 2
            lbl_x1 = lbl_cx + lbl_w / 2
            
            pairs.append({
                'component': {'bbox': r['bbox']},
                'label': {
                    'text': r['cls_name'],
                    'bbox': [round(lbl_x0, 2), round(lbl_y0, 2), round(lbl_x1, 2), round(lbl_y1, 2)]
                }
            })
        
        # Write annotation JSON
        out_path = os.path.join(ANN_DIR, f'page_{page_num}.json')
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump({'pairs': pairs, 'wire_labels': []}, f, indent=2, ensure_ascii=False)
        
        print(f'  Restored page_{page_num}.json: {len(pairs)} pairs')

print('\nDone!')
