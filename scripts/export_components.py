"""Re-export component YOLO dataset with fixed labels."""
import json, os, shutil

ANN_DIR = r'c:\new_atlas\tools\annotator\annotations'
DATA_DIR = r'c:\new_atlas\tools\annotator\data'
META_PATH = os.path.join(DATA_DIR, 'metadata.json')
DEST = r'c:\new_atlas\atlas-components'

# Load metadata
meta = json.load(open(META_PATH, encoding='utf-8'))
pages_map = {pg['page']: pg for pg in meta['pages']}

# Clean and rebuild
if os.path.exists(DEST):
    shutil.rmtree(DEST)

# Find all annotated pages
ann_files = sorted([f for f in os.listdir(ANN_DIR)
                    if f.startswith('page_') and f.endswith('.json')])
print(f'Found {len(ann_files)} annotation files')

# Collect all unique marks
all_marks = set()
for fname in ann_files:
    data = json.load(open(os.path.join(ANN_DIR, fname), encoding='utf-8'))
    for pair in data.get('pairs', []):
        if pair.get('type') == 'continuation':
            continue
        t = pair.get('label', {}).get('text', '')
        if t:
            all_marks.add(t)

CLASSES = sorted(all_marks)
cls_idx = {c: i for i, c in enumerate(CLASSES)}
print(f'Classes ({len(CLASSES)}): {CLASSES}')

# 80/20 split
n_train = max(1, round(len(ann_files) * 0.8))
train_set = set(ann_files[:n_train])
print(f'Train: {n_train}, Valid: {len(ann_files) - n_train}')

# Build folders
for split in ('train', 'valid'):
    os.makedirs(os.path.join(DEST, split, 'images'), exist_ok=True)
    os.makedirs(os.path.join(DEST, split, 'labels'), exist_ok=True)

# data.yaml
with open(os.path.join(DEST, 'data.yaml'), 'w') as f:
    f.write('train: train/images\n')
    f.write('val: valid/images\n')
    f.write(f'nc: {len(CLASSES)}\n')
    names_str = "[" + ", ".join(f"'{c}'" for c in CLASSES) + "]\n"
    f.write(f'names: {names_str}')

for fname in ann_files:
    pg_num = int(fname.replace('page_', '').replace('.json', ''))
    pg_meta = pages_map.get(pg_num)
    if not pg_meta:
        print(f'  SKIP {fname}: no metadata')
        continue

    img_w, img_h = pg_meta['display_size']
    pdf_w = pg_meta['pdf_width']
    pdf_h = pg_meta['pdf_height']
    sx = img_w / pdf_w
    sy = img_h / pdf_h

    split = 'train' if fname in train_set else 'valid'
    img_name = pg_meta['image']
    src_img = os.path.join(DATA_DIR, 'pages', img_name)
    if not os.path.exists(src_img):
        print(f'  SKIP {fname}: image not found')
        continue

    shutil.copy2(src_img, os.path.join(DEST, split, 'images', img_name))

    ann = json.load(open(os.path.join(ANN_DIR, fname), encoding='utf-8'))
    lines = []
    for pair in ann.get('pairs', []):
        if pair.get('type') == 'continuation':
            continue
        comp = pair.get('component')
        label_text = pair.get('label', {}).get('text', '')
        if not comp or not label_text or label_text not in cls_idx:
            continue
        x0, y0, x1, y1 = comp['bbox']
        px0 = max(0.0, x0 * sx); px1 = min(img_w, x1 * sx)
        py0 = max(0.0, y0 * sy); py1 = min(img_h, y1 * sy)
        if px1 <= px0 or py1 <= py0:
            continue
        cx = ((px0 + px1) / 2) / img_w
        cy = ((py0 + py1) / 2) / img_h
        bw = (px1 - px0) / img_w
        bh = (py1 - py0) / img_h
        lines.append(f'{cls_idx[label_text]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}')

    lbl_name = img_name.replace('.png', '.txt').replace('.jpg', '.txt')
    with open(os.path.join(DEST, split, 'labels', lbl_name), 'w') as f:
        f.write('\n'.join(lines))
    print(f'  {split}/{img_name}: {len(lines)} labels')

# Summary
for split in ('train', 'valid'):
    n_img = len(os.listdir(os.path.join(DEST, split, 'images')))
    n_lbl = len(os.listdir(os.path.join(DEST, split, 'labels')))
    print(f'{split}: {n_img} images, {n_lbl} labels')

print('\nDone! Dataset at:', DEST)
