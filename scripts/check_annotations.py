import json, os

ANN_DIR = r'tools\annotator\annotations'
META = json.load(open(r'tools\annotator\data\metadata.json', encoding='utf-8'))

# For each annotation file, print page name + first bbox + the image dimensions
for fname in sorted(os.listdir(ANN_DIR)):
    if not fname.startswith('page_') or not fname.endswith('.json'):
        continue
    page_num = int(fname.replace('page_', '').replace('.json', ''))
    idx = page_num - 1

    data = json.load(open(os.path.join(ANN_DIR, fname), encoding='utf-8'))
    pairs = data.get('pairs', [])
    if not pairs:
        print(f'{fname}: (empty)')
        continue

    # Get first pair component bbox
    first = pairs[0]
    comp = first.get('component', {}).get('bbox', None)
    lbl  = first.get('label', {}).get('text', '?')

    # Get page image size from metadata
    pg = META['pages'][idx] if idx < len(META['pages']) else {}
    w = pg.get('width', '?')
    h = pg.get('height', '?')

    print(f'{fname}: {len(pairs)} pairs | first={lbl} comp_bbox={comp} | page size={w}x{h}')
