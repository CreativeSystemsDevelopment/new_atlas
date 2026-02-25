import json
meta = json.load(open('tools/annotator/data/metadata.json', encoding='utf-8'))
pages = meta['pages']
print(f'Total pages: {len(pages)}')
for i, p in enumerate(pages):
    print(f'  idx {i}: image={p["image"]}')
