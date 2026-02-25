import json
from PIL import Image

meta = json.load(open('tools/annotator/data/metadata.json', encoding='utf-8'))
pg = meta['pages'][6]  # page_7 = index 6
print('metadata display_size:', pg.get('display_size'))
print('metadata pdf_width:', pg.get('pdf_width'))
print('metadata pdf_height:', pg.get('pdf_height'))

img = Image.open('tools/annotator/data/pages/page_7.png')
print('data/pages image size:', img.size)

img2 = Image.open(r'c:\new_atlas\_backup2\atlas_dataset_v2_augmented\train\images\page_7.png')
print('backup image size:', img2.size)
