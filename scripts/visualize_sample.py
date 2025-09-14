# scripts/visualize_sample.py
import json, cv2, os
from random import choice

ANN = 'data/processed/annotations/kitti_coco_val.json'
IMG_DIR = 'data/processed/images/val'
OUT = 'outputs/vis/vis_sample.jpg'
os.makedirs(os.path.dirname(OUT), exist_ok=True)

c = json.load(open(ANN, 'r'))
images = c['images']
anns = c['annotations']
cats = {cat['id']: cat['name'] for cat in c['categories']}

img = choice(images)
img_path = os.path.join(IMG_DIR, img['file_name'])
im = cv2.imread(img_path)
if im is None:
    raise SystemExit('Cannot open image: ' + img_path)

for a in anns:
    if a['image_id'] != img['id']:
        continue
    x,y,w,h = [int(v) for v in a['bbox']]
    cv2.rectangle(im, (x,y), (x+w, y+h), (0,255,0), 2)
    cls = cats.get(a['category_id'], 'N/A')
    cv2.putText(im, cls, (x, max(0,y-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

cv2.imwrite(OUT, im)
print('Wrote', OUT)
