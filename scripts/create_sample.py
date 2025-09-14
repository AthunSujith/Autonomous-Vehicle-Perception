# scripts/create_sample.py
import json, os, shutil
from pathlib import Path

SRC_ANN = 'data/processed/annotations/kitti_coco_train.json'
SRC_IMG_DIR = 'data/processed/images/train'
OUT_IMG_DIR = 'data/sample/images'
OUT_ANN = 'data/sample/kitti_sample.json'
N = 200

os.makedirs(OUT_IMG_DIR, exist_ok=True)

c = json.load(open(SRC_ANN, 'r'))
images = c['images'][:N]
img_ids = set([i['id'] for i in images])
anns = [a for a in c['annotations'] if a['image_id'] in img_ids]
out = {'images': images, 'annotations': anns, 'categories': c['categories']}
json.dump(out, open(OUT_ANN, 'w'))
for im in images:
    src = os.path.join(SRC_IMG_DIR, im['file_name'])
    dst = os.path.join(OUT_IMG_DIR, im['file_name'])
    if os.path.exists(src):
        shutil.copy(src, dst)
print('Created sample with', len(images), 'images ->', OUT_IMG_DIR)
