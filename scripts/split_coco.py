# scripts/split_coco.py
import json, os, shutil, random
from pathlib import Path

random.seed(42)

COCO_IN = 'data/processed/annotations/kitti_coco.json'
OUT_DIR = 'data/processed'
IMG_SRC_DIR = 'data/raw/kitti/images'
TRAIN_DIR = os.path.join(OUT_DIR, 'images/train')
VAL_DIR   = os.path.join(OUT_DIR, 'images/val')
TEST_DIR  = os.path.join(OUT_DIR, 'images/test')
ANN_DIR   = os.path.join(OUT_DIR, 'annotations')

os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(VAL_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)
os.makedirs(ANN_DIR, exist_ok=True)

coco = json.load(open(COCO_IN, 'r'))
images = coco.get('images', [])
n = len(images)
random.shuffle(images)

n_train = int(0.7 * n)
n_val = int(0.2 * n)
train_imgs = images[:n_train]
val_imgs = images[n_train:n_train+n_val]
test_imgs = images[n_train+n_val:]

def write_split(img_list, out_json, out_img_dir):
    img_ids = set([img['id'] for img in img_list])
    anns = [a for a in coco.get('annotations', []) if a['image_id'] in img_ids]
    out = {'images': img_list, 'annotations': anns, 'categories': coco.get('categories', [])}
    with open(out_json, 'w') as f:
        json.dump(out, f)
    # copy images
    for img in img_list:
        src = os.path.join(IMG_SRC_DIR, img['file_name'])
        dst = os.path.join(out_img_dir, img['file_name'])
        if os.path.exists(src):
            shutil.copy(src, dst)

write_split(train_imgs, os.path.join(ANN_DIR, 'kitti_coco_train.json'), TRAIN_DIR)
write_split(val_imgs,   os.path.join(ANN_DIR, 'kitti_coco_val.json'),   VAL_DIR)
write_split(test_imgs,  os.path.join(ANN_DIR, 'kitti_coco_test.json'),  TEST_DIR)

print('Train/Val/Test split done. Counts:', len(train_imgs), len(val_imgs), len(test_imgs))
