# scripts/check_files.py
#Why: quick sanity check for missing/corrupt files before conversion — prevents errors later.
import os
from PIL import Image

IMG_DIR = 'data/raw/kitti/images'
LAB_DIR = 'data/raw/kitti/labels'

imgs = sorted([f for f in os.listdir(IMG_DIR) if f.lower().endswith(('.jpg','.png'))])
labs = sorted([f for f in os.listdir(LAB_DIR) if f.lower().endswith('.txt')])

print(f'Images found: {len(imgs)}')
print(f'Label files found: {len(labs)}')

# Show a few name pairs to visually confirm matching names
for i in range(min(10, len(imgs))):
    img_name = imgs[i]
    lab_name = os.path.splitext(img_name)[0] + '.txt'
    exists = 'OK' if lab_name in labs else 'MISSING'
    print(f'{img_name}  ->  {lab_name} : {exists}')

# try opening first image to check corruption
if imgs:
    p = os.path.join(IMG_DIR, imgs[0])
    im = Image.open(p)
    print('First image size:', im.size)
