# scripts/convert_kitti_to_coco.py
#Why: COCO format is standard for training & evaluation; this script also filters to common classes and writes correct widths/heights for images.
import os, json
from PIL import Image
from pathlib import Path

KITTI_LABELS = ['Car','Van','Truck','Pedestrian','Person_sitting','Cyclist','Tram','Misc']  # common KITTI classes
# We'll map only the primary detection classes commonly used; adjust as needed
USE_CLASSES = ['Car','Pedestrian','Cyclist']  # classes we want to keep

def parse_label_file(label_path):
    objs = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 8:
                continue
            cls = parts[0]
            if cls not in USE_CLASSES:
                continue
            # KITTI bbox format: left, top, right, bottom are parts[4:8]
            l = float(parts[4]); t = float(parts[5]); r = float(parts[6]); b = float(parts[7])
            objs.append({'category': cls, 'bbox': [l, t, r, b]})
    return objs

def main(img_dir, label_dir, out_json):
    images = []
    annotations = []
    categories = [{'id': i+1, 'name': c} for i,c in enumerate(USE_CLASSES)]
    ann_id = 1
    img_files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg','.png'))])
    for img_id, img_name in enumerate(img_files, start=1):
        img_path = os.path.join(img_dir, img_name)
        try:
            w,h = Image.open(img_path).size
        except Exception as e:
            print('Skipping corrupt image:', img_path, e)
            continue
        images.append({'id': img_id, 'file_name': img_name, 'width': w, 'height': h})
        label_file = os.path.splitext(img_name)[0] + '.txt'
        label_path = os.path.join(label_dir, label_file)
        if not os.path.exists(label_path):
            continue
        objs = parse_label_file(label_path)
        for o in objs:
            l,t,r,b = o['bbox']
            w_bb = r - l
            h_bb = b - t
            if w_bb <=0 or h_bb <=0:
                continue
            cat_id = USE_CLASSES.index(o['category']) + 1
            annotations.append({
                'id': ann_id,
                'image_id': img_id,
                'category_id': cat_id,
                'bbox': [l, t, w_bb, h_bb],
                'area': w_bb * h_bb,
                'iscrowd': 0
            })
            ann_id += 1

    coco = {'images': images, 'annotations': annotations, 'categories': categories}
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, 'w') as f:
        json.dump(coco, f)
    print('Wrote', out_json)
    print('Images:', len(images), 'Annotations:', len(annotations))

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--images', required=True)
    p.add_argument('--labels', required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    main(args.images, args.labels, args.out)
