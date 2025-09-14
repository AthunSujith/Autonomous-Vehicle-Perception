# scripts/coco_to_yolo.py
import json, os
from pathlib import Path

# Input COCO jsons
COCO_TRAIN = "data/processed/annotations/kitti_coco_train.json"
COCO_VAL   = "data/processed/annotations/kitti_coco_val.json"

# Where images live (copied earlier by resplit)
IMG_TRAIN_DIR = "data/processed/images/train"
IMG_VAL_DIR   = "data/processed/images/val"

# Output YOLO label folders (one .txt per image)
LBL_TRAIN_DIR = "data/processed/labels/train"
LBL_VAL_DIR   = "data/processed/labels/val"

os.makedirs(LBL_TRAIN_DIR, exist_ok=True)
os.makedirs(LBL_VAL_DIR, exist_ok=True)

def convert(coco_json, out_label_dir):
    print("Converting", coco_json, "->", out_label_dir)
    coco = json.load(open(coco_json))
    imgs = {im['id']:{'file_name':im['file_name'],'width':im.get('width',None),'height':im.get('height',None)} for im in coco.get('images',[])}
    anns_by_img = {}
    for a in coco.get('annotations', []):
        anns_by_img.setdefault(a['image_id'], []).append(a)
    # Build cat mapping to 0-based YOLO class ids
    cats = coco.get('categories', [])
    catid_to_yid = {c['id']: idx for idx,c in enumerate(cats)}
    converted = 0
    for img_id, meta in imgs.items():
        fname = meta['file_name']
        w = meta['width']; h = meta['height']
        if w is None or h is None:
            # try to read from disk if dimensions missing
            from PIL import Image
            p = os.path.join( IMG_TRAIN_DIR if 'train' in out_label_dir else IMG_VAL_DIR, fname )
            if os.path.exists(p):
                im = Image.open(p)
                w,h = im.size
            else:
                # fallback: skip image
                print("Skipping (no size) ", fname)
                continue
        anns = anns_by_img.get(img_id, [])
        lines = []
        for a in anns:
            x,y,ww,hh = a['bbox']  # COCO x,y,w,h (absolute)
            # convert to YOLO normalized x_center y_center w h
            xc = x + ww/2.0
            yc = y + hh/2.0
            xc /= w; yc /= h; nw = ww / w; nh = hh / h
            cls = catid_to_yid[a['category_id']]
            lines.append(f"{cls} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}")
        out_path = os.path.join(out_label_dir, os.path.splitext(fname)[0] + ".txt")
        with open(out_path, "w") as f:
            f.write("\n".join(lines))
        converted += 1
    print("Wrote", converted, "label files to", out_label_dir)

if __name__ == "__main__":
    convert(COCO_TRAIN, LBL_TRAIN_DIR)
    convert(COCO_VAL,   LBL_VAL_DIR)
