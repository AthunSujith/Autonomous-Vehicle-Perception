# scripts/register_coco_dataset.py
from detectron2.data.datasets import register_coco_instances

def register_all():
    register_coco_instances("kitti_train", {}, "data/processed/annotations/kitti_coco_train.json", "data/processed/images/train")
    register_coco_instances("kitti_val",   {}, "data/processed/annotations/kitti_coco_val.json",   "data/processed/images/val")
    register_coco_instances("kitti_test",  {}, "data/processed/annotations/kitti_coco_test.json",  "data/processed/images/test")

if __name__ == "__main__":
    register_all()
    print('Datasets registered.')
