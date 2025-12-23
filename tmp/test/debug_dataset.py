from tmp.data.coco_dataset import CocoDataset
from tmp.data.transforms import build_transforms
import os, sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir, os.pardir))
sys.path.insert(0, PROJECT_ROOT)

img_dir = "tmp/data/dataset/images"
ann_path = "tmp/data/dataset/result.json"

tf = build_transforms(
    is_train=True,
    input_size=(640, 640),
    pipeline={"color_jitter": True},
    keep_ratio=True,
)

ds = CocoDataset(
    img_path=img_dir,
    ann_path=ann_path,
    input_size=(640, 640),
    keep_ratio=True,
    transform=tf,
)

print("Num images:", len(ds))
img, target, meta = ds[0]
print("Image shape:", img.shape)
print("Boxes:", target["boxes"])
print("Labels:", target["labels"])

