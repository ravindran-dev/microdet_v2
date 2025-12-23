# tmp/data/coco_dataset.py
import os
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Dict, Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from pycocotools.coco import COCO


class CocoDataset(Dataset):
    def __init__(
        self,
        img_dir: str,
        ann_path: str,
        transforms=None,
        input_size: Tuple[int, int] = (640, 640),
        keep_ratio: bool = True,
        class_names: Optional[Sequence[str]] = None,
    ):
        self.img_dir = str(img_dir)
        self.ann_path = str(ann_path)
        self.coco = COCO(self.ann_path)
        self.ids = list(self.coco.imgs.keys())
        self.transforms = transforms
        self.input_size = tuple(input_size)
        self.keep_ratio = keep_ratio
        self.class_names = list(class_names) if class_names is not None else ["person"]
        self.cat_ids = self.coco.getCatIds(catNms=self.class_names)

    def __len__(self) -> int:
        return len(self.ids)

    def _possible_image_paths(self, file_name: str) -> List[str]:
        # generate candidate paths to try opening the image
        candidates = []
        # If file_name is absolute, try it first
        if os.path.isabs(file_name):
            candidates.append(file_name)
        # direct join: img_dir + file_name
        candidates.append(os.path.join(self.img_dir, file_name))
        # sometimes file_name already contains a folder like "images/xxx.jpg"
        candidates.append(os.path.join(self.img_dir, os.path.basename(file_name)))
        # sometimes img_dir already ends with 'images' and file_name starts with 'images/...'
        candidates.append(os.path.join(self.img_dir, file_name.lstrip("/")))
        # try relative to annotation json parent dir (useful if img_dir is relative)
        ann_parent = os.path.dirname(self.ann_path)
        if ann_parent:
            candidates.append(os.path.join(ann_parent, file_name))
            candidates.append(os.path.join(ann_parent, self.img_dir, file_name))
        # de-duplicate while preserving order
        seen = set()
        dedup = []
        for p in candidates:
            if p not in seen:
                seen.add(p)
                dedup.append(os.path.normpath(p))
        return dedup

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Dict[str, Any], Dict[str, Any]]:
        img_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=img_id, catIds=self.cat_ids, iscrowd=None)
        anns = self.coco.loadAnns(ann_ids)

        img_info = self.coco.loadImgs(img_id)[0]
        file_name = img_info.get("file_name", "")
        tried = self._possible_image_paths(file_name)

        img = None
        tried_str = "\n".join(tried)
        for p in tried:
            if os.path.exists(p):
                img = cv2.imread(p)
                if img is not None:
                    break

        if img is None:
            raise FileNotFoundError(
                f"Image not found for img_id={img_id} (file_name='{file_name}').\n"
                f"Tried the following paths:\n{tried_str}\n"
                "Check your `img_dir` in the config and the `file_name` entries in your COCO JSON."
            )

        # convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        bboxes = []
        labels = []
        for a in anns:
            x, y, w, h = a["bbox"]
            # skip degenerate boxes
            if w <= 0 or h <= 0:
                continue
            bboxes.append([x, y, x + w, y + h])
            # map categories to 0..C-1 (we assume single class mapping here)
            labels.append(0)

        if len(bboxes) > 0:
            bboxes = np.array(bboxes, dtype=np.float32)
            labels = np.array(labels, dtype=np.int64)
        else:
            bboxes = np.zeros((0, 4), dtype=np.float32)
            labels = np.zeros((0,), dtype=np.int64)

        if self.transforms:
            # expect transform to return (img_tensor, targets, meta) as your code uses
            img_t, target, meta = self.transforms(img, bboxes, labels)
        else:
            img_t = torch.from_numpy(np.transpose(img, (2, 0, 1))).float() / 255.0
            target = {
                "boxes": torch.from_numpy(bboxes).float(),
                "labels": torch.from_numpy(labels).long(),
            }
            meta = {"input_size": self.input_size}

        target["image_id"] = torch.tensor([img_id], dtype=torch.int64)
        return img_t, target, meta
