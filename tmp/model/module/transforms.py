from tmp.src.common_imports import *
import numpy as np, cv2, torch, random
from typing import Any, Dict

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _letterbox(img, dst_size=(640, 640), pad_val=(0, 0, 0)):
    ih, iw = img.shape[:2]
    th, tw = dst_size
    r = min(tw / iw, th / ih)
    nw, nh = int(iw * r), int(ih * r)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top = (th - nh) // 2
    left = (tw - nw) // 2
    canvas = np.full((th, tw, 3), pad_val, dtype=resized.dtype)
    canvas[top:top + nh, left:left + nw] = resized
    return canvas, {
        "scale": r,
        "pad_top": top,
        "pad_left": left,
        "orig_h": ih,
        "orig_w": iw,
        "dst_h": th,
        "dst_w": tw,
    }


def _boxes_to_net_space(boxes, meta):
    if boxes is None or len(boxes) == 0:
        return boxes
    r = meta["scale"]
    top = meta["pad_top"]
    left = meta["pad_left"]
    boxes = boxes.copy()
    boxes[:, [0, 2]] = boxes[:, [0, 2]] * r + left
    boxes[:, [1, 3]] = boxes[:, [1, 3]] * r + top
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, meta["dst_w"] - 1)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, meta["dst_h"] - 1)
    return boxes


def _hflip(img, boxes):
    img = img[:, ::-1, :]
    if boxes is not None and len(boxes) > 0:
        w = img.shape[1]
        boxes[:, [0, 2]] = w - boxes[:, [2, 0]] - 1
    return img, boxes


def _color_jitter(img, b=0.2, c=0.2, s=0.2):
    out = img.astype(np.float32)
    out *= 1 + random.uniform(-b, b)
    m = out.mean(axis=(0, 1), keepdims=True)
    out = (out - m) * (1 + random.uniform(-c, c)) + m
    hsv = cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 1] *= 1 + random.uniform(-s, s)
    out = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2RGB)
    return out


def _normalize(img, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    img = img.astype(np.float32) / 255.0
    return (img - mean) / std


def _to_tensor(img):
    return torch.from_numpy(np.transpose(img, (2, 0, 1)).astype(np.float32))


class TrainCompose:
    def __init__(self, input_size=(640, 640), keep_ratio=True, pipeline=None, min_box=1):
        self.input_size = tuple(input_size)
        self.pipeline = pipeline or {}
        self.min_box = int(min_box)

    def __call__(self, sample):
        img = sample["image"]
        boxes = sample.get("boxes", np.zeros((0, 4), np.float32))
        labels = sample.get("labels", np.zeros((0,), np.int64))

        if random.random() < 0.5:
            img, boxes = _hflip(img, boxes)

        if self.pipeline.get("color_jitter", True):
            img = _color_jitter(img, 0.1, 0.1, 0.1)

        img, meta = _letterbox(img, self.input_size)
        boxes = _boxes_to_net_space(boxes, meta)

        if len(boxes) > 0:
            w = boxes[:, 2] - boxes[:, 0]
            h = boxes[:, 3] - boxes[:, 1]
            keep = (w >= self.min_box) & (h >= self.min_box)
            boxes = boxes[keep]
            labels = labels[keep]

        img_t = _to_tensor(_normalize(img))

        return {
            "image": img_t,
            "boxes": boxes,
            "labels": labels,
            "meta": {"letterbox": meta, "input_size": self.input_size},
        }


class ValCompose:
    def __init__(self, input_size=(640, 640), keep_ratio=True, min_box=1):
        self.input_size = tuple(input_size)
        self.min_box = int(min_box)

    def __call__(self, sample):
        img = sample["image"]
        boxes = sample.get("boxes", np.zeros((0, 4), np.float32))
        labels = sample.get("labels", np.zeros((0,), np.int64))

        img, meta = _letterbox(img, self.input_size)
        boxes = _boxes_to_net_space(boxes, meta)

        if len(boxes) > 0:
            w = boxes[:, 2] - boxes[:, 0]
            h = boxes[:, 3] - boxes[:, 1]
            keep = (w >= self.min_box) & (h >= self.min_box)
            boxes = boxes[keep]
            labels = labels[keep]

        img_t = _to_tensor(_normalize(img))

        return {
            "image": img_t,
            "boxes": boxes,
            "labels": labels,
            "meta": {"letterbox": meta, "input_size": self.input_size},
        }


def build_transforms(is_train, input_size=(640, 640), pipeline=None, keep_ratio=True, min_box=1):
    return TrainCompose(input_size, keep_ratio, pipeline, min_box) if is_train else ValCompose(
        input_size, keep_ratio, min_box
    )
