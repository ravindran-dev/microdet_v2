import random
from typing import Tuple, Dict, Any, List, Optional

import cv2
import numpy as np
import torch


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _hflip(img, boxes):
    img = img[:, ::-1, :]
    if boxes is not None and len(boxes) > 0:
        w = img.shape[1]
        x1 = boxes[:, 0].copy()
        x2 = boxes[:, 2].copy()
        boxes[:, 0] = w - x2
        boxes[:, 2] = w - x1
    return img, boxes


def _color_jitter(img, b=0.2, c=0.2, s=0.2):
    img = img.astype(np.float32)
    if b > 0:
        img *= 1.0 + random.uniform(-b, b)
    if c > 0:
        m = img.mean(axis=(0, 1), keepdims=True)
        img = (img - m) * (1.0 + random.uniform(-c, c)) + m
    if s > 0:
        hsv = cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[..., 1] = np.clip(hsv[..., 1] * (1.0 + random.uniform(-s, s)), 0, 255)
        img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)
    return np.clip(img, 0, 255)


def _random_perspective(img, boxes, degrees=1.0, translate=0.01, scale=0.01):
    h, w = img.shape[:2]
    ang = random.uniform(-degrees, degrees)
    sx = 1 + random.uniform(-scale, scale)
    sy = 1 + random.uniform(-scale, scale)
    tx = random.uniform(-translate, translate) * w
    ty = random.uniform(-translate, translate) * h

    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1)
    M[:, 0] *= sx
    M[:, 1] *= sy
    M[:, 2] += [tx, ty]

    img2 = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    if boxes is not None and len(boxes) > 0:
        xy1 = np.stack([boxes[:, 0], boxes[:, 1], np.ones(len(boxes))], 1)
        xy2 = np.stack([boxes[:, 2], boxes[:, 3], np.ones(len(boxes))], 1)
        p1 = (M @ xy1.T).T
        p2 = (M @ xy2.T).T
        x1 = np.minimum(p1[:, 0], p2[:, 0])
        y1 = np.minimum(p1[:, 1], p2[:, 1])
        x2 = np.maximum(p1[:, 0], p2[:, 0])
        y2 = np.maximum(p1[:, 1], p2[:, 1])
        boxes = np.stack([x1, y1, x2, y2], 1)
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, w)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, h)
    return img2, boxes


def _letterbox_any(img, target, color=(0, 0, 0)):
    ih, iw = img.shape[:2]
    th, tw = target
    r = min(tw / iw, th / ih)
    nw, nh = int(iw * r), int(ih * r)

    img_resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top = (th - nh) // 2
    left = (tw - nw) // 2

    canvas = np.full((th, tw, 3), color, dtype=img_resized.dtype)
    canvas[top:top + nh, left:left + nw] = img_resized

    return canvas, {
        "scale": r,
        "pad_top": top,
        "pad_left": left,
        "orig_h": ih,
        "orig_w": iw,
        "new_h": nh,
        "new_w": nw,
    }


def _boxes_to_net_space(boxes, meta):
    if boxes is None or len(boxes) == 0:
        return boxes
    r = meta["scale"]
    t = meta["pad_top"]
    l = meta["pad_left"]
    b = boxes.copy()
    b[:, [0, 2]] = b[:, [0, 2]] * r + l
    b[:, [1, 3]] = b[:, [1, 3]] * r + t
    return b


def _normalize(img, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    img = img.astype(np.float32) / 255.0
    return (img - mean) / std


class TrainTransform:
    def __init__(self, input_size=(640, 640), keep_ratio=True, pipeline=None, min_box=1):
        self.input_size = tuple(input_size)
        self.keep_ratio = keep_ratio
        self.pipeline = pipeline or {}
        self.min_box = min_box

    def __call__(self, img, boxes=None, labels=None):
        if random.random() < 0.5:
            img, boxes = _hflip(img, boxes)

        if self.pipeline.get("color_jitter", True):
            img = _color_jitter(img)

        if self.pipeline.get("perspective", False):
            img, boxes = _random_perspective(img, boxes)

        img_lb, meta = _letterbox_any(img, self.input_size)
        boxes_net = _boxes_to_net_space(boxes, meta) if boxes is not None else None

        if boxes_net is not None and len(boxes_net) > 0:
            w_ = boxes_net[:, 2] - boxes_net[:, 0]
            h_ = boxes_net[:, 3] - boxes_net[:, 1]
            mask = (w_ >= self.min_box) & (h_ >= self.min_box)
            boxes_net = boxes_net[mask]
            labels = labels[mask]

        img_t = torch.from_numpy(
            np.transpose(_normalize(img_lb), (2, 0, 1)).astype(np.float32)
        )

        targets = {
            "boxes": torch.from_numpy(boxes_net).float() if boxes_net is not None else torch.zeros((0, 4)),
            "labels": torch.from_numpy(labels).long() if labels is not None else torch.zeros((0,), dtype=torch.long),
        }

        return img_t, targets, {"letterbox": meta, "input_size": self.input_size}


class ValTransform:
    def __init__(self, input_size=(640, 640), keep_ratio=True, min_box=1):
        self.input_size = tuple(input_size)
        self.keep_ratio = keep_ratio
        self.min_box = min_box

    def __call__(self, img, boxes=None, labels=None):
        img_lb, meta = _letterbox_any(img, self.input_size)
        boxes_net = _boxes_to_net_space(boxes, meta) if boxes is not None else None

        if boxes_net is not None and len(boxes_net) > 0:
            w_ = boxes_net[:, 2] - boxes_net[:, 0]
            h_ = boxes_net[:, 3] - boxes_net[:, 1]
            mask = (w_ >= self.min_box) & (h_ >= self.min_box)
            boxes_net = boxes_net[mask]
            labels = labels[mask]

        img_t = torch.from_numpy(
            np.transpose(_normalize(img_lb), (2, 0, 1)).astype(np.float32)
        )

        targets = {
            "boxes": torch.from_numpy(boxes_net).float() if boxes_net is not None else torch.zeros((0, 4)),
            "labels": torch.from_numpy(labels).long() if labels is not None else torch.zeros((0,), dtype=torch.long),
        }

        return img_t, targets, {"letterbox": meta, "input_size": self.input_size}


def build_transforms(is_train=True, input_size=(640, 640), pipeline=None, keep_ratio=True, min_box=1):
    if is_train:
        return TrainTransform(input_size, keep_ratio, pipeline, min_box)
    return ValTransform(input_size, keep_ratio, min_box)
