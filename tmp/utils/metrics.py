import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def box_iou(box1: np.ndarray, box2: np.ndarray) -> np.ndarray:
    N = box1.shape[0]
    M = box2.shape[0]
    if N == 0 or M == 0:
        return np.zeros((N, M))

    b1 = box1[:, None, :]  # (N,1,4)
    b2 = box2[None, :, :]  # (1,M,4)

    inter_x1 = np.maximum(b1[..., 0], b2[..., 0])
    inter_y1 = np.maximum(b1[..., 1], b2[..., 1])
    inter_x2 = np.minimum(b1[..., 2], b2[..., 2])
    inter_y2 = np.minimum(b1[..., 3], b2[..., 3])

    inter_w = np.maximum(0, inter_x2 - inter_x1)
    inter_h = np.maximum(0, inter_y2 - inter_y1)
    inter = inter_w * inter_h

    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])

    union = area1[:, None] + area2 - inter
    return inter / np.clip(union, 1e-6, None)


def precision_recall(pred_boxes, gt_boxes, iou_threshold=0.5):
    if len(pred_boxes) == 0 and len(gt_boxes) == 0:
        return 1.0, 1.0
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return 0.0, 0.0

    p = np.array(pred_boxes)
    g = np.array(gt_boxes)
    ious = box_iou(p, g)
    matched = (ious.max(axis=1) >= iou_threshold).sum()

    precision = matched / len(pred_boxes)
    recall = matched / len(gt_boxes)
    return precision, recall


def coco_map_eval(ann_path: str, results_json: str):
    coco_gt = COCO(ann_path)
    coco_dt = coco_gt.loadRes(results_json)
    eval = COCOeval(coco_gt, coco_dt, "bbox")
    eval.evaluate()
    eval.accumulate()
    eval.summarize()
    return eval.stats
