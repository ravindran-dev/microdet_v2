import math
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch


def bbox_overlaps_iou(b1: torch.Tensor, b2: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    N, M = b1.size(0), b2.size(0)
    if N == 0 or M == 0:
        return b1.new_zeros((N, M))
    x1 = torch.max(b1[:, None, 0], b2[None, :, 0])
    y1 = torch.max(b1[:, None, 1], b2[None, :, 1])
    x2 = torch.min(b1[:, None, 2], b2[None, :, 2])
    y2 = torch.min(b1[:, None, 3], b2[None, :, 3])
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    area1 = (b1[:, 2] - b1[:, 0]).clamp(min=0) * (b1[:, 3] - b1[:, 1]).clamp(min=0)
    area2 = (b2[:, 2] - b2[:, 0]).clamp(min=0) * (b2[:, 3] - b2[:, 1]).clamp(min=0)
    union = area1[:, None] + area2[None, :] - inter + eps
    return inter / union


def bbox2distance(points_xy: torch.Tensor, boxes_xyxy: torch.Tensor) -> torch.Tensor:
    l = points_xy[:, 0] - boxes_xyxy[:, 0]
    t = points_xy[:, 1] - boxes_xyxy[:, 1]
    r = boxes_xyxy[:, 2] - points_xy[:, 0]
    b = boxes_xyxy[:, 3] - points_xy[:, 1]
    return torch.stack([l, t, r, b], dim=1)


def distance2bbox(
    points_xy: torch.Tensor,
    distances_ltrb: torch.Tensor,
    max_shape: Optional[Tuple[int, int]] = None,
    clamp: float = -1.0,
) -> torch.Tensor:
    if clamp > 0:
        distances_ltrb = torch.clamp(distances_ltrb, min=0.0, max=clamp)
    else:
        distances_ltrb = torch.clamp_min(distances_ltrb, 0.0)

    x = points_xy[:, 0]
    y = points_xy[:, 1]
    l = distances_ltrb[:, 0]
    t = distances_ltrb[:, 1]
    r = distances_ltrb[:, 2]
    b = distances_ltrb[:, 3]

    x1 = x - l
    y1 = y - t
    x2 = x + r
    y2 = y + b

    boxes = torch.stack([x1, y1, x2, y2], dim=1)

    if max_shape is not None:
        h, w = int(max_shape[0]), int(max_shape[1])
        boxes[:, 0].clamp_(min=0.0, max=float(w - 1))
        boxes[:, 2].clamp_(min=0.0, max=float(w - 1))
        boxes[:, 1].clamp_(min=0.0, max=float(h - 1))
        boxes[:, 3].clamp_(min=0.0, max=float(h - 1))

    return boxes


def multi_apply(fn: Callable, *iterables: Iterable, **kwargs) -> Tuple[List[Any], ...]:
    results = [fn(*args, **kwargs) for args in zip(*iterables)]
    if not results:
        return ([],)
    first = results[0]
    if isinstance(first, tuple):
        num_out = len(first)
        transposed = tuple([list(x) for x in zip(*results)])
        assert len(transposed) == num_out
        return transposed
    else:
        return (results,)


def images_to_levels(targets: List[torch.Tensor], num_level_anchors: Sequence[int]) -> List[torch.Tensor]:
    assert isinstance(targets, (list, tuple)) and len(targets) > 0
    total_per_img = sum(num_level_anchors)
    for t in targets:
        assert t.shape[0] == total_per_img
    level_targets: List[torch.Tensor] = []
    start = 0
    for n in num_level_anchors:
        end = start + n
        level_t = torch.stack([t[start:end] for t in targets], dim=0)
        level_targets.append(level_t)
        start = end
    return level_targets


def overlay_bbox_cv(
    img: np.ndarray,
    dets: Dict[int, List[Sequence[float]]],
    class_names: Sequence[str],
    score_thresh: float = 0.3,
    box_color: Tuple[int, int, int] = (0, 255, 0),
    text_color: Tuple[int, int, int] = (255, 255, 255),
    thickness: int = 2,
) -> np.ndarray:
    assert img.dtype == np.uint8 and img.ndim == 3 and img.shape[2] == 3
    vis = cv2.cvtColor(img.copy(), cv2.COLOR_RGB2BGR)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    for cid, boxes in dets.items():
        if cid < 0 or cid >= len(class_names):
            cls_name = f"id{cid}"
        else:
            cls_name = class_names[cid]
        for b in boxes:
            if len(b) < 5:
                continue
            x1, y1, x2, y2, score = b[:5]
            if score < score_thresh:
                continue
            x1i, y1i, x2i, y2i = int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))
            cv2.rectangle(vis, (x1i, y1i), (x2i, y2i), box_color, thickness, lineType=cv2.LINE_AA)
            label = f"{cls_name}:{score:.2f}"
            (tw, th), baseline = cv2.getTextSize(label, font, font_scale, 1)
            cv2.rectangle(
                vis,
                (x1i, max(0, y1i - th - baseline - 2)),
                (x1i + tw + 2, y1i),
                box_color,
                -1,
            )
            cv2.putText(
                vis,
                label,
                (x1i + 1, y1i - baseline - 1),
                font,
                font_scale,
                text_color,
                1,
                cv2.LINE_AA,
            )
    vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
    return vis_rgb

import torch
from types import SimpleNamespace


class AssignResult(dict):
    def __init__(self, **kwargs):
        super().__init__(kwargs)
        for k, v in kwargs.items():
            setattr(self, k, v)
class CenterAssigner:
    def __init__(self, center_radius=None):
        if center_radius is None:
            center_radius = {}
        self.center_radius = {int(k): float(v) for k, v in center_radius.items()}

    def _extract_per_level_points(self, points_all):
        out = []
        if isinstance(points_all, torch.Tensor):
            if points_all.dim() == 3 and points_all.shape[0] == 1:
                points_all = [points_all[0]]
            elif points_all.dim() == 2:
                points_all = [points_all]
            else:
                raise ValueError("Unsupported tensor shape for points_all")
        if isinstance(points_all, (list, tuple)):
            for p in points_all:
                if isinstance(p, torch.Tensor):
                    if p.dim() == 3 and p.shape[0] > 1:
                        if p.shape[0] == 1:
                            out.append(p[0].detach())
                        else:
                            out.append(p[0].detach())
                    elif p.dim() == 2:
                        out.append(p.detach())
                    else:
                        raise ValueError("point tensor shape not supported")
                else:
                    arr = torch.tensor(p, dtype=torch.float32)
                    if arr.dim() == 2:
                        out.append(arr)
                    elif arr.dim() == 3 and arr.shape[0] == 1:
                        out.append(arr[0])
                    else:
                        raise ValueError("point array shape not supported")
            return out
        raise ValueError("Unsupported points_all type")

    def _make_center_priors(self, per_level_points, strides, device=None, dtype=torch.float32):
        priors = []
        for lvl_idx, pts in enumerate(per_level_points):
            s = int(strides[lvl_idx])
            pts = pts.to(device=device, dtype=dtype)
            x = pts[:, 0]
            y = pts[:, 1]
            stride_vec = x.new_full((x.shape[0],), s)
            pri = torch.stack([x, y, stride_vec, stride_vec], dim=-1)
            priors.append(pri)
        if len(priors) == 0:
            return torch.zeros((0, 4), dtype=dtype, device=device)
        return torch.cat(priors, dim=0)

    def build_targets(self, points_all, strides, target):
        per_level = self._extract_per_level_points(points_all)
        device = None
        dtype = torch.float32
        if isinstance(per_level[0], torch.Tensor):
            device = per_level[0].device
            dtype = per_level[0].dtype
        center_priors = self._make_center_priors(per_level, strides, device=device, dtype=dtype)
        gt_bboxes = target.get("boxes", None)
        gt_labels = target.get("labels", None)
        if gt_bboxes is None or (isinstance(gt_bboxes, (list, tuple)) and len(gt_bboxes) == 0):
            num_priors = center_priors.shape[0]
            labels = center_priors.new_full((num_priors,), -1, dtype=torch.long)
            label_weights = center_priors.new_zeros((num_priors,), dtype=torch.float)
            bbox_targets = center_priors.new_zeros((num_priors, 4), dtype=center_priors.dtype)
            dist_targets = center_priors.new_zeros((num_priors, 4), dtype=center_priors.dtype)
            num_pos = 0
            return labels, label_weights, bbox_targets, dist_targets, num_pos
        if isinstance(gt_bboxes, (list, tuple)):
            gt_bboxes = torch.tensor(gt_bboxes, dtype=center_priors.dtype, device=device)
        if isinstance(gt_labels, (list, tuple)):
            gt_labels = torch.tensor(gt_labels, dtype=torch.long, device=device)
        pri_x = center_priors[:, 0].unsqueeze(1)
        pri_y = center_priors[:, 1].unsqueeze(1)
        gt_centers_x = (gt_bboxes[:, 0] + gt_bboxes[:, 2]) / 2.0
        gt_centers_y = (gt_bboxes[:, 1] + gt_bboxes[:, 3]) / 2.0
        gt_cx = gt_centers_x.unsqueeze(0)
        gt_cy = gt_centers_y.unsqueeze(0)
        dx = torch.abs(pri_x - gt_cx)
        dy = torch.abs(pri_y - gt_cy)
        dcenter = torch.sqrt(dx * dx + dy * dy)
        stride_vec = center_priors[:, 2].unsqueeze(1)
        radii = torch.tensor([self.center_radius.get(int(s.item()), 2.5) for s in center_priors[:, 2]], device=device, dtype=center_priors.dtype)
        radii = radii.unsqueeze(1) * stride_vec
        inside_mask = dcenter <= radii
        num_priors = center_priors.shape[0]
        labels = center_priors.new_full((num_priors,), -1, dtype=torch.long)
        label_weights = center_priors.new_zeros((num_priors,), dtype=torch.float)
        bbox_targets = center_priors.new_zeros((num_priors, 4), dtype=center_priors.dtype)
        dist_targets = center_priors.new_zeros((num_priors, 4), dtype=center_priors.dtype)
        assigned_any = inside_mask.sum(dim=1) > 0
        if assigned_any.sum().item() > 0:
            assigned_idx = torch.argmax((inside_mask.to(dtype=torch.float32) * (1.0 / (dcenter + 1e-6))), dim=1)
            pri_inds = torch.nonzero(assigned_any, as_tuple=False).squeeze(1)
            sel_pris = pri_inds
            sel_gts = assigned_idx[assigned_any]
            labels[sel_pris] = gt_labels[sel_gts]
            label_weights[sel_pris] = 1.0
            gt_sel = gt_bboxes[sel_gts]
            bbox_targets[sel_pris, :] = gt_sel
            px = center_priors[sel_pris, 0]
            py = center_priors[sel_pris, 1]
            l = px - gt_sel[:, 0]
            t = py - gt_sel[:, 1]
            r = gt_sel[:, 2] - px
            b = gt_sel[:, 3] - py
            dist = torch.stack([l, t, r, b], dim=1)
            dist_targets[sel_pris, :] = dist
        num_pos = int((labels >= 0).sum().item())
        return labels, label_weights, bbox_targets, dist_targets, num_pos

    def _iou(self, boxes1, boxes2):
        lt = torch.max(boxes1[:, None, :2], boxes2[:, :2][None, ...])
        rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:][None, ...])
        wh = (rb - lt).clamp(min=0)
        inter = wh[:, :, 0] * wh[:, :, 1]
        area1 = ((boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1]))[:, None]
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        union = area1 + area2[None, :] - inter
        iou = inter / (union + 1e-7)
        return iou

   

    def assign(self, cls_preds, center_priors, decoded_bboxes, gt_bboxes, gt_labels, gt_bboxes_ignore=None):
        device = center_priors.device if isinstance(center_priors, torch.Tensor) else torch.device("cpu")
        dtype = center_priors.dtype if isinstance(center_priors, torch.Tensor) else torch.float32

        if isinstance(center_priors, (list, tuple)):
            center_priors = torch.cat([p.squeeze(0) if p.dim() == 3 and p.size(0) == 1 else p for p in center_priors], dim=0)

        num_priors = center_priors.shape[0]
        if gt_bboxes is None or (isinstance(gt_bboxes, (list, tuple)) and len(gt_bboxes) == 0):
            gt_inds = center_priors.new_zeros((num_priors,), dtype=torch.long)
            max_overlaps = center_priors.new_zeros((0,), dtype=dtype)
            labels = center_priors.new_full((num_priors,), -1, dtype=torch.long)
            label_scores = center_priors.new_zeros((num_priors,), dtype=dtype)
            label_weights = center_priors.new_zeros((num_priors,), dtype=dtype)
            bbox_targets = center_priors.new_zeros((num_priors, 4), dtype=dtype)
            dist_targets = center_priors.new_zeros((num_priors, 4), dtype=dtype)
            num_pos = 0
            return AssignResult(
                gt_inds=gt_inds,
                max_overlaps=max_overlaps,
                labels=labels,
                label_scores=label_scores,
                label_weights=label_weights,
                bbox_targets=bbox_targets,
                dist_targets=dist_targets,
                num_pos=num_pos,
                q_targets=label_scores,
            )

        if isinstance(gt_bboxes, (list, tuple, torch.Tensor)):
            gt_bboxes_t = torch.as_tensor(gt_bboxes, dtype=dtype, device=device)
        else:
            gt_bboxes_t = torch.empty((0, 4), dtype=dtype, device=device)

        if isinstance(gt_labels, (list, tuple, torch.Tensor)):
            gt_labels_t = torch.as_tensor(gt_labels, dtype=torch.long, device=device)
        else:
            gt_labels_t = torch.empty((0,), dtype=torch.long, device=device)

        labels_list, label_scores_list, label_weights_list, bbox_targets_list, dist_targets_list, num_pos = self.build_targets(
            [center_priors[:, :2]], [int(center_priors[0, 2].item())], {"boxes": gt_bboxes_t, "labels": gt_labels_t}
        )

        labels = labels_list if isinstance(labels_list, torch.Tensor) else torch.as_tensor(labels_list, dtype=torch.long, device=device)
        label_weights = label_weights_list if isinstance(label_weights_list, torch.Tensor) else torch.as_tensor(label_weights_list, dtype=dtype, device=device)
        bbox_targets = bbox_targets_list if isinstance(bbox_targets_list, torch.Tensor) else torch.as_tensor(bbox_targets_list, dtype=dtype, device=device)
        dist_targets = dist_targets_list if isinstance(dist_targets_list, torch.Tensor) else torch.as_tensor(dist_targets_list, dtype=dtype, device=device)

        assigned_pos = (labels >= 0) & (labels < 1e9)
        gt_inds = center_priors.new_zeros((num_priors,), dtype=torch.long)
        if assigned_pos.any():
            pos_idx = torch.nonzero(assigned_pos, as_tuple=False).squeeze(-1)
            assigned_gt_idx = labels[pos_idx]
            gt_inds[pos_idx] = assigned_gt_idx + 1
            if assigned_gt_idx.numel() > 0:
                sel_priors = center_priors[pos_idx][:, :4]
                sel_gts = gt_bboxes_t[assigned_gt_idx]
                a_x1, a_y1, a_x2, a_y2 = sel_priors[:, 0], sel_priors[:, 1], sel_priors[:, 0] + sel_priors[:, 2], sel_priors[:, 1] + sel_priors[:, 3]
                b_x1, b_y1, b_x2, b_y2 = sel_gts[:, 0], sel_gts[:, 1], sel_gts[:, 2], sel_gts[:, 3]
                inter_x1 = torch.max(a_x1, b_x1)
                inter_y1 = torch.max(a_y1, b_y1)
                inter_x2 = torch.min(a_x2, b_x2)
                inter_y2 = torch.min(a_y2, b_y2)
                inter_w = (inter_x2 - inter_x1).clamp(min=0)
                inter_h = (inter_y2 - inter_y1).clamp(min=0)
                inter = inter_w * inter_h
                area_a = (a_x2 - a_x1).clamp(min=0) * (a_y2 - a_y1).clamp(min=0)
                area_b = (b_x2 - b_x1).clamp(min=0) * (b_y2 - b_y1).clamp(min=0)
                union = area_a + area_b - inter
                ious = (inter / union).clamp(min=0) if union.numel() > 0 else torch.zeros((pos_idx.numel(),), dtype=dtype, device=device)
                max_overlaps = ious
            else:
                max_overlaps = center_priors.new_zeros((0,), dtype=dtype)
        else:
            max_overlaps = center_priors.new_zeros((0,), dtype=dtype)

        label_scores = center_priors.new_zeros((num_priors,), dtype=dtype)
        if assigned_pos.any() and max_overlaps.numel() > 0:
            pos_idx = torch.nonzero(assigned_pos, as_tuple=False).squeeze(-1)
            label_scores[pos_idx] = max_overlaps

        return AssignResult(
            gt_inds=gt_inds,
            max_overlaps=max_overlaps,
            labels=labels,
            label_scores=label_scores,
            label_weights=label_weights,
            bbox_targets=bbox_targets,
            dist_targets=dist_targets,
            num_pos=int(num_pos) if not isinstance(num_pos, torch.Tensor) else int(num_pos.item()),
            q_targets=label_scores,
        )


    def build_targets_for_batch(self, points_all, strides, targets):
        out = []
        for t in targets:
            out.append(self.build_targets(points_all, strides, t))
        return out
