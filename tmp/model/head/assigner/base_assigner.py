from abc import ABCMeta, abstractmethod
from typing import List, Tuple

from tmp.src.common_imports import *
import torch


class BaseAssigner(metaclass=ABCMeta):
    @abstractmethod
    @torch.no_grad()
    def assign(
        self,
        cls_scores: torch.Tensor,
        bbox_preds: torch.Tensor,
        gt_bboxes: torch.Tensor,
        gt_labels: torch.Tensor,
        featmap_sizes: List[Tuple[int, int]],
        strides: List[int],
        img_size: Tuple[int, int],
    ):
        pass

    @staticmethod
    def make_level_points(featmap_sizes, strides, device, dtype=torch.float32):
        pts_all = []
        s_all = []
        for (h, w), s in zip(featmap_sizes, strides):
            ys, xs = torch.meshgrid(
                torch.arange(h, device=device),
                torch.arange(w, device=device),
                indexing="ij",
            )
            xs = (xs + 0.5) * s
            ys = (ys + 0.5) * s
            pts = torch.stack([xs.reshape(-1), ys.reshape(-1)], dim=1).to(dtype)
            pts_all.append(pts)
            s_all.append(torch.full((pts.size(0),), s, device=device, dtype=dtype))
        points = torch.cat(pts_all, dim=0)
        strides_cat = torch.cat(s_all, dim=0)
        return points, strides_cat

    @staticmethod
    def bbox_overlaps_iou(b1: torch.Tensor, b2: torch.Tensor, eps=1e-9):
        if b1.numel() == 0 or b2.numel() == 0:
            return b1.new_zeros((b1.size(0), b2.size(0)))
        x1 = torch.max(b1[:, None, 0], b2[None, :, 0])
        y1 = torch.max(b1[:, None, 1], b2[None, :, 1])
        x2 = torch.min(b1[:, None, 2], b2[None, :, 2])
        y2 = torch.min(b1[:, None, 3], b2[None, :, 3])
        inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
        a1 = (b1[:, 2] - b1[:, 0]).clamp(min=0) * (b1[:, 3] - b1[:, 1]).clamp(min=0)
        a2 = (b2[:, 2] - b2[:, 0]).clamp(min=0) * (b2[:, 3] - b2[:, 1]).clamp(min=0)
        union = a1[:, None] + a2[None, :] - inter + eps
        return inter / union
