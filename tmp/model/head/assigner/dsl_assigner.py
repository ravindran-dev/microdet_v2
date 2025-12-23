from tmp.src.common_imports import *
from tmp.model.head.assigner.base_assigner import BaseAssigner
from tmp.model.head.assigner.assign_result import AssignResult


class DSLAssigner(BaseAssigner):
    def __init__(self, center_radius: Dict[int, float] = None, num_classes: int = 1):
        super().__init__()
        self.center_radius = center_radius or {8: 2.5, 16: 2.5, 32: 2.5, 64: 2.5}
        self.num_classes = int(num_classes)

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
        device = bbox_preds.device
        dtype = bbox_preds.dtype
        N = bbox_preds.size(0)
        G = gt_bboxes.size(0)

        points, strides_cat = self.make_level_points(
            featmap_sizes, strides, device, dtype=dtype
        )

        if G == 0:
            gt_inds = bbox_preds.new_full((N,), 0, dtype=torch.long)
            max_overlaps = bbox_preds.new_zeros((N,), dtype=torch.float32)
            labels = (
                bbox_preds.new_full((N,), 0, dtype=torch.long)
                if self.num_classes > 0
                else None
            )
            result = AssignResult(
                num_gts=0, gt_inds=gt_inds, max_overlaps=max_overlaps, labels=labels
            )
            pos_mask = gt_inds.new_zeros((N,), dtype=torch.bool)
            matched_boxes = bbox_preds.new_zeros((N, 4))
            return result, pos_mask, matched_boxes, strides_cat, points

        gt_cx = 0.5 * (gt_bboxes[:, 0] + gt_bboxes[:, 2])
        gt_cy = 0.5 * (gt_bboxes[:, 1] + gt_bboxes[:, 3])

        rad = torch.zeros((N,), device=device, dtype=dtype)
        for s, mult in self.center_radius.items():
            mask_s = strides_cat == float(s)
            rad[mask_s] = float(mult) * float(s)

        dx = torch.abs(points[:, None, 0] - gt_cx[None, :])
        dy = torch.abs(points[:, None, 1] - gt_cy[None, :])
        center_ok = (dx <= rad[:, None]) & (dy <= rad[:, None])

        ious = self.bbox_overlaps_iou(bbox_preds, gt_bboxes)
        ious_masked = torch.where(center_ok, ious, ious.new_full((), -1.0)).to(dtype)

        max_iou, gt_idx = ious_masked.max(dim=1)
        pos_mask = max_iou > 0

        gt_inds = gt_idx.new_zeros((N,), dtype=torch.long)
        gt_inds[pos_mask] = gt_idx[pos_mask] + 1

        max_overlaps = torch.zeros((N,), dtype=torch.float32, device=device)
        max_overlaps[pos_mask] = max_iou[pos_mask].float().clamp_(0.0, 1.0)

        if self.num_classes > 0:
            labels = torch.zeros((N,), dtype=torch.long, device=device)
            if pos_mask.any():
                labels[pos_mask] = gt_labels[gt_idx[pos_mask]].long()
        else:
            labels = None

        matched_boxes = torch.zeros((N, 4), dtype=bbox_preds.dtype, device=device)
        if pos_mask.any():
            matched_boxes[pos_mask] = gt_bboxes[gt_idx[pos_mask]]

        result = AssignResult(
            num_gts=G, gt_inds=gt_inds, max_overlaps=max_overlaps, labels=labels
        )
        return result, pos_mask, matched_boxes, strides_cat, points
