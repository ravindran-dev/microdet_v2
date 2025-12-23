from tmp.src.common_imports import *
from tmp.model.loss import giou
from tmp.model.module import nms
from .assign_result import AssignResult
from .base_assigner import BaseAssigner


def bbox_overlaps(b1: torch.Tensor, b2: torch.Tensor) -> torch.Tensor:
    if hasattr(bbox_overlaps, "__call__"):
        return bbox_overlaps(b1, b2)
    N, M = b1.size(0), b2.size(0)
    if N == 0 or M == 0:
        return b1.new_zeros((N, M))
    x1 = torch.max(b1[:, None, 0], b2[None, :, 0])
    y1 = torch.max(b1[:, None, 1], b2[None, :, 1])
    x2 = torch.min(b1[:, None, 2], b2[None, :, 2])
    y2 = torch.min(b1[:, None, 3], b2[None, :, 3])
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    a1 = (b1[:, 2] - b1[:, 0]).clamp(min=0) * (b1[:, 3] - b1[:, 1]).clamp(min=0)
    a2 = (b2[:, 2] - b2[:, 0]).clamp(min=0) * (b2[:, 3] - b2[:, 1]).clamp(min=0)
    union = a1[:, None] + a2[None, :] - inter + 1e-9
    return inter / union


class ATSSAssigner(BaseAssigner):
    def __init__(self, topk: int = 9, ignore_iof_thr: float = -1.0):
        super().__init__()
        self.topk = int(topk)
        self.ignore_iof_thr = float(ignore_iof_thr)

    @torch.no_grad()
    def assign(
        self,
        bboxes: torch.Tensor,
        num_level_bboxes: List[int],
        gt_bboxes: torch.Tensor,
        gt_bboxes_ignore: Optional[torch.Tensor] = None,
        gt_labels: Optional[torch.Tensor] = None,
    ) -> AssignResult:
        device = bboxes.device
        INF = 1e8
        bboxes = bboxes[:, :4]
        num_gt = int(gt_bboxes.size(0))
        num_bboxes = int(bboxes.size(0))
        overlaps = bbox_overlaps(bboxes, gt_bboxes)
        assigned_gt_inds = overlaps.new_full((num_bboxes,), 0, dtype=torch.long)

        if num_gt == 0 or num_bboxes == 0:
            max_overlaps = overlaps.new_zeros((num_bboxes,))
            assigned_labels = None if gt_labels is None else overlaps.new_full((num_bboxes,), -1, dtype=torch.long)
            return AssignResult(num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels)

        gt_cx = (gt_bboxes[:, 0] + gt_bboxes[:, 2]) * 0.5
        gt_cy = (gt_bboxes[:, 1] + gt_bboxes[:, 3]) * 0.5
        gt_pts = torch.stack([gt_cx, gt_cy], dim=1)
        bcx = (bboxes[:, 0] + bboxes[:, 2]) * 0.5
        bcy = (bboxes[:, 1] + bboxes[:, 3]) * 0.5
        cand_pts = torch.stack([bcx, bcy], dim=1)
        distances = torch.cdist(cand_pts, gt_pts, p=2)

        if self.ignore_iof_thr > 0 and gt_bboxes_ignore is not None and gt_bboxes_ignore.numel() > 0:
            ignore_overlaps = self._bbox_overlaps_iof(bboxes, gt_bboxes_ignore)
            ignore_max, _ = ignore_overlaps.max(dim=1)
            ignore_mask = ignore_max > self.ignore_iof_thr
            distances[ignore_mask, :] = INF
            assigned_gt_inds[ignore_mask] = -1

        candidate_idxs = []
        start = 0
        for lvl_count in num_level_bboxes:
            end = start + int(lvl_count)
            d_lvl = distances[start:end, :]
            k = min(self.topk, d_lvl.size(0))
            _, topk_idx = d_lvl.topk(k, dim=0, largest=False)
            candidate_idxs.append(topk_idx + start)
            start = end
        candidate_idxs = torch.cat(candidate_idxs, dim=0)

        cand_overlaps = overlaps[candidate_idxs, torch.arange(num_gt, device=device)]
        thr = cand_overlaps.mean(dim=0) + cand_overlaps.std(dim=0)
        is_pos = cand_overlaps >= thr[None, :]

        flat_idx = (
            candidate_idxs
            + (torch.arange(num_gt, device=device) * num_bboxes)[None, :]
        ).reshape(-1)

        bcx_rep = bcx.view(1, -1).expand(num_gt, num_bboxes).reshape(-1)
        bcy_rep = bcy.view(1, -1).expand(num_gt, num_bboxes).reshape(-1)

        l = bcx_rep[flat_idx].view(-1, num_gt) - gt_bboxes[:, 0]
        t = bcy_rep[flat_idx].view(-1, num_gt) - gt_bboxes[:, 1]
        r = gt_bboxes[:, 2] - bcx_rep[flat_idx].view(-1, num_gt)
        b = gt_bboxes[:, 3] - bcy_rep[flat_idx].view(-1, num_gt)
        center_in_gt = torch.stack([l, t, r, b], dim=2).amin(dim=2) > 0.01

        is_pos = is_pos & center_in_gt

        overlaps_inf = overlaps.new_full((num_bboxes, num_gt), -INF)
        sel = is_pos.view(-1)
        if sel.any():
            overlaps_flat = overlaps.t().contiguous().view(-1)
            overlaps_inf_flat = overlaps_inf.t().contiguous().view(-1)
            overlaps_inf_flat[flat_idx[sel]] = overlaps_flat[flat_idx[sel]]
            overlaps_inf = overlaps_inf_flat.view(num_gt, num_bboxes).t()

        max_overlaps, argmax_overlaps = overlaps_inf.max(dim=1)
        pos_mask = max_overlaps > -INF
        assigned_gt_inds[pos_mask] = argmax_overlaps[pos_mask] + 1

        if gt_labels is not None:
            assigned_labels = assigned_gt_inds.new_full((num_bboxes,), -1)
            pos_inds = (assigned_gt_inds > 0).nonzero(as_tuple=False).squeeze(1)
            if pos_inds.numel() > 0:
                assigned_labels[pos_inds] = gt_labels[assigned_gt_inds[pos_inds] - 1]
        else:
            assigned_labels = None

        return AssignResult(num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels)

    @staticmethod
    def _bbox_overlaps_iof(b1: torch.Tensor, b2: torch.Tensor) -> torch.Tensor:
        N, M = b1.size(0), b2.size(0)
        if N == 0 or M == 0:
            return b1.new_zeros((N, M))
        x1 = torch.max(b1[:, None, 0], b2[None, :, 0])
        y1 = torch.max(b1[:, None, 1], b2[None, :, 1])
        x2 = torch.min(b1[:, None, 2], b2[None, :, 2])
        y2 = torch.min(b1[:, None, 3], b2[None, :, 3])
        inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
        area_b1 = (b1[:, 2] - b1[:, 0]).clamp(min=0) * (b1[:, 3] - b1[:, 1]).clamp(min=0)
        return inter / (area_b1[:, None] + 1e-9)
