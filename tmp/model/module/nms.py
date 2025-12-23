from tmp.src.common_imports import torch, Tuple, Dict, Optional

1
def box_area(boxes: torch.Tensor) -> torch.Tensor:
    return (boxes[..., 2] - boxes[..., 0]).clamp_(min=0) * (boxes[..., 3] - boxes[..., 1]).clamp_(min=0)


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    inter = (rb - lt).clamp(min=0).prod(2)
    return inter / (area1[:, None] + area2 - inter + 1e-7)


def nms_single_image(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    iou_thresh: float = 0.5,
    topk: int = 1000,
    max_det: int = 300,
) -> torch.Tensor:
    N = boxes.size(0)
    if N == 0:
        return boxes.new_zeros((0,), dtype=torch.long)

    if N > topk:
        scores, idx = torch.topk(scores, k=topk)
        boxes = boxes.index_select(0, idx)
        map_back = idx
    else:
        map_back = torch.arange(N, device=boxes.device)

    order = torch.argsort(scores, descending=True)
    boxes = boxes.index_select(0, order)
    scores = scores.index_select(0, order)
    map_back = map_back.index_select(0, order)

    keep = []
    while boxes.numel() > 0 and len(keep) < max_det:
        keep.append(int(map_back[0]))
        if boxes.size(0) == 1:
            break
        ious = box_iou(boxes[:1], boxes[1:]).squeeze(0)
        mask = ious <= iou_thresh
        boxes = boxes[1:][mask]
        scores = scores[1:][mask]
        map_back = map_back[1:][mask]

    if len(keep) == 0:
        return boxes.new_zeros((0,), dtype=torch.long)

    return torch.tensor(keep, device=map_back.device, dtype=torch.long)


def batched_nms_class_agnostic(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    iou_thresh: float = 0.5,
    conf_thresh: float = 0.25,
    topk: int = 1000,
    max_det: int = 300,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, N, _ = boxes.shape
    out_boxes, out_scores, out_bids = [], [], []

    for b in range(B):
        mask = scores[b] >= conf_thresh
        if mask.any():
            bb = boxes[b][mask]
            ss = scores[b][mask]
            keep = nms_single_image(bb, ss, iou_thresh, topk, max_det)
            if keep.numel() > 0:
                out_boxes.append(bb[keep])
                out_scores.append(ss[keep])
                out_bids.append(torch.full((keep.numel(),), b, dtype=torch.long, device=bb.device))

    if not out_boxes:
        d = boxes.device
        return (
            torch.zeros((0, 4), device=d),
            torch.zeros((0,), device=d),
            torch.zeros((0,), dtype=torch.long, device=d),
        )

    return torch.cat(out_boxes), torch.cat(out_scores), torch.cat(out_bids)


def nms_infer_wrapper(
    cls_logits: torch.Tensor,
    decoded_boxes: torch.Tensor,
    conf_thresh: float = 0.9,
    iou_thresh: float = 0.5,
    topk: int = 1000,
    max_det: int = 50,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    scores = torch.sigmoid(cls_logits.squeeze(-1))
    return batched_nms_class_agnostic(decoded_boxes, scores, iou_thresh, conf_thresh, topk, max_det)


def multiclass_nms_torchvision(
    multi_bboxes: torch.Tensor,
    multi_scores: torch.Tensor,
    score_thr: float,
    nms_cfg: Dict,
    max_num: int = -1,
    score_factors: Optional[torch.Tensor] = None,
):
    from torchvision.ops import nms

    num_classes = multi_scores.size(1) - 1
    if multi_bboxes.shape[1] > 4:
        bboxes = multi_bboxes.view(multi_scores.size(0), -1, 4)
    else:
        bboxes = multi_bboxes[:, None].expand(multi_scores.size(0), num_classes, 4)

    scores = multi_scores[:, :-1]
    valid = scores > score_thr
    bboxes = torch.masked_select(bboxes, valid[..., None]).view(-1, 4)

    if score_factors is not None:
        scores = scores * score_factors[:, None]
    scores = torch.masked_select(scores, valid)
    labels = valid.nonzero(as_tuple=False)[:, 1]

    if not bboxes.numel():
        d = multi_bboxes.new_zeros
        return d((0, 5)), d((0,), dtype=torch.long)

    cfg = dict(nms_cfg or {})
    iou_thr = float(cfg.get("iou_threshold", 0.5))
    class_agnostic = bool(cfg.get("class_agnostic", False))
    split_thr = int(cfg.get("split_thr", 10000))

    if not class_agnostic:
        max_coord = bboxes.max()
        offsets = labels.to(bboxes) * (max_coord + 1)
        boxes_for_nms = bboxes + offsets[:, None]
    else:
        boxes_for_nms = bboxes

    if bboxes.size(0) < split_thr:
        keep = nms(boxes_for_nms, scores, iou_thr)
        if max_num > 0:
            keep = keep[:max_num]
        dets = torch.cat([bboxes[keep], scores[keep, None]], dim=1)
        return dets, labels[keep]

    keep_mask = scores.new_zeros(scores.size(), dtype=torch.bool)
    for c in torch.unique(labels):
        mask = labels == c
        keep_c = nms(boxes_for_nms[mask], scores[mask], iou_thr)
        keep_mask[mask.nonzero()[keep_c]] = True

    keep = keep_mask.nonzero().squeeze(1)
    keep = keep[scores[keep].argsort(descending=True)]
    if max_num > 0:
        keep = keep[:max_num]
    dets = torch.cat([bboxes[keep], scores[keep, None]], 1)
    return dets, labels[keep]
