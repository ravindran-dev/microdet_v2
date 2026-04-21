import torch

from src.model.postprocess import canonicalize_preds, decode_predictions


@torch.no_grad()
def _box_iou_xyxy(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if a.numel() == 0 or b.numel() == 0:
        return a.new_zeros((a.size(0), b.size(0)))
    lt = torch.maximum(a[:, None, :2], b[None, :, :2])
    rb = torch.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    area_a = (a[:, 2] - a[:, 0]).clamp(min=0) * (a[:, 3] - a[:, 1]).clamp(min=0)
    area_b = (b[:, 2] - b[:, 0]).clamp(min=0) * (b[:, 3] - b[:, 1]).clamp(min=0)
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-6)


@torch.no_grad()
def eval_model(model, val_loader, post_cfg):
    device = next(model.parameters()).device

    conf_thres = float(post_cfg.get("conf_thres", 0.35))
    iou_thres = float(post_cfg.get("iou_thres", 0.50))
    max_det = int(post_cfg.get("max_det", 100))
    min_box = float(post_cfg.get("min_box", 8.0))
    score_temperature = float(post_cfg.get("score_temperature", 1.0))

    tp = 0
    fp = 0
    fn = 0
    detections = 0
    score_sum = 0.0
    n_images = 0

    for images, targets, _ in val_loader:
        images = images.to(device)
        preds = canonicalize_preds(model(images))

        det_boxes, det_scores, det_batch = decode_predictions(
            preds,
            model=model,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            max_det=max_det,
            min_box=min_box,
            score_temperature=score_temperature,
        )

        B = images.size(0)
        n_images += B
        for bi in range(B):
            gt = targets[bi].get("boxes", torch.zeros((0, 4), dtype=torch.float32))
            gt = gt.to(device).float()

            mask = det_batch == bi
            pb = det_boxes[mask]
            ps = det_scores[mask]

            detections += int(pb.size(0))
            score_sum += float(ps.sum().item()) if ps.numel() > 0 else 0.0

            if gt.numel() == 0 and pb.numel() == 0:
                continue
            if gt.numel() == 0:
                fp += int(pb.size(0))
                continue
            if pb.numel() == 0:
                fn += int(gt.size(0))
                continue

            ious = _box_iou_xyxy(pb, gt)
            matched_gt = torch.zeros((gt.size(0),), dtype=torch.bool, device=device)

            order = torch.argsort(ps, descending=True)
            for pi in order:
                pi = int(pi)
                max_iou, gidx = ious[pi].max(dim=0)
                gidx = int(gidx)
                if max_iou >= 0.5 and not matched_gt[gidx]:
                    matched_gt[gidx] = True
                    tp += 1
                else:
                    fp += 1

            fn += int((~matched_gt).sum().item())

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = (2 * precision * recall) / max(precision + recall, 1e-12)
    mean_conf = score_sum / max(detections, 1)
    det_per_img = detections / max(n_images, 1)

    # Keep mAP key for checkpoint selection; AP50 proxy is better than a hardcoded zero.
    ap50_proxy = precision * recall

    return {
        "mAP": float(ap50_proxy),
        "AP50": float(ap50_proxy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "mean_conf": float(mean_conf),
        "det_per_img": float(det_per_img),
    }
