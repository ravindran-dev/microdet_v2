import torch
import torch.nn.functional as F
import cv2
import tomllib
import numpy as np
from torchvision.ops import nms

from tmp.model.model_wrapper import MicroDet
from tmp.train.train import _normalize_preds_for_criterion


# ================= CONFIG =================
CFG_PATH = "microdet.toml"
CKPT_PATH = "runs/microdet_drone/weights/last.ckpt"
IMAGE_PATH = "tmp/infer/image.png"
OUT_PATH = "tmp/infer/output.png"

CONF_THRES = 0.30          # IMPORTANT (after retraining)
NMS_IOU_THRES = 0.5
STRIDES = [8, 16, 32]      # must match training
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ==========================================


def preprocess(img_path, size=(640, 640)):
    img0 = cv2.imread(img_path)
    assert img0 is not None, "Image not found"

    h0, w0 = img0.shape[:2]

    img = cv2.resize(img0, size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype("float32") / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)

    return img, img0, (w0, h0), size


def dfl_decode(reg, reg_max):
    """
    reg: (N, 4*(reg_max+1))
    return: (N, 4)
    """
    reg = reg.reshape(-1, 4, reg_max + 1)
    prob = F.softmax(reg, dim=2)
    proj = torch.arange(reg_max + 1, device=reg.device).float()
    return (prob * proj).sum(dim=2)


def decode_level(cls, reg, stride, input_size, orig_size):
    """
    cls: (1, C, H, W)
    reg: (1, 4*K, H, W)
    """
    B, C, H, W = cls.shape
    K = reg.shape[1] // 4

    cls = cls.sigmoid().view(C, -1)
    scores, labels = cls.max(dim=0)

    keep = scores > CONF_THRES
    if keep.sum() == 0:
        return None

    scores = scores[keep]
    labels = labels[keep]

    reg = reg.permute(0, 2, 3, 1).reshape(-1, 4 * K)[keep]
    dist = dfl_decode(reg, K - 1)

    # ---- IMPORTANT FIX: center + 0.5 ----
    ys, xs = torch.meshgrid(
        torch.arange(H, device=reg.device),
        torch.arange(W, device=reg.device),
        indexing="ij",
    )
    centers = torch.stack((xs + 0.5, ys + 0.5), dim=-1).reshape(-1, 2)
    centers = centers[keep] * stride

    x1 = centers[:, 0] - dist[:, 0] * stride
    y1 = centers[:, 1] - dist[:, 1] * stride
    x2 = centers[:, 0] + dist[:, 2] * stride
    y2 = centers[:, 1] + dist[:, 3] * stride

    boxes = torch.stack([x1, y1, x2, y2], dim=1)

    # ---- scale back to original image ----
    iw, ih = input_size
    ow, oh = orig_size

    scale_x = ow / iw
    scale_y = oh / ih

    boxes[:, [0, 2]] *= scale_x
    boxes[:, [1, 3]] *= scale_y

    boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, ow)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, oh)

    return boxes, scores, labels


@torch.no_grad()
def main():
    with open(CFG_PATH, "rb") as f:
        cfg = tomllib.load(f)

    model = MicroDet(cfg["model"]).to(DEVICE).eval()

    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=False)
    print("✔ Loaded weights")

    img_tensor, img0, orig_size, input_size = preprocess(IMAGE_PATH)
    img_tensor = img_tensor.to(DEVICE)

    raw_preds = model(img_tensor)
    preds = _normalize_preds_for_criterion(raw_preds)

    all_boxes = []
    all_scores = []

    for lvl, level_pred in enumerate(preds):
        decoded = decode_level(
            level_pred["cls"],
            level_pred["reg"],
            STRIDES[lvl],
            input_size,
            orig_size,
        )
        if decoded is None:
            continue

        boxes, scores, _ = decoded
        all_boxes.append(boxes)
        all_scores.append(scores)

    if len(all_boxes) == 0:
        print("⚠ No detections")
        cv2.imwrite(OUT_PATH, img0)
        return

    boxes = torch.cat(all_boxes)
    scores = torch.cat(all_scores)

    keep = nms(boxes, scores, NMS_IOU_THRES)
    boxes = boxes[keep]
    scores = scores[keep]

    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = map(int, box.tolist())
        cv2.rectangle(img0, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            img0,
            f"{score:.2f}",
            (x1, max(0, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    cv2.imwrite(OUT_PATH, img0)
    print(f"✔ Output saved to: {OUT_PATH}")


if __name__ == "__main__":
    main()
