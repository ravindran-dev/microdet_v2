import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import tomllib

from src.data.transforms import _letterbox_any
from src.model.model_wrapper import MicroDet
from src.model.postprocess import canonicalize_preds, decode_predictions, undo_letterbox


def preprocess_image(image_path: str, input_size=(640, 640)):
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_lb, lb_meta = _letterbox_any(img_rgb, input_size)
    img_lb = img_lb.astype(np.float32) / 255.0
    img_t = torch.from_numpy(np.transpose(img_lb, (2, 0, 1))).unsqueeze(0).float()

    return img_t, img_bgr, lb_meta


@torch.no_grad()
def run_inference(args):
    with open(args.config, "rb") as f:
        cfg = tomllib.load(f)

    input_size = tuple(cfg.get("model", {}).get("input_size", [640, 640]))
    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu")

    model = MicroDet(cfg["model"]).to(device).eval()
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt.get("model", ckpt), strict=False)

    img_t, img_bgr, lb_meta = preprocess_image(args.image, input_size=input_size)
    img_t = img_t.to(device)

    preds = canonicalize_preds(model(img_t))
    det_boxes, det_scores, det_batch = decode_predictions(
        preds,
        model=model,
        conf_thres=float(args.conf_thres),
        iou_thres=float(args.iou_thres),
        max_det=int(args.max_det),
        min_box=float(args.min_box),
        score_temperature=float(args.score_temperature),
    )

    if det_boxes.numel() == 0:
        print("No detections.")
        cv2.imwrite(args.output, img_bgr)
        return

    keep = det_batch == 0
    boxes = undo_letterbox(det_boxes[keep], lb_meta)
    scores = det_scores[keep]

    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = [int(v) for v in box.tolist()]
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            img_bgr,
            f"person {float(score):.2f}",
            (x1, max(0, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.output, img_bgr)
    print(f"Saved: {args.output} | detections={boxes.shape[0]}")


def parse_args():
    parser = argparse.ArgumentParser("MicroDet inference")
    parser.add_argument("--config", type=str, default="microdet.toml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--output", type=str, default="tmp/infer/output.png")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conf-thres", dest="conf_thres", type=float, default=0.35)
    parser.add_argument("--iou-thres", dest="iou_thres", type=float, default=0.5)
    parser.add_argument("--max-det", dest="max_det", type=int, default=100)
    parser.add_argument("--min-box", dest="min_box", type=float, default=8.0)
    parser.add_argument("--score-temperature", dest="score_temperature", type=float, default=1.0)
    return parser.parse_args()


if __name__ == "__main__":
    run_inference(parse_args())
