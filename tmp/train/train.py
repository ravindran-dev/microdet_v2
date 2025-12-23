#!/usr/bin/env python3
import os
import time
from pathlib import Path
import tomllib

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast

from tmp.data.coco_dataset import CocoDataset
from tmp.data.collate import coco_collate_fn
from tmp.data.transforms import build_transforms

from tmp.model.model_wrapper import MicroDet
from tmp.model.weight_averager.ema import ModelEMA
from tmp.model.loss.criterion import DetectionCriterion
from tmp.assigners.center_assigner import CenterAssigner
from tmp.train.validate import eval_model

from tmp.utils.logger import TBLogger, CSVLogger
from tmp.utils.profiler import profile_model_once
from tmp.utils.seed import set_seed
from tmp.train.pred_utils import normalize_preds

def build_dataloaders(cfg, workers: int, batch: int):
    train_cfg = cfg["data"]["train"]["config"]
    val_cfg = cfg["data"]["val"]["config"]

    train_img_dir, train_ann_path, train_input_size, train_keep_ratio, train_pipeline = train_cfg
    val_img_dir, val_ann_path, val_input_size, val_keep_ratio, val_pipeline = val_cfg

    tf_train = build_transforms(
        is_train=True,
        input_size=tuple(train_input_size),
        pipeline=train_pipeline,
        keep_ratio=bool(train_keep_ratio),
    )
    train_set = CocoDataset(
        img_dir=train_img_dir,
        ann_path=train_ann_path,
        transforms=tf_train,
        input_size=tuple(train_input_size),
        keep_ratio=bool(train_keep_ratio),
        class_names=cfg.get("class_names", ["drone"]),
    )

    tf_val = build_transforms(
        is_train=False,
        input_size=tuple(val_input_size),
        pipeline=val_pipeline,
        keep_ratio=bool(val_keep_ratio),
    )
    val_set = CocoDataset(
        img_dir=val_img_dir,
        ann_path=val_ann_path,
        transforms=tf_val,
        input_size=tuple(val_input_size),
        keep_ratio=bool(val_keep_ratio),
        class_names=cfg.get("class_names", ["drone"]),
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=coco_collate_fn,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=min(8, batch),
        shuffle=False,
        num_workers=max(1, workers // 2),
        pin_memory=True,
        drop_last=False,
        collate_fn=coco_collate_fn,
    )

    return train_loader, val_loader


def build_optimizer(model, opt_cfg_list):
    name, lr, wd, no_norm_decay, no_bias_decay = opt_cfg_list
    name = str(name).lower()
    lr = float(lr)
    wd = float(wd)
    no_norm_decay = bool(no_norm_decay)
    no_bias_decay = bool(no_bias_decay)

    if no_norm_decay:
        decay, no_decay = [], []
        for pname, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if any(k in pname.lower() for k in ["bias", "bn", "norm", "ln"]):
                no_decay.append(p)
            else:
                decay.append(p)
        params = [
            {"params": decay, "weight_decay": wd},
            {"params": no_decay, "weight_decay": 0.0},
        ]
    else:
        params = [
            {
                "params": [p for p in model.parameters() if p.requires_grad],
                "weight_decay": wd,
            }
        ]

    if name == "adamw":
        return optim.AdamW(params, lr=lr, weight_decay=wd)
    if name == "sgd":
        return optim.SGD(params, lr=lr, momentum=0.9, nesterov=True, weight_decay=wd)

    raise ValueError(f"Unsupported optimizer: {name}")


def build_scheduler(cfg, optimizer):
    lr_cfg = cfg["schedule"]["lr_schedule"]["config"]
    name, eta_min = lr_cfg
    name = str(name).lower()
    eta_min = float(eta_min)

    total_epochs = int(cfg["schedule"]["total_epochs"])

    if name == "cosineannealinglr":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_epochs, eta_min=eta_min
        )
    else:
        raise ValueError(f"Unsupported LR scheduler: {name}")

    warm_cfg = cfg["schedule"]["warmup"]
    warmup_steps = int(warm_cfg["steps"])
    warmup_ratio = float(warm_cfg["ratio"])

    return scheduler, warmup_steps, warmup_ratio


def _find_key_config(obj, key):
    if isinstance(obj, dict):
        if key in obj and isinstance(obj[key], dict) and "config" in obj[key]:
            return obj[key]["config"]
        if key in obj and isinstance(obj[key], list):
            return obj[key]
        for v in obj.values():
            found = _find_key_config(v, key)
            if found is not None:
                return found
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            found = _find_key_config(item, key)
            if found is not None:
                return found
    return None


def _find_loss_config(obj):
    return _find_key_config(obj, "loss")


def build_loss_cfg(model_cfg):
    loss_list = _find_loss_config(model_cfg)
    if loss_list is None:
        raise ValueError("Could not find 'head.loss.config' in model config (searched recursively).")
    if isinstance(loss_list, dict) and "config" in loss_list:
        loss_list = loss_list["config"]
    if not isinstance(loss_list, (list, tuple)) or len(loss_list) < 5:
        raise ValueError("Found loss config but its format is unexpected. Expected a list like [lambda_qfl, lambda_dfl, lambda_box, reg_max, 'iou_mode'].")
    lambda_qfl = float(loss_list[0])
    lambda_dfl = float(loss_list[1])
    lambda_box = float(loss_list[2])
    reg_max = int(loss_list[3])
    iou_mode = str(loss_list[4])
    return {
        "lambda_qfl": lambda_qfl,
        "lambda_dfl": lambda_dfl,
        "lambda_box": lambda_box,
        "reg_max": reg_max,
        "iou_mode": iou_mode,
    }


def build_assigner(model_cfg, strides):
    assigner_cfg = _find_key_config(model_cfg, "assigner_cfg")
    if assigner_cfg is None:
        assigner_cfg = _find_key_config(model_cfg, "assigner") or assigner_cfg
    if assigner_cfg is None:
        raise ValueError("Could not find assigner config in model config.")
    if isinstance(assigner_cfg, dict) and "config" in assigner_cfg:
        assigner_cfg = assigner_cfg["config"]
    if isinstance(assigner_cfg, (list, tuple)) and len(assigner_cfg) >= 2:
        name = str(assigner_cfg[0])
        radius_map = assigner_cfg[1]
    elif isinstance(assigner_cfg, dict):
        name = assigner_cfg.get("name", "CenterAssigner")
        radius_map = assigner_cfg.get("center_radius", {})
    else:
        raise ValueError("Unsupported assigner config format.")
    if name.lower() != "centerassigner" and "center" not in name.lower():
        raise ValueError(f"Only CenterAssigner-like assigner is supported, got {name}")
    center_radius = {int(k): float(v) for k, v in dict(radius_map).items()}
    for s in strides:
        if s not in center_radius:
            center_radius[s] = 2.5
    return CenterAssigner(center_radius=center_radius)


def save_checkpoint(path, model, optimizer, scheduler, ema, epoch, best_metric):
    print(f"[CKPT] Saving -> {path}")
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer else None,
        "scheduler": scheduler.state_dict() if scheduler else None,
        "ema": ema.state_dict() if ema else None,
        "epoch": epoch,
        "best_metric": best_metric,
    }
    torch.save(state, str(path))


def _normalize_preds_for_criterion(preds):
    if isinstance(preds, dict):
        if ("cls_logits" in preds and "reg_dfl" in preds) or ("cls" in preds and "reg" in preds):
            return preds
        if isinstance(preds.get("cls_logits"), (list, tuple)) and isinstance(preds.get("reg_dfl"), (list, tuple)):
            return preds
        return preds

    if isinstance(preds, (list, tuple)):
        if len(preds) == 2 and isinstance(preds[0], (list, tuple)) and isinstance(preds[1], (list, tuple)):
            cls_list = list(preds[0])
            reg_list = list(preds[1])
            if len(cls_list) == len(reg_list) and all(isinstance(c, torch.Tensor) for c in cls_list) and all(isinstance(r, torch.Tensor) for r in reg_list):
                return {"cls_logits": cls_list, "reg_dfl": reg_list}
        if all(isinstance(p, torch.Tensor) for p in preds) and len(preds) % 2 == 0:
            cls_list = preds[0::2]
            reg_list = preds[1::2]
            return {"cls_logits": list(cls_list), "reg_dfl": list(reg_list)}
        if all(isinstance(p, (list, tuple)) and len(p) == 2 and isinstance(p[0], torch.Tensor) and isinstance(p[1], torch.Tensor) for p in preds):
            return [{"cls": p[0], "reg": p[1]} for p in preds]
        if len(preds) == 2 and isinstance(preds[0], torch.Tensor) and isinstance(preds[1], torch.Tensor):
            return {"cls_logits": [preds[0]], "reg_dfl": [preds[1]]}
        if len(preds) == 1 and isinstance(preds[0], torch.Tensor):
            t = preds[0]
            dummy_reg = torch.zeros((t.size(0), 4, t.size(2), t.size(3)), dtype=t.dtype, device=t.device)
            return {"cls_logits": [t], "reg_dfl": [dummy_reg]}

        # If we reach here, it may be a tuple/list of tensors that look like feature maps (neck outputs),
        # not head outputs. Print a helpful diagnostic and raise a clear error.
        if all(isinstance(p, torch.Tensor) for p in preds):
            shapes = [tuple(p.shape) for p in preds]
            msg = (
                "Model returned a tuple/list of tensors that look like intermediate feature maps, not "
                "final head predictions. Shapes: " + str(shapes) + ".\n\n"
                "DetectionCriterion expects either:\n"
                "  (A) a dict {'cls_logits': [lvl1, lvl2, ...], 'reg_dfl': [lvl1, lvl2, ...]} where each lvl* is a tensor\n"
                "  (B) a list of per-level dicts [{'cls': tensor, 'reg': tensor}, ...]\n\n"
                "Likely causes:\n"
                "  - model_wrapper did not add/execute the head layer(s), returning neck outputs instead\n"
                "  - your model config (model.head) is malformed so parser skipped head\n\n"
                "Suggested checks:\n"
                "  1) print model (repr(model)) and inspect final layers (backbone/neck/head/detect) to confirm head exists\n"
                "     e.g. in a Python REPL: `from tmp.model.model_wrapper import MicroDet; m=MicroDet(cfg['model']); print(m)`\n"
                "  2) print the model cfg your wrapper received: `print(cfg['model'])` and ensure head is configured and parsed\n"
                "  3) inspect tmp/model/model_wrapper.py parsing code and ensure it constructs/instantiates head modules\n\n"
                "Quick debug print below (first few tensors):\n"
            )
            print(msg)
            for i, p in enumerate(preds[:6]):
                try:
                    print(f"  level {i}: dtype={p.dtype} shape={tuple(p.shape)} device={p.device}")
                except Exception:
                    print(f"  level {i}: <could not print shape>")
            raise RuntimeError("Model output is intermediate features — fix model to return head predictions (see printed diagnostic).")

    if isinstance(preds, torch.Tensor):
        t = preds
        dummy_reg = torch.zeros((t.size(0), 4, t.size(2), t.size(3)), dtype=t.dtype, device=t.device)
        return {"cls_logits": [t], "reg_dfl": [dummy_reg]}

    try:
        tinfo = (type(preds), getattr(preds, "__class__", None))
        print("[normalize_preds] Could not canonicalize preds automatically. type:", tinfo)
    except Exception:
        pass
    return preds



def main():
    import argparse

    parser = argparse.ArgumentParser("MicroDet drone training")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--load_model", type=str, default="")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--precision", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    cli_args = parser.parse_args()

    with open(cli_args.config, "rb") as f:
        cfg = tomllib.load(f)

    save_dir = Path(cfg["save"]["dir"])
    (save_dir / "weights").mkdir(parents=True, exist_ok=True)
    (save_dir / "logs").mkdir(parents=True, exist_ok=True)
    (save_dir / "eval").mkdir(parents=True, exist_ok=True)

    set_seed(42)
    use_cuda = torch.cuda.is_available() and cli_args.device == "cuda"
    device = torch.device("cuda" if use_cuda else "cpu")

    dev_workers, dev_batch, dev_precision = cfg["device"]["config"]
    workers = cli_args.workers if cli_args.workers is not None else int(dev_workers)
    batch = cli_args.batch if cli_args.batch is not None else int(dev_batch)
    precision = cli_args.precision if cli_args.precision is not None else int(dev_precision)

    amp_enabled = (precision == 16) and use_cuda

    model = MicroDet(cfg["model"])
    model = model.to(device)


    try:
        profile_model_once(model, input_shape=(1, 3, 640, 640))
    except Exception:
        pass

    optimizer = build_optimizer(model, cfg["schedule"]["optimizer"]["config"])
    scheduler, warmup_steps, warmup_ratio = build_scheduler(cfg, optimizer)

    ema = None
    if "weight_averager" in cfg.get("model", {}):
        ema = ModelEMA(
            model,
            decay=float(cfg["model"]["weight_averager"].get("decay", 0.9998)),
        )

    model_head = cfg["model"].get("head", [])
    head_cfgs = []
    if isinstance(model_head, dict) and "config" in model_head:
        head_cfgs = [model_head["config"]]
    elif isinstance(model_head, (list, tuple)):
        for item in model_head:
            if isinstance(item, dict) and "config" in item:
                head_cfgs.append(item["config"])
            else:
                head_cfgs.append(item)
    else:
        raise ValueError("Unsupported format for cfg['model']['head']")

    strides = None
    for cfg_item in head_cfgs:
        if isinstance(cfg_item, (list, tuple)) and len(cfg_item) >= 4:
            args = cfg_item[3]
            if isinstance(args, (list, tuple)) and len(args) >= 7:
                candidate = args[6]
                if isinstance(candidate, (list, tuple)):
                    strides = [int(x) for x in candidate]
                    break
    if strides is None:
        raise ValueError("Could not parse strides from model.head in config. Expected to find a list of strides as the 7th element of head args.")

    loss_cfg = build_loss_cfg(cfg["model"])
    assigner = build_assigner(cfg["model"], strides)

    criterion = DetectionCriterion(
        loss_cfg,
        strides=strides,
        assigner=assigner,
        reg_max=loss_cfg["reg_max"],
    ).to(device)

    train_loader, val_loader = build_dataloaders(cfg, workers=workers, batch=batch)

    if cli_args.load_model:
        state = torch.load(cli_args.load_model, map_location="cpu")
        sd = state["model"] if isinstance(state, dict) and "model" in state else state
        model.load_state_dict(sd, strict=False)

    start_epoch = 0
    if cli_args.resume and os.path.isfile(cli_args.resume):
        ckpt = torch.load(cli_args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model"], strict=True)
        if ckpt.get("optimizer"):
            optimizer.load_state_dict(ckpt["optimizer"])
        if ckpt.get("scheduler"):
            scheduler.load_state_dict(ckpt["scheduler"])
        if ema is not None and ckpt.get("ema"):
            ema.load_state_dict(ckpt["ema"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1

    tb = TBLogger(save_dir / "tb")
    csv = CSVLogger(save_dir / "logs" / "scalars.csv")

    scaler = torch.amp.GradScaler(enabled=amp_enabled)

    total_epochs = int(cfg["schedule"]["total_epochs"])
    val_interval = int(cfg["schedule"].get("val_interval", 5))
    best_map = -1.0
    global_step = start_epoch * len(train_loader)

    for epoch in range(start_epoch, total_epochs):
        model.train()

        detach_epoch = int(cfg["model"].get("detach_epoch", 0))
        freeze_epochs = min(5, max(0, detach_epoch))
        if epoch < freeze_epochs:
            for name, p in model.named_parameters():
                if "backbone" in name:
                    p.requires_grad = False
        else:
            for p in model.parameters():
                p.requires_grad = True

        epoch_loss = 0.0
        t0 = time.time()

        for it, batch_data in enumerate(train_loader):
            images, targets, _ = batch_data
            images = images.to(device, non_blocking=True)

            targets = [
                {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in t.items()}
                for t in targets
            ]

            if global_step < warmup_steps:
                warm = float(global_step + 1) / float(warmup_steps)
                base_lr = float(cfg["schedule"]["optimizer"]["config"][1])
                lr_now = max(warm * base_lr, base_lr * warmup_ratio)
                for pg in optimizer.param_groups:
                    pg["lr"] = lr_now

            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type="cuda", enabled=amp_enabled):
                raw_preds = model(images)
                
                preds = normalize_preds(raw_preds)

                loss_total, loss_qfl, loss_dfl, loss_box = criterion(preds, targets)

            scaler.scale(loss_total).backward()

            grad_clip = float(cfg["schedule"]["grad_clip"].get("max_norm", 0.0))
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            scaler.step(optimizer)
            scaler.update()

            if ema is not None:
                ema.update(model)

            epoch_loss += loss_total.item()
            global_step += 1

            if (it + 1) % int(cfg["log"]["interval"]) == 0:
                lr0 = optimizer.param_groups[0]["lr"]
                tb.add_scalar("train/loss_total", loss_total.item(), global_step)
                tb.add_scalar("train/loss_qfl", loss_qfl.item(), global_step)
                tb.add_scalar("train/loss_dfl", loss_dfl.item(), global_step)
                tb.add_scalar("train/loss_box", loss_box.item(), global_step)
                tb.add_scalar("train/lr", lr0, global_step)
                csv.write(
                    {
                        "step": global_step,
                        "epoch": epoch,
                        "loss_total": loss_total.item(),
                        "loss_qfl": loss_qfl.item(),
                        "loss_dfl": loss_dfl.item(),
                        "loss_box": loss_box.item(),
                        "lr": lr0,
                    }
                )

        scheduler.step()
        dt = time.time() - t0
        print(
            f"Epoch {epoch+1}/{total_epochs} | "
            f"loss {epoch_loss/len(train_loader):.4f} | "
            f"time {dt:.1f}s"
        )

        if (epoch + 1) % val_interval == 0 or (epoch + 1) == total_epochs:
            eval_model_to_use = ema.ema if ema is not None else model
            eval_model_to_use.eval()

            with torch.no_grad():
                metrics = eval_model(
                    eval_model_to_use,
                    val_loader,
                    post_cfg={
                        "conf_thres": 0.90,
                        "iou_thres": 0.50,
                        "max_det": 50,
                        "min_box": 12,
                    },
                )

            mAP = float(metrics.get("mAP", -1.0))
            print(f"Val epoch {epoch+1}: mAP={mAP:.4f}")

            save_checkpoint(
                save_dir / "weights" / "last.ckpt",
                model,
                optimizer,
                scheduler,
                ema,
                epoch,
                best_map,
            )
            print(
                f"Epoch {epoch+1}/{total_epochs} | "
                f"loss {epoch_loss/len(train_loader):.4f} | "
                f"time {dt:.1f}s"
            )

            if mAP > best_map:
                best_map = mAP
                save_checkpoint(
                    save_dir / "weights" / "best.ckpt",
                    model,
                    optimizer,
                    scheduler,
                    ema,
                    epoch,
                    best_map,
                )
                if ema is not None:
                    torch.save(
                        {
                            "model": eval_model_to_use.state_dict(),
                            "epoch": epoch,
                            "best_metric": best_map,
                        },
                        save_dir / "weights" / "best_ema.ckpt",
                    )

    print("Training complete.")
    tb.close()
    csv.close()


if __name__ == "__main__":
    main()
