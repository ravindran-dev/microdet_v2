from tmp.src.common_imports import *

_NORMS = (GroupNorm, LayerNorm, _BatchNorm)

def build_optimizer(model: nn.Module, config: Dict[str, Any]) -> torch.optim.Optimizer:
    cfg = copy.deepcopy(config)
    logger = logging.getLogger("NanoDet")
    name = cfg.pop("name")
    optim_cls = getattr(torch.optim, name)

    no_norm_decay = bool(cfg.pop("no_norm_decay", False))
    no_bias_decay = bool(cfg.pop("no_bias_decay", False))
    param_level_cfg: Dict[str, Dict[str, float]] = cfg.pop("param_level_cfg", {})

    base_lr = cfg.get("lr", None)
    base_wd = cfg.get("weight_decay", None)

    overrides: Dict[nn.Parameter, Dict[str, float]] = {}

    for pname, p in model.named_parameters():
        if not p.requires_grad:
            continue
        overrides[p] = {"__name__": pname}

        for key, rules in param_level_cfg.items():
            if key in pname:
                if "lr_mult" in rules and base_lr is not None:
                    overrides[p]["lr"] = float(base_lr) * float(rules["lr_mult"])
                if "decay_mult" in rules and base_wd is not None:
                    overrides[p]["weight_decay"] = float(base_wd) * float(rules["decay_mult"])
                break

    if no_norm_decay:
        for _, m in model.named_modules():
            if isinstance(m, _NORMS):
                if hasattr(m, "weight") and m.weight in overrides:
                    overrides[m.weight]["weight_decay"] = 0.0
                if hasattr(m, "bias") and m.bias in overrides:
                    overrides[m.bias]["weight_decay"] = 0.0

    if no_bias_decay:
        for m in model.modules():
            if hasattr(m, "bias") and isinstance(m.bias, nn.Parameter) and m.bias in overrides:
                overrides[m.bias]["weight_decay"] = 0.0

    param_groups: List[Dict[str, Any]] = []
    for p, over in overrides.items():
        name_tag = over.pop("__name__", None)
        g = {"params": [p]}
        if "lr" in over:
            g["lr"] = over["lr"]
        if "weight_decay" in over:
            g["weight_decay"] = over["weight_decay"]
        if len(g) > 1:
            logger.info(f"special optimizer hyperparameter: {name_tag} - { {k:v for k,v in g.items() if k!='params'} }")
        param_groups.append(g)

    return optim_cls(param_groups, **cfg)
