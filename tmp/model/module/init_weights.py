# src/model/module/init_weights.py
from tmp.src.common_imports import math, nn


def kaiming_init(module: nn.Module, a=0.0, mode="fan_out", nonlinearity="relu",
                 bias=0.0, dist="normal"):
    if hasattr(module, "weight") and module.weight is not None:
        fn = nn.init.kaiming_uniform_ if dist == "uniform" else nn.init.kaiming_normal_
        fn(module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def xavier_init(module: nn.Module, gain=1.0, bias=0.0, dist="normal"):
    if hasattr(module, "weight") and module.weight is not None:
        fn = nn.init.xavier_uniform_ if dist == "uniform" else nn.init.xavier_normal_
        fn(module.weight, gain=gain)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def normal_init(module: nn.Module, mean=0.0, std=1.0, bias=0.0):
    if hasattr(module, "weight") and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def bn_init(module: nn.BatchNorm2d, w=1.0, b=0.0, eps=1e-5, momentum=0.1):
    nn.init.constant_(module.weight, w)
    nn.init.constant_(module.bias, b)
    module.eps = eps
    module.momentum = momentum


def conv_module_init(module: nn.Module, nonlinearity="relu"):
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            kaiming_init(m, nonlinearity=nonlinearity)
        elif isinstance(m, nn.BatchNorm2d):
            bn_init(m)


def dw_pw_init(dw: nn.Conv2d, pw: nn.Conv2d, nonlinearity="relu"):
    kaiming_init(dw, nonlinearity=nonlinearity)
    kaiming_init(pw, nonlinearity=nonlinearity)


def linear_init(module: nn.Linear, bias=0.0):
    xavier_init(module, gain=1.0, bias=bias)


def head_conv_init(module: nn.Module):
    for m in module.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            normal_init(m, std=0.01)


def set_cls_prior_bias(conv: nn.Conv2d, prior=0.01):
    if conv.bias is not None:
        val = -math.log((1 - prior) / max(prior, 1e-8))
        nn.init.constant_(conv.bias, val)


def initialize_model(model: nn.Module, cls_prior=0.01, nonlinearity="relu"):
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            kaiming_init(m, nonlinearity=nonlinearity)
        elif isinstance(m, nn.BatchNorm2d):
            bn_init(m)
        elif isinstance(m, nn.Linear):
            linear_init(m)

        if isinstance(m, nn.Conv2d) and ("cls" in name or "class" in name):
            set_cls_prior_bias(m, cls_prior)
