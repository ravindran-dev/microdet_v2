# src/model/module/activation.py
from tmp.src.common_imports  import nn


# Activation factory
_ACT = {
    "relu":         lambda: nn.ReLU(inplace=True),
    "leakyrelu":    lambda: nn.LeakyReLU(0.1, inplace=True),
    "relu6":        lambda: nn.ReLU6(inplace=True),
    "selu":         lambda: nn.SELU(inplace=True),
    "elu":          lambda: nn.ELU(inplace=True),
    "gelu":         lambda: nn.GELU(),
    "prelu":        lambda: nn.PReLU(),
    "silu":         lambda: nn.SiLU(inplace=True),
    "swish":        lambda: nn.SiLU(inplace=True),      # alias
    "hardswish":    lambda: nn.Hardswish(inplace=True),
    None:           lambda: nn.Identity(),
}


def act_layers(name: str | None):
    """
    Returns activation layer instance by name.
    Example:
        act_layers("LeakyReLU")
    """
    if name is None:
        return nn.Identity()

    key = name.lower()
    if key not in _ACT:
        raise ValueError(f"Unsupported activation: {name}")
    return _ACT[key]()
