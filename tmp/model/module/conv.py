# src/model/module/conv.py
import torch
import torch.nn as nn
from tmp.model.registry import register_module



def make_activation(name: str | None = "LeakyReLU") -> nn.Module:
    if not name:
        return nn.Identity()
    name = name.lower()
    act_map = {
        "relu":        lambda: nn.ReLU(inplace=True),
        "relu6":       lambda: nn.ReLU6(inplace=True),
        "leakyrelu":   lambda: nn.LeakyReLU(0.1, inplace=True),
        "leaky":       lambda: nn.LeakyReLU(0.1, inplace=True),
        "silu":        lambda: nn.SiLU(inplace=True),
        "swish":       lambda: nn.SiLU(inplace=True),
        "gelu":        lambda: nn.GELU(),
        "prelu":       lambda: nn.PReLU(),
        "selu":        lambda: nn.SELU(inplace=True),
        "elu":         lambda: nn.ELU(inplace=True),
        "hardswish":   lambda: nn.Hardswish(inplace=True),
        "identity":    lambda: nn.Identity(),
        "none":        lambda: nn.Identity(),
    }
    return act_map.get(name, lambda: nn.LeakyReLU(0.1, inplace=True))()


@register_module
class ConvModule(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int | None = None,
        groups: int = 1,
        bias: bool = False,
        norm: str = "BN",
        activation: str = "LeakyReLU",
    ):
        super().__init__()
        padding = kernel_size // 2 if padding is None else padding

        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size, stride, padding,
            groups=groups, bias=bias
        )
        self.bn = nn.BatchNorm2d(out_channels) if norm and norm.upper() == "BN" else nn.Identity()
        self.act = make_activation(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


@register_module
class DepthwiseConvModule(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int | None = None,
        bias: bool = False,
        norm: str = "BN",
        activation: str = "LeakyReLU",
    ):
        super().__init__()
        padding = kernel_size // 2 if padding is None else padding

        self.dw = nn.Conv2d(
            in_channels, in_channels,
            kernel_size, stride, padding,
            groups=in_channels, bias=bias,
        )
        self.dw_bn = nn.BatchNorm2d(in_channels) if norm.upper() == "BN" else nn.Identity()
        self.dw_act = make_activation(activation)

        self.pw = nn.Conv2d(in_channels, out_channels, 1, bias=bias)
        self.pw_bn = nn.BatchNorm2d(out_channels) if norm.upper() == "BN" else nn.Identity()
        self.pw_act = make_activation(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dw_act(self.dw_bn(self.dw(x)))
        x = self.pw_act(self.pw_bn(self.pw(x)))
        return x


@register_module
class SeparableConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int | None = None,
        activation: str = "LeakyReLU",
        bias: bool = False,
        norm: str = "BN",
    ):
        super().__init__()
        padding = kernel_size // 2 if padding is None else padding

        self.dw = nn.Conv2d(
            in_channels, in_channels,
            kernel_size, stride, padding,
            groups=in_channels, bias=bias
        )
        self.dw_bn = nn.BatchNorm2d(in_channels) if norm.upper() == "BN" else nn.Identity()
        self.dw_act = make_activation(activation)
        self.pw = nn.Conv2d(in_channels, out_channels, 1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dw_act(self.dw_bn(self.dw(x)))
        return self.pw(x)



def kaiming_init_module(module: nn.Module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
