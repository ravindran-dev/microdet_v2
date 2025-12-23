from tmp.src.common_imports  import torch, nn


def get_act(name: str = "LeakyReLU") -> nn.Module:
    n = (name or "LeakyReLU").lower()
    if n == "relu":
        return nn.ReLU(inplace=True)
    if n in ("silu", "swish"):
        return nn.SiLU(inplace=True)
    if n in ("leakyrelu", "leaky"):
        return nn.LeakyReLU(0.1, inplace=True)
    return nn.Identity()


class IdentityNorm(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


def get_norm(norm: str, num_features: int) -> nn.Module:
    n = (norm or "BN").upper()
    if n == "BN":
        return nn.BatchNorm2d(num_features)
    if n == "GN":
        g = 32 if num_features % 32 == 0 else max(1, num_features // 4)
        return nn.GroupNorm(g, num_features)
    if n in ("IN", "INSTANCE"):
        return nn.InstanceNorm2d(num_features, affine=True)
    if n in ("NONE", "ID"):
        return IdentityNorm()
    return nn.BatchNorm2d(num_features)


class ConvNormAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        k: int = 1,
        s: int = 1,
        p: int | None = None,
        g: int = 1,
        bias: bool = False,
        norm: str = "BN",
        act: str = "LeakyReLU",
    ):
        super().__init__()
        p = k // 2 if p is None else p
        self.conv = nn.Conv2d(in_channels, out_channels, k, s, p, groups=g, bias=bias)
        self.norm = get_norm(norm, out_channels)
        self.act = get_act(act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))
