import torch
import torch.nn as nn
from tmp.model.module.conv import ConvModule, DepthwiseConvModule
from tmp.model.registry import register_module


@register_module
class GhostBlocks(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expand: float = 1.0,
        kernel_size: int = 5,
        num_blocks: int = 1,
        use_res: bool = False,
        act: str = "LeakyReLU",
    ):
        super().__init__()
        self.use_res = use_res
        if use_res:
            self.shortcut = ConvModule(in_channels, out_channels, 1, 1, 0, activation=act)

        blocks = []
        mid_channels = max(1, int(out_channels * expand))
        for _ in range(num_blocks):
            blocks.append(
                nn.Sequential(
                    ConvModule(in_channels, mid_channels, kernel_size=1, stride=1, padding=0, activation=act),
                    DepthwiseConvModule(mid_channels, mid_channels, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, activation=act),
                    ConvModule(mid_channels, out_channels, kernel_size=1, stride=1, padding=0, activation=act),
                )
            )
            in_channels = out_channels
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.blocks(x)
        return out + self.shortcut(x) if self.use_res else out


@register_module
class GhostPAN(nn.Module):
    def __init__(
        self,
        in_channels: list,
        out_channels: int,
        use_depthwise: bool = False,
        kernel_size: int = 5,
        expand: float = 1.0,
        num_blocks: int = 1,
        use_res: bool = False,
        num_extra_level: int = 0,
        upsample_mode: str = "bilinear",
        activation: str = "LeakyReLU",
    ):
        super().__init__()
        self.in_channels = list(in_channels)
        self.out_channels = out_channels
        Conv = DepthwiseConvModule if use_depthwise else ConvModule

        self.upsample = nn.Upsample(scale_factor=2, mode=upsample_mode, align_corners=False)
        self.reduce_layers = nn.ModuleList(
            [ConvModule(c, out_channels, 1, 1, 0, activation=activation) for c in self.in_channels]
        )

        num_pairs = max(0, len(self.in_channels) - 1)
        self.top_down_blocks = nn.ModuleList(
            [
                GhostBlocks(out_channels * 2, out_channels, expand, kernel_size, num_blocks, use_res, activation)
                for _ in range(num_pairs)
            ]
        )

        self.downsamples = nn.ModuleList(
            [Conv(out_channels, out_channels, kernel_size, stride=2, activation=activation) for _ in range(num_pairs)]
        )

        self.bottom_up_blocks = nn.ModuleList(
            [
                GhostBlocks(out_channels * 2, out_channels, expand, kernel_size, num_blocks, use_res, activation)
                for _ in range(num_pairs)
            ]
        )

        self.extra_in = nn.ModuleList()
        self.extra_out = nn.ModuleList()
        for _ in range(num_extra_level):
            self.extra_in.append(Conv(out_channels, out_channels, kernel_size, stride=2, activation=activation))
            self.extra_out.append(Conv(out_channels, out_channels, kernel_size, stride=2, activation=activation))

    def forward(self, feats: tuple) -> tuple:
        feats = [r(f) for f, r in zip(feats, self.reduce_layers)]

        inner = [feats[-1]]
        for i in range(len(self.in_channels) - 1, 0, -1):
            up = self.upsample(inner[0])
            merged = torch.cat([up, feats[i - 1]], dim=1)
            idx = len(self.in_channels) - 1 - i
            inner.insert(0, self.top_down_blocks[idx](merged))

        outs = [inner[0]]
        for i in range(len(self.in_channels) - 1):
            down = self.downsamples[i](outs[-1])
            merged = torch.cat([down, inner[i + 1]], dim=1)
            outs.append(self.bottom_up_blocks[i](merged))

        for in_conv, out_conv in zip(self.extra_in, self.extra_out):
            outs.append(in_conv(feats[-1]) + out_conv(outs[-1]))

        return tuple(outs)
